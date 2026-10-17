from collections import deque
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import av
import numpy as np
from fastapi.testclient import TestClient

import backend.app as server
from backend.denoise import AudioError, _decode_frames, read_audio_bytes, write_wav_bytes
from backend.speech import SpeechDenoiser
from scripts.evaluate_denoising import evaluate
from scripts.evaluate_quality_matrix import make_mixture


def m4a_fixture(rate=16000, channels=1, seconds=2, mode="faststart", codec="aac"):
    samples = (.05 * np.sin(2 * np.pi * 440 * np.arange(round(rate * seconds)) / rate)).astype(np.float32)
    options = {"movflags": "+faststart"} if mode == "faststart" else (
        {"movflags": "frag_keyframe+empty_moov"} if mode == "fragmented" else {})
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.m4a"
        with av.open(str(path), "w", format="mp4", options=options) as container:
            stream = container.add_stream(codec, rate=rate)
            stream.layout = "mono" if channels == 1 else "stereo"
            frame = av.AudioFrame.from_ndarray(np.tile(samples, (channels, 1)), format="fltp", layout=stream.layout)
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return path.read_bytes()


def truncate_at_packet(payload):
    with av.open(io.BytesIO(payload), format="mov") as container:
        positions = [packet.pos for packet in container.demux(container.streams.audio[0])
                     if packet.pos is not None and packet.size]
    return payload[:positions[len(positions) // 2]]


def resize_mdat(payload, to_eof=False):
    data = bytearray(payload)
    offset = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        if kind == b"mdat":
            struct.pack_into(">I", data, offset, 0 if to_eof else len(data) - offset)
            return bytes(data)
        offset += size
    raise AssertionError("Synthetic fixture has no media box")


class AudioIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = m4a_fixture()
        cls.truncated = truncate_at_packet(cls.complete)

    def test_packet_boundary_truncation_is_rejected(self):
        with self.assertRaises(AudioError):
            read_audio_bytes(self.truncated, "truncated.m4a")

    def test_duration_guard_rejects_truncation_with_resized_media_box(self):
        for eof in (False, True):
            with self.subTest(to_eof=eof), self.assertRaises(AudioError):
                read_audio_bytes(resize_mdat(self.truncated, eof), "truncated.m4a")

    def test_complete_m4a_variants_and_codec_padding_are_accepted(self):
        cases = [(8000, 1, .01, "faststart"), (16000, 2, .1, "normal"),
                 (44100, 1, .2, "faststart"), (48000, 2, .2, "faststart"),
                 (16000, 1, .2, "fragmented")]
        for rate, channels, seconds, mode in cases:
            with self.subTest(rate=rate, channels=channels, seconds=seconds, mode=mode):
                audio = read_audio_bytes(m4a_fixture(rate, channels, seconds, mode), "complete.m4a")
                self.assertEqual(audio.sample_rate, rate)
                self.assertEqual(audio.channels, channels)
                self.assertTrue(np.isfinite(audio.samples).all())
                self.assertGreater(len(audio.samples), 0)
        audio = read_audio_bytes(resize_mdat(self.complete, True), "size-zero.m4a")
        self.assertGreater(len(audio.samples), 30000)

    def test_lossless_m4a_and_extended_box_sizes_are_accepted(self):
        audio = read_audio_bytes(m4a_fixture(codec="alac"), "lossless.m4a")
        self.assertEqual(audio.samples.shape, (32000, 1))
        extended_free = struct.pack(">I4sQ", 1, b"free", 16)
        audio = read_audio_bytes(self.complete + extended_free, "extended.m4a")
        self.assertGreater(len(audio.samples), 30000)

    def test_corrupt_packet_is_not_sent_to_decoder(self):
        valid, flush, corrupt = Mock(is_corrupt=False), Mock(is_corrupt=False), Mock(is_corrupt=True)
        valid.decode.return_value = ["samples"]
        flush.decode.return_value = ["tail"]
        container = Mock()
        container.demux.return_value = iter((valid, flush))
        self.assertEqual(list(_decode_frames(container, object())), ["samples", "tail"])
        container.demux.return_value = iter((corrupt,))
        with self.assertRaises(AudioError):
            list(_decode_frames(container, object()))
        corrupt.decode.assert_not_called()

    def test_malformed_container_boundaries_are_rejected(self):
        malformed = [self.complete + b"\0", struct.pack(">I4s", 1, b"ftyp"),
                     struct.pack(">I4sQ", 1, b"ftyp", 15), struct.pack(">I4s", 4, b"ftyp")]
        for payload in malformed:
            with self.subTest(size=len(payload)), self.assertRaises(AudioError):
                read_audio_bytes(payload, "malformed.m4a")

    def test_truncated_http_upload_does_not_save_a_partial_recording(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "MEDIA_DIR", Path(directory)), \
                TestClient(server.app, headers={"origin": "http://testserver"}) as client:
            client.get("/api/session").raise_for_status()
            for payload in (self.truncated, resize_mdat(self.truncated)):
                response = client.post("/api/denoise", files={"file": ("truncated.m4a", payload)})
                self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(client.get("/api/recordings").json()["recordings"], [])
            response = client.post("/api/denoise", files={"file": ("complete.m4a", self.complete)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.delete("/api/recordings").status_code, 204)
            self.assertEqual(server.app.state.file_slots._value, 1)
            self.assertFalse(server.app.state.store.active_owners)

    def test_nonfinite_model_output_is_rejected_before_mixing(self):
        for value in (np.nan, np.inf, -np.inf):
            frame = av.AudioFrame.from_ndarray(np.full((1, 480), value, np.float32), format="fltp", layout="mono")
            reducer = SpeechDenoiser.__new__(SpeechDenoiser)
            reducer.channels, reducer.skip, reducer.pending_offset, reducer.emitted = 1, 0, 0, 0
            reducer.pending = deque([(np.full((480, 1), .2, np.float32), .85)])
            reducer.sink = Mock()
            reducer.sink.pull.side_effect = [frame, av.error.BlockingIOError(11, "would block")]
            with self.subTest(value=value), self.assertRaises(AudioError):
                reducer._pull()

    def test_wrong_model_output_channel_shape_is_rejected(self):
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 480), np.float32), format="fltp", layout="mono")
        reducer = SpeechDenoiser.__new__(SpeechDenoiser)
        reducer.channels, reducer.skip, reducer.pending_offset, reducer.emitted = 2, 0, 0, 0
        reducer.pending = deque([(np.full((480, 2), .2, np.float32), .85)])
        reducer.sink = Mock()
        reducer.sink.pull.side_effect = [frame, av.error.BlockingIOError(11, "would block")]
        with self.assertRaises(AudioError):
            reducer._pull()

    def test_live_model_failure_releases_capacity_and_does_not_save_audio(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "MEDIA_DIR", Path(directory)), \
                TestClient(server.app, headers={"origin": "http://testserver"}) as client:
            client.get("/api/session").raise_for_status()
            with patch("backend.speech.SpeechDenoiser._pull", side_effect=AudioError("Invalid model output")), \
                    client.websocket_connect("/api/live", headers={"origin": "http://testserver"}) as socket:
                socket.send_json({"type": "start", "sampleRate": 16000})
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_bytes(np.zeros(4000, dtype="<f4").tobytes())
                self.assertEqual(socket.receive_json()["type"], "error")
            self.assertEqual(client.get("/api/recordings").json()["recordings"], [])
            self.assertFalse(server.app.state.live_owners)
            self.assertFalse(server.app.state.store.active_owners)
            self.assertEqual(server.app.state.live_slots._value, 2)

    def test_encoder_rejects_nonfinite_values_without_cast_warnings(self):
        for value in (np.nan, np.inf, -np.inf):
            samples = np.zeros((17000, 2), np.float32)
            samples[-1, 1] = value
            with self.subTest(value=value), warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaises(AudioError):
                    write_wav_bytes(samples, 16000)

    def test_silent_evaluation_reference_fails_cleanly_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silence.wav"
            path.write_bytes(write_wav_bytes(np.zeros((16000, 1), np.float32), 16000))
            with warnings.catch_warnings(), patch("scripts.evaluate_denoising.reduce_noise") as inference:
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "silent"):
                    evaluate(path, Path(directory) / "metrics")
                inference.assert_not_called()
            self.assertFalse((Path(directory) / "metrics").exists())

    def test_invalid_mixture_inputs_fail_cleanly(self):
        reference = np.full(10, .1)
        noise = np.full(10, .01)
        active = np.ones(10, dtype=bool)
        cases = [(reference, np.zeros(10), active, 5), (np.zeros(10), noise, active, 5),
                 (reference, noise, np.zeros(10, dtype=bool), 5), (reference, noise[:-1], active, 5),
                 (reference, np.full(10, np.nan), active, 5), (reference, noise, active, np.inf),
                 (reference, noise, active, 10000), (reference, noise, active, -10000)]
        for clean, background, mask, snr in cases:
            with self.subTest(snr=snr), warnings.catch_warnings(), np.errstate(all="raise"):
                warnings.simplefilter("error", RuntimeWarning)
                with self.assertRaises(ValueError):
                    make_mixture(clean, background, mask, snr)


if __name__ == "__main__":
    unittest.main()

import io
from pathlib import Path
import tempfile
import unittest
import wave
from unittest.mock import Mock, patch

import numpy as np

from backend.denoise import AudioError, reduce_noise, write_wav_bytes
from backend.recording import WavSpool


def reference_wav(samples, rate):
    if samples.ndim == 1:
        samples = samples[:, None]
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(samples.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes((np.clip(samples, -1, 1) * 32767.0).astype("<i2").tobytes())
    return output.getvalue()


class RecordingTests(unittest.TestCase):
    def test_chunked_quantization_preserves_pcm_and_headers(self):
        for dtype in (np.float32, np.float64):
            for channels in (1, 2):
                with self.subTest(dtype=dtype, channels=channels):
                    samples = np.random.default_rng(51).normal(0, .7, (40001, channels)).astype(dtype)
                    expected = reference_wav(samples, 48000)
                    self.assertEqual(write_wav_bytes(samples, 48000), expected)
                    with WavSpool(48000, channels) as recording:
                        for start in range(0, len(samples), 127):
                            recording.write(samples[start:start + 127])
                        self.assertEqual(recording.frames, len(samples))
                        self.assertEqual(recording.read_wav(), expected)
                        self.assertEqual(recording.read_wav(), expected)

    def test_quantization_scratch_is_bounded(self):
        samples = np.zeros((48000 * 3, 2), np.float32)
        with patch("backend.denoise.np.clip", wraps=np.clip) as clipped:
            write_wav_bytes(samples, 48000)
        self.assertGreater(clipped.call_count, 1)
        self.assertTrue(all(len(call.args[0]) <= 16384 for call in clipped.call_args_list))

    def test_preallocated_output_preserves_incremental_order_and_tail(self):
        samples = np.zeros((16003, 1), np.float32)
        expected = np.random.default_rng(42).normal(0, .1, samples.shape).astype(np.float32)
        reducer = Mock()
        reducer.process.side_effect = [expected[:15993], expected[15993:15996]]
        reducer.finish.return_value = expected[15996:]
        with patch("backend.speech.SpeechDenoiser", return_value=reducer):
            actual = reduce_noise(samples, 8000)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(reducer.process.call_count, 2)

    def test_incomplete_preallocated_output_is_never_returned(self):
        reducer = Mock()
        reducer.process.return_value = np.zeros((4, 1), np.float32)
        reducer.finish.return_value = np.zeros((0, 1), np.float32)
        with patch("backend.speech.SpeechDenoiser", return_value=reducer), self.assertRaises(AudioError):
            reduce_noise(np.zeros((5, 1), np.float32), 8000)

    def test_disk_recording_closes_on_error_and_cannot_append_after_finish(self):
        recording = WavSpool(16000)
        with recording:
            recording.write(np.ones(81, np.float32) * .1)
            self.assertTrue(recording.read_wav().startswith(b"RIFF"))
            with self.assertRaises(AudioError):
                recording.write(np.zeros(1, np.float32))
        self.assertTrue(recording.file.closed)
        recording.close()
        recording = WavSpool(16000)
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with recording:
                raise RuntimeError("stop")
        self.assertTrue(recording.file.closed)

    def test_live_disconnect_closes_disk_buffers_and_releases_capacity(self):
        from fastapi.testclient import TestClient
        import backend.app as server

        buffers = []

        def create_buffer(*args):
            buffer = WavSpool(*args)
            buffers.append(buffer)
            return buffer

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, "MEDIA_DIR", Path(directory)), \
                patch.object(server, "WavSpool", side_effect=create_buffer), \
                TestClient(server.app, headers={"origin": "http://testserver"}) as client:
            client.get("/api/session").raise_for_status()
            with client.websocket_connect("/api/live", headers={"origin": "http://testserver"}) as socket:
                socket.send_json({"type": "start", "sampleRate": 8000})
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_bytes(np.zeros(2000, dtype="<f4").tobytes())
                self.assertIsInstance(socket.receive_bytes(), bytes)
            self.assertEqual(len(buffers), 2)
            self.assertTrue(all(buffer.file.closed for buffer in buffers))
            self.assertFalse(server.app.state.live_owners)
            self.assertFalse(server.app.state.store.active_owners)
            self.assertEqual(server.app.state.live_slots._value, 2)

    def test_disk_recording_rejects_invalid_shape_values_and_duration(self):
        with WavSpool(8000) as recording:
            for samples in (np.zeros((10, 2), np.float32), np.array([np.nan]), np.zeros((2, 2, 2))):
                with self.subTest(shape=samples.shape), self.assertRaises(AudioError):
                    recording.write(samples)
            self.assertEqual(recording.frames, 0)
            recording.frames = 8000 * 120
            with self.assertRaises(AudioError):
                recording.write(np.zeros(1, np.float32))


if __name__ == "__main__":
    unittest.main()

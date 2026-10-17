import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import backend.app as server
from backend.denoise import AudioError, StreamingNoiseReducer, read_audio_bytes, reduce_noise, write_wav_bytes


class HumTests(unittest.TestCase):
    def test_targeted_hum_is_suppressed_without_only_lowering_master_gain(self):
        rate = 16000
        t = np.arange(rate * 3) / rate
        for frequency in (50, 60):
            tone = (0.05 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
            plain = reduce_noise(tone, rate, 1)[:, 0]
            filtered = reduce_noise(tone, rate, 1, str(frequency))[:, 0]
            self.assertLess(np.mean(filtered[rate:] ** 2), np.mean(plain[rate:] ** 2) / 100)

    def test_zero_strength_is_exact_even_with_hum_enabled(self):
        x = np.random.default_rng(11).normal(0, 0.03, (4003, 2)).astype(np.float32)
        for hum in ("off", "50", "60"):
            np.testing.assert_array_equal(reduce_noise(x, 16000, 0, hum), x)

    def test_hum_stream_matches_file_and_keeps_tail_at_multiple_rates(self):
        for rate in (8000, 16000, 44100, 48000):
            x = np.random.default_rng(rate).normal(0, 0.04, rate + 127).astype(np.float32)
            for hum in ("50", "60"):
                reducer = StreamingNoiseReducer(rate, hum)
                chunks = [reducer.process(x[i:i + 127], 0.85) for i in range(0, len(x), 127)]
                actual = np.concatenate([*chunks, reducer.finish()])
                np.testing.assert_allclose(actual, reduce_noise(x, rate, 0.85, hum)[:, 0], atol=1e-6, rtol=0)

    def test_stereo_and_short_frames_are_finite(self):
        for rate in (8000, 11025, 16000, 22050, 44100, 48000):
            for length in (1, 127, rate // 4 + 17):
                x = np.random.default_rng(length).normal(0, 0.02, (length, 2)).astype(np.float32)
                output = reduce_noise(x, rate, 0.85, "60")
                self.assertEqual(output.shape, x.shape)
                self.assertTrue(np.all(np.isfinite(output)))

    def test_invalid_filter_options_cannot_become_ffmpeg_options(self):
        for hum in ("auto", "60:width=1000", "../model", 60, None):
            with self.assertRaises(AudioError):
                reduce_noise(np.zeros(100), 16000, 0, hum)

    def test_default_is_balanced_not_full_suppression(self):
        x = np.random.default_rng(9).normal(0, 0.03, 8000).astype(np.float32)
        np.testing.assert_array_equal(reduce_noise(x, 16000), reduce_noise(x, 16000, 0.85))


class HumApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.media = patch.object(server, "MEDIA_DIR", Path(self.directory.name))
        self.media.start()
        self.client = TestClient(server.app, headers={"origin": "http://testserver"})
        self.client.__enter__()
        self.client.get("/api/session").raise_for_status()
        self.samples = np.random.default_rng(4).normal(0, 0.02, 4000).astype(np.float32)
        self.wav = write_wav_bytes(self.samples, 16000)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.media.stop()
        self.directory.cleanup()

    def test_upload_uses_hum_setting_and_persists_it(self):
        response = self.client.post("/api/denoise", files={"file": ("x.wav", self.wav)}, data={"humFilter": "60"})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual((result["strength"], result["humFilter"]), (0.85, "60"))
        actual = read_audio_bytes(self.client.get(result["denoisedUrl"]).content, "result.wav").samples
        original = read_audio_bytes(self.wav, "x.wav").samples
        expected = read_audio_bytes(write_wav_bytes(reduce_noise(original, 16000, .85, "60"), 16000), "expected.wav").samples
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(self.client.get("/api/recordings").json()["recordings"][0]["humFilter"], "60")

    def test_invalid_and_duplicate_settings_are_rejected(self):
        for value in ("auto", "60:width=99", "../model"):
            response = self.client.post("/api/denoise", files={"file": ("x.wav", self.wav)}, data={"humFilter": value})
            self.assertEqual(response.status_code, 400)
            with self.client.websocket_connect("/api/live") as socket:
                socket.send_json({"type": "start", "sampleRate": 16000, "humFilter": value})
                self.assertEqual(socket.receive_json()["type"], "error")
        response = self.client.post("/api/denoise", files=[("file", ("x.wav", self.wav)),
                                    ("humFilter", (None, "50")), ("humFilter", (None, "60"))])
        self.assertEqual(response.status_code, 400)

    def test_live_saves_the_selected_filter_and_matches_local_inference(self):
        with self.client.websocket_connect("/api/live") as socket:
            socket.send_json({"type": "start", "sampleRate": 16000, "humFilter": "50"})
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_bytes(self.samples.astype("<f4").tobytes())
            chunks = [np.frombuffer(socket.receive_bytes(), dtype="<f4")]
            socket.send_json({"type": "stop"})
            while True:
                message = socket.receive()
                if message.get("bytes") is not None:
                    chunks.append(np.frombuffer(message["bytes"], dtype="<f4"))
                else:
                    result = json.loads(message["text"])
                    break
        self.assertEqual(result["humFilter"], "50")
        np.testing.assert_allclose(np.concatenate(chunks), reduce_noise(self.samples, 16000, .85, "50")[:, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()

import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend.app as server
from backend.denoise import read_audio_bytes, reduce_noise, write_wav_bytes


def compressed_fixture(extension, channels=1):
    rate = 16000
    samples = (0.25 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
    target = io.BytesIO()
    with av.open(target, mode="w", format="mp3" if extension == "mp3" else "mp4") as output:
        stream = output.add_stream("mp3" if extension == "mp3" else "aac", rate=rate)
        stream.layout = "mono" if channels == 1 else "stereo"
        frame = av.AudioFrame.from_ndarray(np.tile(samples, (channels, 1)), format="fltp", layout=stream.layout)
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return target.getvalue()


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.media_patch = patch.object(server, "MEDIA_DIR", Path(self.directory.name))
        self.media_patch.start()
        self.client = TestClient(server.app, headers={"origin": "http://testserver"})
        self.client.__enter__()
        self.assertEqual(self.client.get("/api/session").status_code, 200)

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.media_patch.stop()
        self.directory.cleanup()

    def test_file_uploads_wav_mp3_m4a_and_downloads(self):
        signal = np.random.default_rng(0).normal(0, 0.03, (16000, 1)).astype(np.float32)
        fixtures = {
            "wav": write_wav_bytes(signal, 16000),
            "mp3": compressed_fixture("mp3"),
            "m4a": compressed_fixture("m4a", channels=2),
        }
        for extension, payload in fixtures.items():
            with self.subTest(extension=extension):
                response = self.client.post("/api/denoise", files={"file": ("test." + extension, payload)}, data={"strength": "0.82"})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["inputFormat"], extension)
                self.assertEqual(result["engine"], "rnnoise")
                self.assertGreater(result["expiresAt"], time.time())
                original = self.client.get(result["originalUrl"])
                cleaned = self.client.get(result["denoisedUrl"])
                self.assertEqual(original.status_code, 200)
                self.assertEqual(cleaned.status_code, 200)
                raw_audio = read_audio_bytes(original.content, "original.wav")
                clean_audio = read_audio_bytes(cleaned.content, "cleaned.wav")
                self.assertEqual(raw_audio.samples.shape, clean_audio.samples.shape)
                self.assertEqual(raw_audio.channels, 2 if extension == "m4a" else 1)
                self.assertTrue(np.all(np.isfinite(clean_audio.samples)))

    def test_zero_strength_preserves_audio_across_block_boundaries(self):
        samples = np.random.default_rng(1).normal(0, 0.1, (70000, 2)).astype(np.float32)
        np.testing.assert_array_equal(reduce_noise(samples, 16000, 0), samples)
        cleaned = reduce_noise(samples, 16000, 0.82)
        self.assertEqual(cleaned.shape, samples.shape)
        self.assertLess(np.mean(cleaned ** 2), np.mean(samples ** 2))

    def test_rejects_empty_corrupt_unsupported_and_long_files(self):
        fixtures = [
            ("empty.wav", write_wav_bytes(np.empty((0, 1)), 16000)),
            ("broken.wav", b"RIFF"),
            ("broken.mp3", b"not audio"),
            ("test.txt", b"not audio"),
            ("long.wav", write_wav_bytes(np.zeros((8000 * 121, 1)), 8000)),
        ]
        for name, payload in fixtures:
            with self.subTest(name=name):
                response = self.client.post("/api/denoise", files={"file": (name, payload)})
                self.assertEqual(response.status_code, 400, response.text)

    def test_nonfinite_strength_and_oversize_upload_are_rejected(self):
        payload = write_wav_bytes(np.zeros((160, 1)), 16000)
        for strength in ["nan", "inf", "-0.1", "1.1"]:
            response = self.client.post("/api/denoise", files={"file": ("x.wav", payload)}, data={"strength": strength})
            self.assertIn(response.status_code, [400, 422])
        with patch.object(server, "MAX_UPLOAD_BYTES", 100):
            response = self.client.post("/api/denoise", files={"file": ("x.wav", payload)})
            self.assertEqual(response.status_code, 413)

    def test_stream_returns_audio_before_stop_and_saves_exact_stream(self):
        samples = np.random.default_rng(3).normal(0, 0.04, 4000).astype("<f4")
        with self.client.websocket_connect("/api/live") as socket:
            socket.send_json({"type": "start", "sampleRate": 16000, "strength": 0})
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_bytes(samples.tobytes())
            first = np.frombuffer(socket.receive_bytes(), dtype="<f4")
            np.testing.assert_array_equal(first, samples[:len(first)])
            self.assertGreater(len(first), 0)
            socket.send_json({"type": "strength", "value": 0.82})
            socket.send_bytes(samples.tobytes())
            cleaned = np.frombuffer(socket.receive_bytes(), dtype="<f4")
            self.assertLess(np.mean(cleaned ** 2), np.mean(samples ** 2))
            tail = samples[:127]
            socket.send_bytes(tail.tobytes())
            clean_tail = np.frombuffer(socket.receive_bytes(), dtype="<f4")
            socket.send_json({"type": "stop"})
            returned = [first, cleaned, clean_tail]
            while True:
                message = socket.receive()
                if message.get("bytes") is not None:
                    returned.append(np.frombuffer(message["bytes"], dtype="<f4"))
                else:
                    result = json.loads(message["text"])
                    break
            self.assertEqual(result["type"], "complete")
        original = read_audio_bytes(self.client.get(result["originalUrl"]).content, "original.wav")
        denoised = read_audio_bytes(self.client.get(result["denoisedUrl"]).content, "denoised.wav")
        expected_original = np.concatenate([samples, samples, tail])
        expected_cleaned = np.concatenate(returned)
        self.assertEqual(expected_cleaned.size, expected_original.size)
        np.testing.assert_array_equal(expected_cleaned[:len(samples)], samples)
        np.testing.assert_allclose(original.samples[:, 0], expected_original, atol=0.0001)
        np.testing.assert_allclose(denoised.samples[:, 0], expected_cleaned, atol=0.0001)

    def test_stream_errors_release_capacity(self):
        for payload in [b"123", np.array([np.nan], dtype="<f4").tobytes(), bytes(40000)]:
            with self.client.websocket_connect("/api/live") as socket:
                socket.send_json({"type": "start", "sampleRate": 16000})
                self.assertEqual(socket.receive_json()["type"], "ready")
                socket.send_bytes(payload)
                self.assertEqual(socket.receive_json()["type"], "error")
        with self.client.websocket_connect("/api/live") as socket:
            socket.send_json({"type": "start", "sampleRate": 16000})
            self.assertEqual(socket.receive_json()["type"], "ready")
            socket.send_json({"type": "stop"})
            self.assertEqual(socket.receive_json()["type"], "error")

    def test_rejects_cross_origin_microphone(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/api/live", headers={"origin": "https://other.example"}):
                pass

    def test_cleanup_and_frontend_worklet(self):
        old = Path(self.directory.name) / "old.wav"
        recent = Path(self.directory.name) / "recent.wav"
        old.write_bytes(b"old")
        recent.write_bytes(b"recent")
        past = time.time() - server.MEDIA_TTL_SECONDS - 10
        os.utime(old, (past, past))
        server.cleanup_media()
        self.assertFalse(old.exists())
        self.assertTrue(recent.exists())
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertIn("registerProcessor", self.client.get("/capture-worklet.js").text)
        config = self.client.get("/api/config").json()
        self.assertEqual(config["maxDurationSeconds"], 120)
        self.assertEqual(config["engine"], "rnnoise")
        self.assertEqual(config["defaultStrength"], 0.85)
        self.assertEqual(self.client.get("/backend/models/voice.rnnn").status_code, 404)


if __name__ == "__main__":
    unittest.main()

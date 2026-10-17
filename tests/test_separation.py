from contextlib import ExitStack
import io
import os
from pathlib import Path
import subprocess
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
import uvicorn

import backend.app as server
from backend import separation
from backend.denoise import AudioError, write_wav_bytes
import separation_service.app as service
from separation_service.model import verify_assets
from separation_service.worker import main as separation_worker


CONFIG = {"SEPARATION_URL": "http://127.0.0.1:8101", "SEPARATION_ALLOW_HTTP": "1", "SEPARATION_API_KEY": "t" * 40}


class SeparationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.dict(os.environ, CONFIG))
        self.stack.enter_context(patch.object(server, "MEDIA_DIR", self.directory))
        self.owner = self.stack.enter_context(TestClient(server.app, headers={"origin": "http://testserver"}))
        self.other = TestClient(server.app, headers={"origin": "http://testserver"})
        self.owner.get("/api/session").raise_for_status()
        self.other.get("/api/session").raise_for_status()
        self.wav = write_wav_bytes(np.zeros(800, np.float32), 8000)
        self.metadata = {"inputFrames": 800, "sampleRate": 8000, "channels": 1, "durationSeconds": .1,
                         "strength": 0, "inputFormat": "wav", "engine": "decode"}
        self.decode = self.stack.enter_context(patch.object(server, "process_file", return_value=(self.wav, self.wav, self.metadata)))
        self.remote = self.stack.enter_context(patch.object(separation, "separate_audio", return_value=[self.wav, self.wav]))

    def tearDown(self):
        self.other.close()
        self.stack.close()

    def upload(self, client=None, content_type="audio/wav"):
        return (client or self.owner).post("/api/separate", files={"file": ("test.wav", self.wav, content_type)})

    def test_outputs_are_owned_and_history_keeps_the_two_sources(self):
        response = self.upload()
        self.assertEqual(response.status_code, 200, response.text)
        sources = response.json()["sources"]
        self.assertEqual(len(sources), 2)
        self.decode.assert_called_once_with(self.wav, "test.wav", "audio/wav", 0, "off", "decode")
        self.assertNotEqual(sources[0]["recordingId"], sources[1]["recordingId"])
        self.assertEqual([source["sourceIndex"] for source in sources], [1, 2])
        for source in sources:
            self.assertEqual(source["engine"], "sepformer")
            self.assertEqual(self.owner.get(source["denoisedUrl"]).status_code, 200)
            self.assertEqual(self.other.get(source["denoisedUrl"]).status_code, 404)
            self.assertEqual(self.other.delete("/api/recordings/" + source["recordingId"]).status_code, 404)
        self.assertEqual(len(self.owner.get("/api/recordings").json()["recordings"]), 2)
        self.assertNotIn(CONFIG["SEPARATION_API_KEY"], self.owner.get("/api/config").text)

    def test_wav_mime_aliases_are_forwarded_to_the_decoder(self):
        for content_type in ("audio/wav", "audio/x-wav"):
            with self.subTest(content_type=content_type):
                self.decode.reset_mock()
                response = self.upload(content_type=content_type)
                self.assertEqual(response.status_code, 200, response.text)
                self.decode.assert_called_once_with(self.wav, "test.wav", content_type, 0, "off", "decode")

    def test_same_origin_and_owner_are_required_before_decoding(self):
        with TestClient(server.app) as anonymous:
            self.assertEqual(self.upload(anonymous).status_code, 403)
            response = anonymous.post("/api/separate", headers={"origin": "http://testserver"}, files={"file": ("x.wav", self.wav)})
            self.assertEqual(response.status_code, 401)
        response = self.owner.post("/api/separate", headers={"origin": "https://elsewhere.invalid"}, files={"file": ("x.wav", self.wav)})
        self.assertEqual(response.status_code, 403)
        self.decode.assert_not_called()

    def test_missing_configuration_disables_feature_without_reading_file(self):
        with patch.dict(os.environ, {"SEPARATION_URL": "", "SEPARATION_API_KEY": ""}):
            self.assertFalse(self.owner.get("/api/config").json()["separationEnabled"])
            self.assertEqual(self.upload().status_code, 503)
        self.decode.assert_not_called()

    def test_partial_quota_failure_rolls_back_both_sources(self):
        server.app.state.store.owner_bytes = len(self.wav) * 3
        self.assertEqual(self.upload().status_code, 507)
        self.assertEqual(self.owner.get("/api/recordings").json()["recordings"], [])
        self.assertEqual(list(self.directory.glob("*.wav")), [])

    def test_remote_failure_is_generic_and_does_not_save_or_leak_credentials(self):
        self.remote.side_effect = AudioError("private error " + CONFIG["SEPARATION_API_KEY"])
        response = self.upload()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(CONFIG["SEPARATION_API_KEY"], response.text)
        self.assertEqual(list(self.directory.glob("*.wav")), [])
        self.assertEqual(server.app.state.file_slots._value, 1)

    def test_client_cannot_select_an_upstream_url(self):
        response = self.owner.post("/api/separate", files={"file": ("x.wav", self.wav)}, data={"url": "http://169.254.169.254"})
        self.assertEqual(response.status_code, 400)
        self.remote.assert_not_called()


class SeparationTransportTests(unittest.TestCase):
    def test_configuration_rejects_credentials_urls_and_insecure_transport_by_default(self):
        for url in ("http://example.com", "https://u:p@example.com", "https://example.com/?key=x", "file:///secret", "https://example.com:invalid"):
            with patch.dict(os.environ, {**CONFIG, "SEPARATION_URL": url, "SEPARATION_ALLOW_HTTP": "0"}):
                with self.assertRaises(ValueError):
                    separation.settings()

    def test_response_is_bounded_and_must_be_two_channel_eight_khz_audio(self):
        for payload in (b"x" * (separation.MAX_RESPONSE_BYTES + 1), write_wav_bytes(np.zeros(800), 8000),
                        write_wav_bytes(np.zeros((800, 2)), 16000), write_wav_bytes(np.zeros((3000, 2)), 8000)):
            response = io.BytesIO(payload)
            response.status = 200
            response.getheader = Mock(return_value="audio/wav")
            connection = Mock()
            connection.getresponse.return_value = response
            with patch.dict(os.environ, CONFIG), patch.object(separation.http.client, "HTTPConnection", return_value=connection):
                with self.assertRaises(AudioError):
                    separation.separate_audio(b"test", .1)

    def test_transport_passes_only_service_key_not_browser_cookies(self):
        response = io.BytesIO(write_wav_bytes(np.zeros((800, 2)), 8000))
        response.status = 200
        response.getheader = Mock(return_value="audio/wav")
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.dict(os.environ, CONFIG), patch.object(separation.http.client, "HTTPConnection", return_value=connection):
            self.assertEqual(len(separation.separate_audio(b"pcm", .1)), 2)
        headers = connection.request.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer " + CONFIG["SEPARATION_API_KEY"])
        self.assertNotIn("Cookie", headers)
        self.assertNotIn("Connection", headers)

    def test_redirects_are_not_followed_with_the_service_credential(self):
        response = io.BytesIO(b"redirect")
        response.status = 302
        response.getheader = Mock(return_value="audio/wav")
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.dict(os.environ, CONFIG), patch.object(separation.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaises(AudioError):
                separation.separate_audio(b"pcm", .1)
        connection.request.assert_called_once()

    def test_missing_or_tampered_weights_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_assets(directory)

    def test_worker_rejects_long_audio_before_loading_ml_model(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "input.wav").write_bytes(write_wav_bytes(np.zeros(8000 * 16), 8000))
            with patch("separation_service.worker.Separator", side_effect=AssertionError("must not load")):
                with self.assertRaises(ValueError):
                    separation_worker(directory, "unused")


class PrivateServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.dict(os.environ, CONFIG))
        self.stack.enter_context(patch.object(service, "verify_assets"))
        self.client = self.stack.enter_context(TestClient(service.app))
        self.worker = self.stack.enter_context(patch.object(service, "run_worker", return_value=b"result"))
        self.headers = {"authorization": "Bearer " + CONFIG["SEPARATION_API_KEY"], "content-type": "audio/wav"}

    def tearDown(self):
        self.stack.close()

    def test_unauthenticated_requests_cannot_invoke_the_model(self):
        response = self.client.post("/v1/separate", content=b"audio", headers={"content-type": "audio/wav"})
        self.assertEqual(response.status_code, 401)
        self.worker.assert_not_called()
        for path in ("/docs", "/media/input.wav", "/.tools/separation-model/masknet.ckpt"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_body_limits_and_admission_precede_model_execution(self):
        response = self.client.post("/v1/separate", content=b"small", headers={**self.headers, "content-length": str(service.MAX_BODY_BYTES + 1)})
        self.assertEqual(response.status_code, 413)
        service.app.state.slot._value = 0
        response = self.client.post("/v1/separate", content=b"small", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.worker.assert_not_called()

    def test_timeout_releases_the_slot_and_responses_are_private(self):
        self.worker.side_effect = subprocess.TimeoutExpired("private worker", 120)
        response = self.client.post("/v1/separate", content=b"small", headers=self.headers)
        self.assertEqual(response.status_code, 408)
        self.assertEqual(service.app.state.slot._value, 1)
        self.worker.side_effect = None
        response = self.client.post("/v1/separate", content=b"small", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["cache-control"])


class RealTransportTests(unittest.TestCase):
    def test_authenticated_post_waits_for_a_real_http_response(self):
        request_audio = write_wav_bytes(np.zeros(64000, np.float32), 8000)
        result_audio = write_wav_bytes(np.zeros((64000, 2), np.float32), 8000)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            config = {**CONFIG, "SEPARATION_URL": "http://127.0.0.1:" + str(port)}
            with patch.dict(os.environ, config), patch.object(service, "verify_assets"), \
                    patch.object(service, "run_worker", return_value=result_audio) as model:
                server = uvicorn.Server(uvicorn.Config(service.app, log_level="error", access_log=False, loop="asyncio"))
                thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
                thread.start()
                try:
                    deadline = time.monotonic() + 5
                    while not server.started and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(server.started)
                    sources = separation.separate_audio(request_audio, 8)
                    self.assertEqual(len(sources), 2)
                    model.assert_called_once_with(request_audio)
                finally:
                    server.should_exit = True
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()

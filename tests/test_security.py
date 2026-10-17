import asyncio
import tempfile
import time
import unittest
from contextlib import ExitStack, closing
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend.app as server
from backend.storage import MediaStore, SESSION_COOKIE
from backend.worker import process_file
from tests.test_audio import compressed_fixture


class SecurityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(server, "MEDIA_DIR", self.directory))
        self.owner = self.stack.enter_context(TestClient(
            server.app, headers={"origin": "http://testserver"}, client=("192.0.2.1", 10001)))
        self.other = TestClient(server.app, headers={"origin": "http://testserver"},
                                client=("198.51.100.2", 10002))
        self.stranger = TestClient(server.app, client=("203.0.113.3", 10003))
        self.stack.callback(self.other.close)
        self.stack.callback(self.stranger.close)
        for client in [self.owner, self.other]:
            self.assertEqual(client.get("/api/session").status_code, 200)
        self.wav = server.write_wav_bytes(np.zeros((160, 1), dtype=np.float32), 8000)

    def upload(self, client=None):
        response = (client or self.owner).post("/api/denoise", files={"file": ("private.wav", self.wav)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def start(self, socket):
        socket.send_json({"type": "start", "sampleRate": 8000, "strength": 0})
        self.assertEqual(socket.receive_json()["type"], "ready")

    def test_other_users_cannot_get_head_range_or_validate_known_urls(self):
        result = self.upload()
        for key in ["originalUrl", "denoisedUrl"]:
            url = result[key]
            own = self.owner.get(url)
            self.assertEqual(own.status_code, 200)
            self.assertIn("no-store", own.headers["cache-control"])
            self.assertEqual(self.owner.head(url).status_code, 200)
            self.assertEqual(self.owner.get(url, headers={"Range": "bytes=0-15"}).status_code, 206)
            for client in [self.other, self.stranger]:
                for method, headers in [("GET", {}), ("HEAD", {}),
                                        ("GET", {"Range": "bytes=0-15"}),
                                        ("GET", {"If-None-Match": own.headers["etag"]})]:
                    response = client.request(method, url, headers=headers)
                    self.assertEqual(response.status_code, 404)
                    self.assertIn("no-store", response.headers["cache-control"])
                    self.assertNotEqual(response.content, own.content)

    def test_uploaded_mp3_m4a_results_are_also_private(self):
        for extension in ["mp3", "m4a"]:
            response = self.owner.post("/api/denoise", files={
                "file": ("private." + extension, compressed_fixture(extension))})
            self.assertEqual(response.status_code, 200, response.text)
            for kind in ["originalUrl", "denoisedUrl"]:
                self.assertEqual(self.owner.get(response.json()[kind]).status_code, 200)
                self.assertEqual(self.other.get(response.json()[kind]).status_code, 404)

    def test_variant_names_are_independent_and_legacy_names_are_not_public(self):
        result = self.upload()
        self.assertNotEqual(result["originalUrl"], result["denoisedUrl"])
        original = Path(result["originalUrl"]).stem
        denoised = Path(result["denoisedUrl"]).stem
        self.assertNotEqual(original, denoised)
        self.assertEqual(len(original), 32)
        for name in [denoised + "_original.wav", denoised + "_denoised.wav"]:
            self.assertEqual(self.owner.get("/media/" + name).status_code, 404)
        legacy = uuid4().hex + "_original.wav"
        (self.directory / legacy).write_bytes(self.wav)
        self.assertEqual(self.owner.get("/media/" + legacy).status_code, 404)

    def test_owner_only_deletion_revokes_both_variants(self):
        result = self.upload()
        endpoint = "/api/recordings/" + result["recordingId"]
        self.assertEqual(self.other.delete(endpoint).status_code, 404)
        self.assertEqual(self.owner.get(result["originalUrl"]).status_code, 200)
        self.assertEqual(self.owner.delete(endpoint).status_code, 204)
        for key in ["originalUrl", "denoisedUrl"]:
            self.assertEqual(self.owner.get(result[key]).status_code, 404)
            self.assertFalse((self.directory / Path(result[key]).name).exists())

    def test_expiry_denies_access_before_cleanup(self):
        result = self.upload()
        with closing(server.app.state.store.connect()) as db, db:
            db.execute("UPDATE recordings SET expires=? WHERE id=?",
                       (time.time() - 1, result["recordingId"]))
        self.assertTrue((self.directory / Path(result["originalUrl"]).name).exists())
        for key in ["originalUrl", "denoisedUrl"]:
            self.assertEqual(self.owner.get(result[key]).status_code, 404)
            self.assertEqual(self.owner.head(result[key]).status_code, 404)
            self.assertEqual(self.owner.get(result[key], headers={"Range": "bytes=0-15"}).status_code, 404)

    def test_locked_deleted_files_are_revoked_and_cleanup_retries_unlink(self):
        result = self.upload()
        with patch.object(Path, "unlink", side_effect=PermissionError("Temporary file lock")):
            self.assertEqual(self.owner.delete("/api/recordings/" + result["recordingId"]).status_code, 204)
        for key in ["originalUrl", "denoisedUrl"]:
            self.assertEqual(self.owner.get(result[key]).status_code, 404)
            self.assertTrue((self.directory / Path(result[key]).name).exists())
        server.cleanup_media()
        for key in ["originalUrl", "denoisedUrl"]:
            self.assertFalse((self.directory / Path(result[key]).name).exists())

    def test_session_cookie_flags_and_unknown_cookie_rotation(self):
        self.stranger.cookies.set(SESSION_COOKIE, "A" * 43)
        response = self.stranger.get("/api/session")
        self.assertEqual(response.status_code, 200)
        issued = response.cookies.get(SESSION_COOKIE)
        self.assertNotEqual(issued, "A" * 43)
        header = response.headers["set-cookie"].lower()
        self.assertIn("httponly", header)
        self.assertIn("samesite=strict", header)
        self.assertIn("no-store", response.headers["cache-control"])
        with patch.object(server, "COOKIE_SECURE", True):
            secure = TestClient(server.app, base_url="https://testserver")
            self.addCleanup(secure.close)
            response = secure.get("/api/session")
            self.assertIn("; secure", response.headers["set-cookie"].lower())

    def test_expired_or_forged_session_cannot_read_known_audio(self):
        result = self.upload()
        self.stranger.cookies.set(SESSION_COOKIE, "A" * 43)
        self.assertEqual(self.stranger.get(result["originalUrl"]).status_code, 404)
        owner_hash = server.app.state.store.owner(self.owner.cookies.get(SESSION_COOKIE))
        with closing(server.app.state.store.connect()) as db, db:
            db.execute("UPDATE sessions SET expires=? WHERE owner=?", (time.time() - 1, owner_hash))
        self.assertEqual(self.owner.get(result["originalUrl"]).status_code, 404)

    def test_ownership_survives_storage_reopen(self):
        result = self.upload()
        reopened = MediaStore(self.directory, server.MEDIA_TTL_SECONDS)
        owner = reopened.owner(self.owner.cookies.get(SESSION_COOKIE))
        self.assertIsNotNone(owner)
        self.assertIsNotNone(reopened.locate(owner, Path(result["originalUrl"]).name))

    def test_no_media_listing_database_download_or_relative_traversal(self):
        for path in ["/media", "/media/", "/media/private.sqlite3",
                     "/media/%2e%2e/private.sqlite3", "/media/..%5cprivate.sqlite3",
                     "/media/%252e%252e%252fprivate.sqlite3"]:
            self.assertEqual(self.owner.get(path).status_code, 404, path)
        self.assertEqual(self.stranger.get("/api/recordings").status_code, 401)

    def test_upload_requires_session_and_same_origin_before_parsing(self):
        with patch.object(server, "parse_upload", side_effect=AssertionError("Parser must not run")):
            self.assertEqual(self.stranger.post("/api/denoise", headers={
                "origin": "http://testserver"}, content=b"untrusted").status_code, 401)
            self.assertEqual(self.owner.post("/api/denoise", headers={
                "origin": "https://evil.example"}, content=b"untrusted").status_code, 403)
        response = self.stranger.get("/api/session", headers={"sec-fetch-site": "cross-site"})
        self.assertEqual(response.status_code, 403)

    def test_cross_origin_delete_is_rejected(self):
        result = self.upload()
        response = self.owner.delete("/api/recordings/" + result["recordingId"],
                                     headers={"origin": "https://evil.example"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.owner.get(result["originalUrl"]).status_code, 200)

    def test_total_body_limit_precedes_parser_with_or_without_content_length(self):
        with patch.object(server, "MAX_UPLOAD_BYTES", 128), \
                patch.object(server, "MULTIPART_OVERHEAD_BYTES", 64), \
                patch.object(server, "MultiPartParser", side_effect=AssertionError("Parser must not run")):
            cases = [(b"x" * 200, {"content-length": "200"}),
                     (iter([b"x" * 100, b"y" * 100]), {}),
                     (iter([b"x" * 100, b"y" * 100]), {"content-length": "1"})]
            for content, headers in cases:
                response = self.owner.post("/api/denoise", content=content,
                    headers={"content-type": "multipart/form-data; boundary=x", **headers})
                self.assertEqual(response.status_code, 413, response.text)

    def test_extra_files_duplicate_fields_and_unknown_fields_are_rejected(self):
        response = self.owner.post("/api/denoise", files=[
            ("file", ("small.wav", self.wav)), ("unused", ("large.wav", b"x" * 1024))])
        self.assertEqual(response.status_code, 400)
        response = self.owner.post("/api/denoise", files=[
            ("file", ("small.wav", self.wav)), ("strength", (None, "0")),
            ("strength", (None, "1"))])
        self.assertEqual(response.status_code, 400)
        response = self.owner.post("/api/denoise", files={"file": ("small.wav", self.wav)},
                                   data={"unexpected": "value"})
        self.assertEqual(response.status_code, 400)

    def test_busy_admission_happens_before_reading_upload(self):
        with patch.object(server.app.state, "file_slots", asyncio.Semaphore(0)), \
                patch.object(server.app.state.file_admission, "wait_seconds", 0.01), \
                patch.object(server, "parse_upload", side_effect=AssertionError("Must not read body")):
            response = self.owner.post("/api/denoise")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["retry-after"], "3")

    def test_upload_rate_cannot_be_reset_by_creating_another_session(self):
        for unused in range(server.IP_RATE_LIMITS["upload"]):
            server.app.state.rates.allow("upload:ip:192.0.2.1", server.IP_RATE_LIMITS["upload"])
        self.assertEqual(self.owner.post("/api/denoise").status_code, 429)
        replacement = TestClient(server.app, headers={"origin": "http://testserver"},
                                  client=("192.0.2.1", 11001))
        self.addCleanup(replacement.close)
        self.assertEqual(replacement.get("/api/session").status_code, 200)
        self.assertEqual(replacement.post("/api/denoise").status_code, 429)

    def test_owner_and_global_disk_quotas_do_not_create_output_files(self):
        for attribute in ["owner_bytes", "max_bytes"]:
            with patch.object(server.app.state.store, attribute, 1):
                response = self.owner.post("/api/denoise", files={"file": ("small.wav", self.wav)})
                self.assertEqual(response.status_code, 507, response.text)
                self.assertEqual(list(self.directory.glob("*.wav")), [])

    def test_live_requires_session_and_origin(self):
        for client, headers in [(self.stranger, {"origin": "http://testserver"}),
                                (self.owner, {"origin": "https://evil.example"}),
                                (self.stranger, {})]:
            with self.assertRaises(WebSocketDisconnect):
                with client.websocket_connect("/api/live", headers=headers):
                    pass

    def test_same_owner_cannot_take_two_microphone_slots(self):
        with self.owner.websocket_connect("/api/live") as first:
            self.start(first)
            with self.owner.websocket_connect("/api/live") as duplicate:
                self.assertEqual(duplicate.receive_json()["type"], "error")
            with self.other.websocket_connect("/api/live") as independent:
                self.start(independent)
        self.assertFalse(server.app.state.live_owners)

    def test_independent_sessions_on_same_ip_can_share_global_capacity(self):
        replacement = TestClient(server.app, headers={"origin": "http://testserver"},
                                  client=("192.0.2.1", 12001))
        self.addCleanup(replacement.close)
        self.assertEqual(replacement.get("/api/session").status_code, 200)
        with self.owner.websocket_connect("/api/live") as first:
            self.start(first)
            with replacement.websocket_connect("/api/live") as second:
                self.start(second)
                with self.other.websocket_connect("/api/live") as third:
                    third.send_json({"type": "start", "sampleRate": 8000})
                    self.assertIn("busy", third.receive_json()["detail"])

    def test_control_messages_cannot_extend_audio_idle_timeout(self):
        with patch.object(server, "LIVE_PROGRESS_SECONDS", 0.25):
            with self.owner.websocket_connect("/api/live") as socket:
                self.start(socket)
                time.sleep(0.15)
                socket.send_json({"type": "strength", "value": 0})
                time.sleep(0.15)
                self.assertEqual(socket.receive_json()["type"], "error")
        self.assertFalse(server.app.state.live_owners)

    def test_absolute_lifetime_is_enforced_separately_from_sample_count(self):
        with patch.object(server, "MAX_AUDIO_SECONDS", 0.25), patch.object(server, "LIVE_GRACE_SECONDS", 0):
            with self.owner.websocket_connect("/api/live") as socket:
                self.start(socket)
                socket.send_bytes(np.zeros(10, dtype="<f4").tobytes())
                self.assertLessEqual(len(socket.receive_bytes()), 40)
                time.sleep(0.3)
                self.assertEqual(socket.receive_json()["type"], "error")

    def test_live_chunk_count_and_message_rate_are_bounded(self):
        with patch.object(server, "MAX_LIVE_CHUNKS", 1):
            with self.owner.websocket_connect("/api/live") as socket:
                self.start(socket)
                socket.send_bytes(np.zeros(10, dtype="<f4").tobytes())
                socket.receive_bytes()
                socket.send_bytes(np.zeros(10, dtype="<f4").tobytes())
                self.assertIn("chunks", socket.receive_json()["detail"])
        with self.owner.websocket_connect("/api/live") as socket:
            self.start(socket)
            for unused in range(31):
                socket.send_json({"type": "strength", "value": 0})
            self.assertIn("rate", socket.receive_json()["detail"])

    def test_live_completed_recording_has_owner_only_downloads(self):
        with self.owner.websocket_connect("/api/live") as socket:
            self.start(socket)
            socket.send_bytes(np.zeros(400, dtype="<f4").tobytes())
            socket.receive_bytes()
            socket.send_json({"type": "stop"})
            socket.receive_bytes()  # Flush neural look-ahead before the completed result.
            result = socket.receive_json()
        self.assertEqual(result["type"], "complete")
        for key in ["originalUrl", "denoisedUrl"]:
            self.assertEqual(self.owner.get(result[key]).status_code, 200)
            self.assertEqual(self.other.get(result[key]).status_code, 404)
            self.assertEqual(self.stranger.get(result[key]).status_code, 404)

    def test_decoder_cannot_autoprobe_a_playlist_disguised_as_m4a(self):
        playlist = b"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\nfile:///not-accessed\n#EXT-X-ENDLIST\n"
        with self.assertRaises(server.AudioError):
            process_file(playlist, "playlist.m4a", "audio/mp4", 0.8)

    def test_worker_timeout_is_reported_without_saving_results(self):
        import subprocess
        with patch("backend.worker.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            response = self.owner.post("/api/denoise", files={"file": ("small.wav", self.wav)})
        self.assertEqual(response.status_code, 408)
        self.assertFalse(list(self.directory.glob("*.wav")))
        self.assertFalse(server.app.state.store.active_owners)

    def test_untrusted_host_is_rejected(self):
        self.assertEqual(self.owner.get("/", headers={"host": "evil.example"}).status_code, 400)

    def test_security_headers_and_local_script_assets(self):
        response = self.owner.get("/")
        policy = response.headers["content-security-policy"]
        self.assertIn("script-src 'self'", policy)
        self.assertNotIn("unsafe-eval", policy)
        self.assertIn("frame-ancestors 'none'", policy)
        self.assertNotIn("unpkg.com", response.text)
        self.assertEqual(self.owner.get("/vendor/lucide-1.58.0.min.js").status_code, 200)

    def test_multipart_header_excess_is_rejected_without_processing(self):
        body = (b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                + b"X-Extra: value\r\n" * 200 + b"\r\nx\r\n--boundary--\r\n")
        with patch.object(server, "process_file", side_effect=AssertionError("Must not process")):
            response = self.owner.post("/api/denoise", content=body,
                headers={"content-type": "multipart/form-data; boundary=boundary"})
        self.assertEqual(response.status_code, 400, response.text)

    def test_slow_upload_is_timed_out_and_releases_admission(self):
        cookie = self.owner.cookies.get(SESSION_COOKIE)

        async def probe():
            scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                     "method": "POST", "scheme": "http", "path": "/api/denoise",
                     "raw_path": b"/api/denoise", "query_string": b"", "root_path": "",
                     "client": ("192.0.2.1", 10001), "server": ("testserver", 80),
                     "headers": [(b"host", b"testserver"), (b"origin", b"http://testserver"),
                                 (b"cookie", (SESSION_COOKIE + "=" + cookie).encode("ascii")),
                                 (b"content-type", b"multipart/form-data; boundary=x")]}
            sent = []

            async def receive():
                await asyncio.sleep(1)
                return {"type": "http.request", "body": b"", "more_body": False}

            async def send(message):
                sent.append(message)

            await server.app(scope, receive, send)
            return next(message["status"] for message in sent if message["type"] == "http.response.start")

        with patch.object(server, "UPLOAD_SECONDS", 0.02):
            self.assertEqual(asyncio.run(probe()), 408)
        self.assertEqual(server.app.state.file_slots._value, 1)
        self.assertFalse(server.app.state.store.active_owners)

    def test_simultaneous_private_streams_still_return_their_own_audio(self):
        with self.owner.websocket_connect("/api/live") as first, \
                self.other.websocket_connect("/api/live") as second:
            self.start(first)
            self.start(second)
            a = np.full(2000, 0.25, dtype="<f4").tobytes()
            b = np.full(2000, -0.5, dtype="<f4").tobytes()
            first.send_bytes(a)
            second.send_bytes(b)
            output_a = first.receive_bytes()
            output_b = second.receive_bytes()
            self.assertTrue(output_a)
            self.assertTrue(output_b)
            self.assertEqual(output_a, a[:len(output_a)])
            self.assertEqual(output_b, b[:len(output_b)])
            first.send_json({"type": "stop"})
            second.send_json({"type": "stop"})
            self.assertEqual(output_a + first.receive_bytes(), a)
            self.assertEqual(output_b + second.receive_bytes(), b)
            self.assertEqual(first.receive_json()["type"], "complete")
            self.assertEqual(second.receive_json()["type"], "complete")


if __name__ == "__main__":
    unittest.main()

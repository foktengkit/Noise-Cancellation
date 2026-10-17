from contextlib import ExitStack, closing
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
from fastapi.testclient import TestClient
import yaml

import backend.app as server
from backend.storage import MediaStore, SESSION_COOKIE
import scripts.package_release as release


ROOT = Path(__file__).resolve().parent.parent


class RemediationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(server, "MEDIA_DIR", self.directory))
        self.owner = self.stack.enter_context(TestClient(server.app, headers={"origin": "http://testserver"},
                                                        client=("10.0.0.10", 10001)))
        self.other = TestClient(server.app, headers={"origin": "http://testserver"}, client=("10.0.0.10", 10002))
        self.stack.callback(self.other.close)
        for client in (self.owner, self.other):
            self.assertEqual(client.get("/api/session").status_code, 200)
        self.wav = server.write_wav_bytes(np.zeros((80, 1), np.float32), 8000)
        self.metadata = {"sampleRate": 8000, "channels": 1, "durationSeconds": .01,
                         "strength": 1, "inputFormat": "wav", "engine": "rnnoise"}
        self.stack.enter_context(patch.object(server, "process_file", return_value=(self.wav, self.wav, self.metadata)))

    def upload(self, client=None):
        response = (client or self.owner).post("/api/denoise", files={"file": ("test.wav", self.wav)})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_private_list_contains_only_owners_metadata_and_downloads(self):
        owned = self.upload()
        foreign = self.upload(self.other)
        response = self.owner.get("/api/recordings")
        self.assertEqual(response.status_code, 200)
        self.assertIn("no-store", response.headers["cache-control"])
        result = response.json()
        self.assertEqual(len(result["recordings"]), 1)
        row = result["recordings"][0]
        self.assertEqual(row["recordingId"], owned["recordingId"])
        self.assertNotIn(foreign["recordingId"], response.text)
        self.assertNotIn("owner", row)
        self.assertTrue(row["canRestore"])
        for key, value in self.metadata.items():
            self.assertEqual(row[key], value)
        self.assertEqual(result["usedBytes"], len(self.wav) * 2)
        self.assertEqual(result["maxBytes"], server.app.state.store.owner_bytes)

    def test_list_without_origin_supports_same_origin_fetch_but_rejects_cross_site(self):
        client = TestClient(server.app, cookies=self.owner.cookies)
        self.addCleanup(client.close)
        self.assertEqual(client.get("/api/recordings", headers={"sec-fetch-site": "same-origin"}).status_code, 200)
        for site in ("cross-site", "same-site"):
            self.assertEqual(client.get("/api/recordings", headers={"sec-fetch-site": site}).status_code, 403)
        self.assertEqual(client.get("/api/recordings", headers={"origin": "https://evil.example"}).status_code, 403)
        anonymous = TestClient(server.app)
        self.addCleanup(anonymous.close)
        self.assertEqual(anonymous.get("/api/recordings").status_code, 401)

    def test_bulk_delete_cannot_remove_or_reveal_other_users_recordings(self):
        owned = self.upload()
        foreign = self.upload(self.other)
        self.assertEqual(self.other.delete("/api/recordings", headers={"origin": "https://evil.example"}).status_code, 403)
        self.assertEqual(self.owner.delete("/api/recordings").status_code, 204)
        self.assertEqual(self.owner.get(owned["originalUrl"]).status_code, 404)
        self.assertEqual(self.other.get(foreign["originalUrl"]).status_code, 200)
        self.assertEqual(self.other.get("/api/recordings").json()["recordings"][0]["recordingId"], foreign["recordingId"])

    def test_reloaded_session_can_find_delete_and_reclaim_quota(self):
        server.app.state.store.owner_bytes = len(self.wav) * 2 + 1
        owned = self.upload()
        client = TestClient(server.app, headers={"origin": "http://testserver"}, cookies=self.owner.cookies)
        self.addCleanup(client.close)
        self.assertEqual(client.post("/api/denoise", files={"file": ("x.wav", self.wav)}).status_code, 507)
        rows = client.get("/api/recordings").json()["recordings"]
        self.assertEqual(rows[0]["recordingId"], owned["recordingId"])
        self.assertEqual(client.delete("/api/recordings/" + rows[0]["recordingId"]).status_code, 204)
        self.assertEqual(client.post("/api/denoise", files={"file": ("x.wav", self.wav)}).status_code, 200)

    def test_expired_rows_are_omitted_from_list_and_usage_before_cleanup(self):
        owned = self.upload()
        with closing(server.app.state.store.connect()) as db, db:
            db.execute("UPDATE recordings SET expires=? WHERE id=?", (time.time() - 1, owned["recordingId"]))
        result = self.owner.get("/api/recordings").json()
        self.assertEqual(result["recordings"], [])
        self.assertEqual(result["usedBytes"], 0)

    def test_bulk_delete_revokes_locked_files_and_retries_cleanup(self):
        owned = self.upload()
        with patch.object(Path, "unlink", side_effect=PermissionError("Locked")):
            self.assertEqual(self.owner.delete("/api/recordings").status_code, 204)
        self.assertEqual(self.owner.get(owned["originalUrl"]).status_code, 404)
        self.assertTrue(list(self.directory.glob("*.wav")))
        server.cleanup_media()
        self.assertFalse(list(self.directory.glob("*.wav")))

    def test_independent_users_share_proxy_without_sharing_owner_limits(self):
        for unused in range(6):
            self.upload()
        self.assertEqual(self.owner.post("/api/denoise", files={"file": ("x.wav", self.wav)}).status_code, 429)
        self.upload(self.other)

    def test_forwarded_headers_do_not_bypass_aggregate_ip_limit(self):
        for unused in range(server.IP_RATE_LIMITS["upload"]):
            server.app.state.rates.allow("upload:ip:10.0.0.10", server.IP_RATE_LIMITS["upload"])
        response = self.other.post("/api/denoise", files={"file": ("x.wav", self.wav)},
                                   headers={"x-forwarded-for": "198.51.100.42"})
        self.assertEqual(response.status_code, 429)

    def test_missing_session_can_be_renewed_without_reusing_old_ownership(self):
        owned = self.upload()
        old_cookie = self.owner.cookies[SESSION_COOKIE]
        with closing(server.app.state.store.connect()) as db, db:
            db.execute("DELETE FROM sessions")
        self.assertEqual(self.owner.get("/api/recordings").status_code, 401)
        self.assertEqual(self.owner.get("/api/session").status_code, 200)
        self.assertNotEqual(old_cookie, self.owner.cookies[SESSION_COOKIE])
        self.assertEqual(self.owner.get("/api/recordings").json()["recordings"], [])
        self.assertEqual(self.owner.get(owned["originalUrl"]).status_code, 404)
        self.upload()


class DeploymentTests(unittest.TestCase):
    def test_legacy_database_migrates_without_disclosing_or_dropping_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with closing(sqlite3.connect(root / "private.sqlite3")) as db, db:
                db.execute("CREATE TABLE recordings (id TEXT PRIMARY KEY, owner TEXT NOT NULL, "
                           "original TEXT UNIQUE NOT NULL, denoised TEXT UNIQUE NOT NULL, "
                           "expires REAL NOT NULL, size INTEGER NOT NULL)")
                db.execute("INSERT INTO recordings VALUES (?, ?, ?, ?, ?, ?)",
                           (uuid4().hex, "owner", uuid4().hex + ".wav", uuid4().hex + ".wav", time.time() + 100, 8))
            store = MediaStore(root, 3600)
            rows = store.list_recordings("owner")["recordings"]
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["canRestore"])
            self.assertEqual(store.list_recordings("stranger")["recordings"], [])

    def test_ci_uses_deployment_python_and_hashed_runtime_dependencies(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8"))
        job = workflow["jobs"]["audio"]
        self.assertEqual(set(job["strategy"]["matrix"]["os"]), {"ubuntu-latest", "windows-latest"})
        setup = next(step for step in job["steps"] if "setup-python@" in step.get("uses", ""))
        self.assertEqual(setup["with"]["python-version-file"], ".python-version")
        render = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
        python_version = next(v["value"] for v in render["services"][0]["envVars"] if v["key"] == "PYTHON_VERSION")
        self.assertEqual((ROOT / ".python-version").read_text().strip(), str(python_version))
        commands = [step.get("run", "") for step in job["steps"]]
        self.assertIn("python -m pip install --require-hashes -r requirements.lock", commands)
        self.assertIn("python -m pip check", commands)

    def test_release_contains_ci_but_not_private_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (*release.FILES, ".github/workflows/tests.yml", "media/private.sqlite3", "media/private.wav", ".env"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            with patch.object(release, "ROOT", root):
                release.main()
            with ZipFile(root / "dist/Noise-Cancellation.zip") as archive:
                names = archive.namelist()
                self.assertIn(".github/workflows/tests.yml", names)
                self.assertIn(".gitattributes", names)
                self.assertNotIn(".env", names)
                self.assertFalse(any(name.startswith("media/") for name in names))


if __name__ == "__main__":
    unittest.main()

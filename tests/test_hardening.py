from contextlib import ExitStack, closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import backend.app as server
from backend.denoise import write_wav_bytes
from backend.storage import MAX_SESSIONS, SESSION_IDLE_SECONDS, SESSION_SECONDS, MediaStore, StorageFull
from backend.worker import process_file
from backend.worker_security import PYTHON_FLAGS, worker_environment
import separation_service.app as service


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.now = time.time()
        self.stack.enter_context(patch("backend.storage.time.time", side_effect=lambda: self.now))
        self.store = MediaStore(self.root, 3600)
        self.wav = write_wav_bytes(np.zeros(80, np.float32), 8000)

    def test_idle_empty_session_is_invalid_even_before_periodic_cleanup(self):
        cookie = self.store.new_session()
        self.now += SESSION_IDLE_SECONDS + 1
        self.assertIsNone(self.store.owner(cookie))
        self.store.new_session()
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 1)

    def test_recent_activity_refreshes_idle_time_but_not_absolute_expiry(self):
        cookie = self.store.new_session()
        owner = self.store.owner(cookie)
        for unused in range(2):
            self.now += SESSION_IDLE_SECONDS - 1
            self.assertEqual(self.store.owner(cookie), owner)
        with closing(self.store.connect()) as db, db:
            self.assertLess(db.execute("SELECT expires FROM sessions").fetchone()[0], self.now + SESSION_SECONDS)
            db.execute("UPDATE sessions SET expires=?", (self.now + 1,))
        self.now += 2
        self.assertIsNone(self.store.owner(cookie))

    def test_unexpired_audio_keeps_its_owner_after_idle_timeout(self):
        cookie = self.store.new_session()
        owner = self.store.owner(cookie)
        result = self.store.save(owner, self.wav, self.wav, {})
        unused_cookie = self.store.new_session()
        self.now += SESSION_IDLE_SECONDS + 1
        self.store.cleanup()
        self.assertEqual(self.store.owner(cookie), owner)
        self.assertIsNone(self.store.owner(unused_cookie))
        filename = result["originalUrl"].split("/")[-1]
        self.assertIsNotNone(self.store.locate(owner, filename))
        self.assertIsNone(self.store.locate("another-owner", filename))

    def test_full_session_table_recovers_and_active_work_is_not_evicted(self):
        cookie = self.store.new_session()
        owner = self.store.owner(cookie)
        self.store.pin(owner)
        self.store.pin(owner)
        with closing(self.store.connect()) as db, db:
            db.executemany("INSERT INTO sessions (owner, expires, last_seen) VALUES (?, ?, ?)",
                [(hashlib.sha256(str(i).encode()).hexdigest(), self.now + SESSION_SECONDS, self.now)
                 for i in range(MAX_SESSIONS - 1)])
        with self.assertRaises(StorageFull):
            self.store.new_session()
        self.now += SESSION_IDLE_SECONDS + 1
        replacement = self.store.new_session()
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 2)
        self.store.unpin(owner)
        self.store.cleanup()
        with closing(self.store.connect()) as db:
            self.assertIsNotNone(db.execute("SELECT 1 FROM sessions WHERE owner=?", (owner,)).fetchone())
        self.store.unpin(owner)
        self.store.cleanup()
        self.assertIsNone(self.store.owner(cookie))
        self.assertIsNotNone(self.store.owner(replacement))
        self.assertFalse(self.store.active_owners)

    def test_migration_preserves_recordings_and_reclaims_old_unused_sessions(self):
        cookie = self.store.new_session()
        owner = self.store.owner(cookie)
        self.store.save(owner, self.wav, self.wav, {})
        unused_cookie = self.store.new_session()
        with closing(self.store.connect()) as db, db:
            db.execute("ALTER TABLE sessions DROP COLUMN last_seen")
        reopened = MediaStore(self.root, 3600)
        self.now += SESSION_IDLE_SECONDS + 1
        reopened.cleanup()
        self.assertEqual(reopened.owner(cookie), owner)
        self.assertIsNone(reopened.owner(unused_cookie))
        self.assertEqual(len(reopened.list_recordings(owner)["recordings"]), 1)

    def test_pinning_does_not_extend_absolute_lifetime(self):
        cookie = self.store.new_session()
        owner = self.store.owner(cookie)
        self.store.pin(owner)
        self.now += SESSION_SECONDS + 1
        self.assertIsNone(self.store.owner(cookie))
        self.store.cleanup()
        with closing(self.store.connect()) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)
        self.store.unpin(owner)


class WorkerEnvironmentTests(unittest.TestCase):
    def test_only_explicit_nonsecret_settings_are_forwarded(self):
        poisoned = {name: "review-sentinel" for name in ("SEPARATION_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY",
                    "DATABASE_URL", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH", "HTTPS_PROXY")}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, poisoned):
            environment = worker_environment(directory, threads=2)
            self.assertFalse(set(poisoned) & environment.keys())
            self.assertNotIn("review-sentinel", environment.values())
            self.assertEqual(environment["HOME"], str(Path(directory).resolve()))
            self.assertEqual(environment["TMPDIR"], environment["HOME"])
            self.assertEqual(environment["OMP_NUM_THREADS"], "2")

    def test_real_decoder_runs_without_parent_python_settings(self):
        wav = write_wav_bytes(np.zeros(80, np.float32), 8000)
        with patch.dict(os.environ, {"PYTHONPATH":"/not-a-real-import-path", "PYTHONHOME":"/not-python"}):
            original, cleaned, metadata = process_file(wav, "test.wav", "audio/wav", 0)
        self.assertTrue(original.startswith(b"RIFF"))
        self.assertEqual(original, cleaned)
        self.assertEqual(metadata["sampleRate"], 8000)

    def test_both_worker_launchers_drop_keys_and_close_inherited_handles(self):
        wav = write_wav_bytes(np.zeros(80, np.float32), 8000)

        def complete(args, **options):
            self.assertNotIn("SEPARATION_API_KEY", options["env"])
            self.assertTrue(options["close_fds"])
            self.assertEqual(options["stdin"], subprocess.DEVNULL)
            for flag in PYTHON_FLAGS:
                self.assertIn(flag, args)
            module_index = args.index("-m") + 1
            root = Path(args[module_index + 1])
            if args[module_index] == "backend.worker":
                for name in ("original.wav", "denoised.wav"):
                    (root / name).write_bytes(wav)
                (root / "result.json").write_text(json.dumps({"sampleRate":8000}), encoding="utf-8")
            else:
                (root / "sources.wav").write_bytes(wav)
                self.assertEqual(options["env"]["SEPARATION_DEVICE"], "cpu")
            return subprocess.CompletedProcess(args, 0)

        with patch.dict(os.environ, {"SEPARATION_API_KEY":"not-a-real-key", "SEPARATION_DEVICE":"cpu"}), \
                patch("subprocess.run", side_effect=complete) as launch:
            process_file(wav, "test.wav", "audio/wav", 0)
            self.assertEqual(service.run_worker(wav), wav)
            self.assertEqual(launch.call_count, 2)


class SessionApiTests(unittest.TestCase):
    def test_expired_empty_cookie_is_rotated_without_inheriting_other_ownership(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(server, "MEDIA_DIR", Path(directory)), \
                TestClient(server.app, headers={"origin":"http://testserver"}) as client:
            first = client.get("/api/session").cookies["noise_lift_session"]
            with closing(server.app.state.store.connect()) as db, db:
                db.execute("UPDATE sessions SET last_seen=?", (time.time() - SESSION_IDLE_SECONDS - 1,))
            self.assertEqual(client.get("/api/recordings").status_code, 401)
            response = client.get("/api/session")
            self.assertEqual(response.status_code, 200)
            self.assertNotEqual(response.cookies["noise_lift_session"], first)
            self.assertEqual(client.get("/api/recordings").json()["recordings"], [])


if __name__ == "__main__":
    unittest.main()

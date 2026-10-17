from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from collections import Counter
from contextlib import closing, suppress
from pathlib import Path
from uuid import uuid4


SESSION_COOKIE = "noise_lift_session"
SESSION_SECONDS = 8 * 60 * 60
SESSION_IDLE_SECONDS = 5 * 60
MAX_SESSIONS = 4096
PRIVATE_HEADERS = {"Cache-Control": "no-store, private", "Pragma": "no-cache"}
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
FILE_PATTERN = re.compile(r"[0-9a-f]{32}\.wav\Z")


class StorageFull(ValueError):
    pass


class MediaStore:
    """Persist ownership, expiry, and bounded recording storage for one server."""

    def __init__(self, directory: Path, ttl: int, max_bytes: int = 256 * 1024 * 1024,
                 owner_bytes: int = 64 * 1024 * 1024):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "private.sqlite3"
        self.ttl = ttl
        self.max_bytes = max_bytes
        self.owner_bytes = owner_bytes
        self.lock = threading.RLock()
        self.active_owners = Counter()
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    owner TEXT PRIMARY KEY, expires REAL NOT NULL, last_seen REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL,
                    original TEXT UNIQUE NOT NULL, denoised TEXT UNIQUE NOT NULL,
                    expires REAL NOT NULL, size INTEGER NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS recordings_owner ON recordings(owner);
                CREATE TABLE IF NOT EXISTS pending_deletes (filename TEXT PRIMARY KEY);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(recordings)")}
            if "metadata" not in columns:
                db.execute("ALTER TABLE recordings ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
            session_columns = {row["name"] for row in db.execute("PRAGMA table_info(sessions)")}
            if "last_seen" not in session_columns:
                db.execute("ALTER TABLE sessions ADD COLUMN last_seen REAL NOT NULL DEFAULT 0")
                db.execute("UPDATE sessions SET last_seen=MIN(?, expires-?)", (time.time(), SESSION_SECONDS))

    def connect(self):
        connection = sqlite3.connect(self.database, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def owner(self, cookie: str | None) -> str | None:
        if not cookie or not TOKEN_PATTERN.fullmatch(cookie):
            return None
        owner = hashlib.sha256(cookie.encode("ascii")).hexdigest()
        now = time.time()
        with self.lock, closing(self.connect()) as db, db:
            row = db.execute(
                "SELECT owner, last_seen, EXISTS(SELECT 1 FROM recordings "
                "WHERE recordings.owner=sessions.owner AND recordings.expires>?) AS has_recording "
                "FROM sessions WHERE owner=? AND expires>?", (now, owner, now)).fetchone()
            if row and (row["last_seen"] > now - SESSION_IDLE_SECONDS
                        or row["has_recording"] or owner in self.active_owners):
                db.execute("UPDATE sessions SET last_seen=? WHERE owner=?", (now, owner))
                return row["owner"]
        return None

    def pin(self, owner: str) -> None:
        with self.lock:
            self.active_owners[owner] += 1

    def unpin(self, owner: str) -> None:
        with self.lock:
            self.active_owners[owner] -= 1
            if self.active_owners[owner] <= 0:
                del self.active_owners[owner]

    def new_session(self) -> str:
        with self.lock:
            self.cleanup()
            with closing(self.connect()) as db, db:
                if db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] >= MAX_SESSIONS:
                    raise StorageFull("Session capacity reached. Please try again later.")
                cookie = secrets.token_urlsafe(32)
                owner = hashlib.sha256(cookie.encode("ascii")).hexdigest()
                now = time.time()
                db.execute("INSERT INTO sessions (owner, expires, last_seen) VALUES (?, ?, ?)",
                           (owner, now + SESSION_SECONDS, now))
            return cookie

    def save(self, owner: str, original: bytes, denoised: bytes, metadata: dict) -> dict:
        size = len(original) + len(denoised)
        with self.lock:
            self.cleanup()
            with closing(self.connect()) as db, db:
                if not db.execute("SELECT 1 FROM sessions WHERE owner=? AND expires>?",
                                  (owner, time.time())).fetchone():
                    raise StorageFull("Your session expired. Reload the page and try again.")
                count = db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0]
                owner_count, used = db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM recordings WHERE owner=?",
                    (owner,)).fetchone()
                disk_used = sum(path.stat().st_size for path in self.directory.glob("*.wav")
                                if path.is_file())
                if count >= 500 or owner_count >= 20 or used + size > self.owner_bytes:
                    raise StorageFull("Your recording quota is full. Delete a result and try again.")
                if disk_used + size > self.max_bytes:
                    raise StorageFull("Audio storage is full. Please try again later.")
                recording_id = uuid4().hex
                names = [uuid4().hex + ".wav", uuid4().hex + ".wav"]
                expires = int(time.time() + self.ttl)
                try:
                    for name, data in zip(names, [original, denoised]):
                        with (self.directory / name).open("xb") as output:
                            output.write(data)
                    db.execute("INSERT INTO recordings "
                               "(id, owner, original, denoised, expires, size, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (recording_id, owner, *names, expires, size, json.dumps(metadata, allow_nan=False)))
                except BaseException:
                    for name in names:
                        with suppress(OSError):
                            (self.directory / name).unlink(missing_ok=True)
                    raise
        return {**metadata, "recordingId": recording_id,
                "originalUrl": "/media/" + names[0], "denoisedUrl": "/media/" + names[1],
                "expiresAt": int(expires)}

    def list_recordings(self, owner: str) -> dict:
        with self.lock, closing(self.connect()) as db:
            rows = db.execute("SELECT * FROM recordings WHERE owner=? AND expires>? "
                              "ORDER BY expires DESC, id LIMIT 20", (owner, time.time())).fetchall()
            recordings = []
            for row in rows:
                try:
                    metadata = json.loads(row["metadata"])
                except (ValueError, TypeError):
                    metadata = {}
                if not isinstance(metadata, dict):
                    metadata = {}
                recordings.append({**metadata, "recordingId": row["id"], "expiresAt": int(row["expires"]),
                                   "sizeBytes": row["size"], "originalUrl": "/media/" + row["original"],
                                   "denoisedUrl": "/media/" + row["denoised"],
                                   "canRestore": all(key in metadata for key in
                                                     ("sampleRate", "channels", "durationSeconds", "inputFormat", "strength"))})
        return {"recordings": recordings, "usedBytes": sum(row["size"] for row in rows),
                "maxBytes": self.owner_bytes}

    def clear_owner(self, owner: str) -> None:
        with self.lock, closing(self.connect()) as db, db:
            rows = db.execute("SELECT original, denoised FROM recordings WHERE owner=?", (owner,)).fetchall()
            db.execute("DELETE FROM recordings WHERE owner=?", (owner,))
            for row in rows:
                for name in row:
                    self.remove_file(db, name)

    def locate(self, owner: str | None, filename: str) -> tuple[Path, str] | None:
        if not owner or not FILE_PATTERN.fullmatch(filename):
            return None
        with self.lock, closing(self.connect()) as db:
            row = db.execute(
                "SELECT original FROM recordings WHERE owner=? AND expires>? "
                "AND (original=? OR denoised=?)",
                (owner, time.time(), filename, filename)).fetchone()
            if not row:
                return None
            path = self.directory / filename
            if path.is_symlink() or not path.is_file():
                return None
            kind = "original" if filename == row["original"] else "denoised"
            return path, kind

    def delete(self, owner: str, recording_id: str) -> bool:
        with self.lock, closing(self.connect()) as db, db:
            row = db.execute("SELECT original, denoised FROM recordings WHERE id=? AND owner=?",
                             (recording_id, owner)).fetchone()
            if not row:
                return False
            db.execute("DELETE FROM recordings WHERE id=? AND owner=?", (recording_id, owner))
            for name in row:
                self.remove_file(db, name)
        return True

    def remove_file(self, db, name):
        try:
            (self.directory / name).unlink(missing_ok=True)
            db.execute("DELETE FROM pending_deletes WHERE filename=?", (name,))
        except OSError:
            db.execute("INSERT OR IGNORE INTO pending_deletes VALUES (?)", (name,))

    def cleanup(self) -> None:
        now = time.time()
        with self.lock, closing(self.connect()) as db, db:
            expired = db.execute("SELECT original, denoised FROM recordings WHERE expires<=?",
                                 (now,)).fetchall()
            db.execute("DELETE FROM recordings WHERE expires<=?", (now,))
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            unused = db.execute(
                "SELECT owner FROM sessions WHERE last_seen<=? AND NOT EXISTS "
                "(SELECT 1 FROM recordings WHERE recordings.owner=sessions.owner AND recordings.expires>?)",
                (now - SESSION_IDLE_SECONDS, now)).fetchall()
            for row in unused:
                if row["owner"] not in self.active_owners:
                    db.execute("DELETE FROM sessions WHERE owner=?", (row["owner"],))
            for row in expired:
                for name in row:
                    self.remove_file(db, name)
            for row in db.execute("SELECT filename FROM pending_deletes").fetchall():
                self.remove_file(db, row["filename"])
            # Also retire legacy/unregistered files without ever serving them publicly.
            for path in self.directory.glob("*.wav"):
                with suppress(OSError):
                    if path.stat().st_mtime <= now - self.ttl:
                        path.unlink()

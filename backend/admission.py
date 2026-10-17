"""Bounded, per-session admission for the single file-processing worker."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import time


logger = logging.getLogger("uvicorn.error.capacity")


class AdmissionRejected(Exception):
    def __init__(self, detail, status=503):
        super().__init__(detail)
        self.status = status
        self.retry_after = "3"


class FileAdmission:
    def __init__(self, max_waiting=3, wait_seconds=20):
        self.max_waiting = max_waiting
        self.wait_seconds = wait_seconds
        self.owners = set()
        self.active = 0
        self.completed = self.failed = self.rejected = self.timed_out = 0

    @asynccontextmanager
    async def hold(self, owner, slots, kind="denoise"):
        if owner in self.owners:
            self.rejected += 1
            raise AdmissionRejected("An audio job for this session is already running or waiting.", 429)
        if len(self.owners) >= self.max_waiting + 1:
            self.rejected += 1
            raise AdmissionRejected("The audio queue is full. Please try again shortly.")
        self.owners.add(owner)
        admitted = time.monotonic()
        acquired = False
        outcome = "cancelled"
        started = None
        try:
            try:
                await asyncio.wait_for(slots.acquire(), timeout=self.wait_seconds)
            except asyncio.TimeoutError as error:
                self.timed_out += 1
                outcome = "queue_timeout"
                raise AdmissionRejected("The audio queue wait timed out. Please try again shortly.") from error
            acquired = True
            self.active += 1
            started = time.monotonic()
            try:
                yield
            except BaseException:
                self.failed += 1
                outcome = "failed"
                raise
            else:
                self.completed += 1
                outcome = "completed"
        finally:
            self.owners.discard(owner)
            if acquired:
                self.active -= 1
                slots.release()
            now = time.monotonic()
            logger.info("audio_capacity %s", json.dumps({
                "kind": kind, "outcome": outcome,
                "waitSeconds": round((started or now) - admitted, 3),
                "workSeconds": round(now - started, 3) if started is not None else 0,
                "active": self.active, "waiting": len(self.owners) - self.active,
                "completed": self.completed, "failed": self.failed,
                "rejected": self.rejected, "queueTimeouts": self.timed_out,
            }, separators=(",", ":")))

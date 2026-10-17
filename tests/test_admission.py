import asyncio
import sqlite3
import threading
import time
import unittest
from unittest.mock import patch

from backend.admission import AdmissionRejected, FileAdmission
from backend.app import media_cleanup_loop, processing_thread, send_live


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_is_fifo_and_rejects_excess_without_taking_a_slot(self):
        gate, slots = FileAdmission(max_waiting=2), asyncio.Semaphore(1)
        started, release = asyncio.Event(), asyncio.Event()
        order = []

        async def work(owner):
            async with gate.hold(owner, slots):
                order.append(owner)
                if owner == "first":
                    started.set()
                    await release.wait()

        first = asyncio.create_task(work("first"))
        await started.wait()
        second = asyncio.create_task(work("second"))
        third = asyncio.create_task(work("third"))
        await asyncio.sleep(0.01)
        with self.assertRaises(AdmissionRejected) as rejected:
            async with gate.hold("fourth", slots):
                self.fail("Full queue admitted a job")
        self.assertEqual(rejected.exception.status, 503)
        self.assertEqual(gate.active, 1)
        release.set()
        await asyncio.gather(first, second, third)
        self.assertEqual(order, ["first", "second", "third"])
        self.assertEqual(gate.completed, 3)
        self.assertEqual(slots._value, 1)
        self.assertFalse(gate.owners)

    async def test_duplicate_owner_cannot_occupy_active_or_waiting_capacity(self):
        gate, slots = FileAdmission(), asyncio.Semaphore(0)
        waiting = asyncio.create_task(gate.hold("owner", slots).__aenter__())
        await asyncio.sleep(0.01)
        with self.assertRaises(AdmissionRejected) as rejected:
            async with gate.hold("owner", slots):
                self.fail("Duplicate owner admitted")
        self.assertEqual(rejected.exception.status, 429)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        self.assertFalse(gate.owners)
        self.assertEqual(slots._value, 0)

    async def test_queue_timeout_and_failed_job_do_not_leak_or_over_release(self):
        gate, slots = FileAdmission(wait_seconds=0.01), asyncio.Semaphore(0)
        with self.assertRaises(AdmissionRejected):
            async with gate.hold("owner", slots):
                self.fail("A slot must be acquired")
        self.assertEqual(gate.timed_out, 1)
        self.assertFalse(gate.owners)
        self.assertEqual(slots._value, 0)
        slots.release()
        with self.assertRaises(ValueError):
            async with gate.hold("owner", slots):
                raise ValueError("test failure")
        self.assertEqual(gate.failed, 1)
        self.assertEqual(gate.active, 0)
        self.assertEqual(slots._value, 1)

    async def test_cancelled_native_work_keeps_its_slot_until_the_thread_exits(self):
        gate, slots = FileAdmission(), asyncio.Semaphore(1)
        entered, release = threading.Event(), threading.Event()

        def native_work():
            entered.set()
            release.wait(timeout=2)

        async def run():
            async with gate.hold("first", slots):
                await processing_thread(native_work)

        task = asyncio.create_task(run())
        while not entered.is_set():
            await asyncio.sleep(0.001)
        try:
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            self.assertEqual(gate.active, 1)
            self.assertTrue(slots.locked())
        finally:
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(gate.owners)
        self.assertEqual(slots._value, 1)

    async def test_queue_logs_contain_aggregates_not_session_identifiers(self):
        gate = FileAdmission()
        with self.assertLogs("uvicorn.error.capacity", level="INFO") as logs:
            async with gate.hold("private-owner-sentinel", asyncio.Semaphore(1)):
                pass
        self.assertNotIn("private-owner-sentinel", "".join(logs.output))
        self.assertIn('"outcome":"completed"', logs.output[0])

    async def test_stalled_microphone_receiver_has_a_bounded_send(self):
        class Socket:
            async def send_bytes(self, payload):
                await asyncio.sleep(10)

        with patch("backend.app.LIVE_SEND_SECONDS", 0.01):
            with self.assertRaises(asyncio.TimeoutError):
                await send_live(Socket(), "send_bytes", b"audio", time.monotonic() + 10)

    async def test_media_cleanup_recovers_from_temporary_storage_errors(self):
        with patch("backend.app.cleanup_media", side_effect=[OSError("private-path"),
                    sqlite3.OperationalError("private-database"), None]) as cleanup, \
                patch("backend.app.asyncio.sleep", side_effect=[None, None, asyncio.CancelledError()]), \
                self.assertLogs("uvicorn.error", level="WARNING") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await media_cleanup_loop()
        self.assertEqual(cleanup.call_count, 3)
        self.assertEqual(len(logs.output), 2)
        self.assertNotIn("private-", "".join(logs.output))


if __name__ == "__main__":
    unittest.main()

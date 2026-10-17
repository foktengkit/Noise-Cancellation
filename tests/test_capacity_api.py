import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

import backend.app as server
from backend.denoise import write_wav_bytes


class CapacityApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.stack = ExitStack()
        root = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(patch.object(server, "MEDIA_DIR", Path(root)))
        self.lifespan = server.app.router.lifespan_context(server.app)
        await self.lifespan.__aenter__()
        self.clients = []
        for index in range(5):
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                      base_url="http://testserver", headers={"origin": "http://testserver"})
            await client.get("/api/session")
            self.clients.append(client)
        self.tasks = []
        self.wav = write_wav_bytes(np.zeros(160, np.float32), 8000)
        self.metadata = {"sampleRate": 8000, "channels": 1, "durationSeconds": .02,
                         "inputFormat": "wav", "strength": .85}

    async def asyncTearDown(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for client in self.clients:
            await client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.stack.close()

    async def wait_for_owners(self, count):
        async with asyncio.timeout(2):
            while len(server.app.state.file_admission.owners) != count:
                await asyncio.sleep(.001)

    def submit(self, index):
        task = asyncio.create_task(self.clients[index].post(
            "/api/denoise", files={"file": ("test.wav", self.wav)}))
        self.tasks.append(task)
        return task

    async def test_burst_is_bounded_fair_and_does_not_block_health_or_leak_audio(self):
        entered, release = asyncio.Event(), asyncio.Event()
        parsed = []

        async def slow_parse(request):
            parsed.append(request.cookies.get("noise_lift_session"))
            if len(parsed) == 1:
                entered.set()
                await release.wait()
            return self.wav, "test.wav", "audio/wav", .85, "off"

        with patch.object(server, "parse_upload", side_effect=slow_parse), \
                patch.object(server, "process_file", return_value=(self.wav, self.wav, self.metadata)):
            first = self.submit(0)
            await entered.wait()
            rest = []
            for index in (1, 2, 3):
                rest.append(self.submit(index))
                await self.wait_for_owners(index + 1)
            overflow = await self.clients[4].post("/api/denoise", content=b"must-not-be-parsed")
            duplicate = await self.clients[0].post("/api/denoise", content=b"must-not-be-parsed")
            self.assertEqual(overflow.status_code, 503)
            self.assertEqual(duplicate.status_code, 429)
            self.assertEqual(overflow.headers["retry-after"], "3")
            self.assertEqual(len(parsed), 1)
            async with asyncio.timeout(.5):
                self.assertEqual((await self.clients[4].get("/api/health")).status_code, 200)
            release.set()
            responses = await asyncio.gather(first, *rest)
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(parsed, [self.clients[index].cookies.get("noise_lift_session") for index in range(4)])
        for index, response in enumerate(responses):
            url = response.json()["originalUrl"]
            self.assertEqual((await self.clients[index].get(url)).status_code, 200)
            self.assertEqual((await self.clients[(index + 1) % 5].get(url)).status_code, 404)
        self.assertFalse(server.app.state.file_admission.owners)
        self.assertFalse(server.app.state.store.active_owners)
        self.assertEqual(server.app.state.file_slots._value, 1)

    async def test_queued_request_cancellation_releases_only_its_reservation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_parse(request):
            entered.set()
            await release.wait()
            return self.wav, "test.wav", "audio/wav", .85, "off"

        with patch.object(server, "parse_upload", side_effect=slow_parse), \
                patch.object(server, "process_file", return_value=(self.wav, self.wav, self.metadata)):
            first = self.submit(0)
            await entered.wait()
            waiting = self.submit(1)
            await self.wait_for_owners(2)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertEqual(len(server.app.state.file_admission.owners), 1)
            self.assertEqual(server.app.state.file_slots._value, 0)
            release.set()
            self.assertEqual((await first).status_code, 200)
        self.assertFalse(server.app.state.store.active_owners)
        self.assertEqual(server.app.state.file_slots._value, 1)


if __name__ == "__main__":
    unittest.main()

"""Loopback-only multi-user file/live smoke test; never targets public services."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
import numpy as np
from websockets.sync.client import connect

from backend.denoise import read_audio_bytes, write_wav_bytes


async def run(url, output, users=8):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError("Capacity smoke tests accept loopback service URLs only.")
    if not 4 <= users <= 12:
        raise ValueError("Use 4 to 12 file users for the bounded local test.")
    origin = parsed.scheme + "://" + parsed.netloc
    socket_url = ("wss" if parsed.scheme == "https" else "ws") + "://" + parsed.netloc + "/api/live"
    clients = [httpx.AsyncClient(base_url=origin, headers={"origin": origin}, timeout=90) for _ in range(users + 2)]
    probe = httpx.AsyncClient(base_url=origin, timeout=5)
    statuses, health_times, file_times = Counter(), [], []
    done = asyncio.Event()
    started = time.perf_counter()

    async def health_loop():
        while not done.is_set():
            start = time.perf_counter()
            response = await probe.get("/api/health")
            response.raise_for_status()
            health_times.append(time.perf_counter() - start)
            await asyncio.sleep(.1)

    async def upload(index):
        rate = 16000
        t = np.arange(rate) / rate
        source = (.08 * np.sin(2 * np.pi * (220 + index * 31) * t)
                  + np.random.default_rng(index).normal(0, .01, rate)).astype(np.float32)
        wav = write_wav_bytes(source, rate)
        start = time.perf_counter()
        for attempt in range(4):
            response = await clients[index].post("/api/denoise", files={"file": ("capacity.wav", wav)})
            statuses[response.status_code] += 1
            if response.status_code in {429, 503} and attempt < 3:
                await asyncio.sleep(min(6, max(1, int(response.headers.get("retry-after", "3")))))
                continue
            response.raise_for_status()
            result = response.json()
            file_times.append(time.perf_counter() - start)
            return index, result
        raise AssertionError("Upload did not finish")

    def capture(index, cookie):
        rate = 16000
        source = np.random.default_rng(index).normal(0, .02, rate * 4).astype("<f4")
        pieces, times = [], []
        with connect(socket_url, origin=origin, additional_headers={"Cookie": cookie},
                     open_timeout=15, max_size=131072) as socket:
            socket.send(json.dumps({"type": "start", "sampleRate": rate}))
            assert json.loads(socket.recv(timeout=15))["type"] == "ready"
            for start in range(0, len(source), rate // 4):
                timestamp = time.perf_counter()
                socket.send(source[start:start + rate // 4].tobytes())
                received = socket.recv(timeout=15)
                assert isinstance(received, bytes), received
                times.append(time.perf_counter() - timestamp)
                pieces.append(np.frombuffer(received, dtype="<f4"))
                time.sleep(.15)
            socket.send(json.dumps({"type": "stop"}))
            while True:
                message = socket.recv(timeout=15)
                if isinstance(message, bytes):
                    pieces.append(np.frombuffer(message, dtype="<f4"))
                else:
                    result = json.loads(message)
                    assert result["type"] == "complete", result
                    break
        returned = np.concatenate(pieces)
        assert len(returned) == len(source) and np.isfinite(returned).all()
        return index, result, times

    health_task = None
    jobs = []
    results = []
    try:
        for client in clients:
            (await client.get("/api/session")).raise_for_status()
        health_task = asyncio.create_task(health_loop())
        jobs = [asyncio.create_task(upload(index)) for index in range(users)]
        for index in range(users, users + 2):
            cookie = "; ".join(name + "=" + value for name, value in clients[index].cookies.items())
            jobs.append(asyncio.create_task(asyncio.to_thread(capture, index, cookie)))
        results = await asyncio.gather(*jobs)
        for item in results:
            index, result = item[:2]
            response = await clients[index].get(result["denoisedUrl"])
            response.raise_for_status()
            audio = read_audio_bytes(response.content, "test.wav")
            assert np.isfinite(audio.samples).all()
            assert (await clients[(index + 1) % len(clients)].get(result["denoisedUrl"])).status_code == 404
        done.set()
        await health_task
        assert len(health_times) >= 2
        assert max(health_times) < 3, "Health endpoint stalled during mixed load"
        live_times = [value for item in results if len(item) == 3 for value in item[2]]
        report = {"fileUsers": users, "simultaneousLiveStreams": 2, "completed": len(results),
                  "uploadStatusCounts": dict(statuses), "healthProbes": len(health_times),
                  "healthP95Seconds": round(float(np.percentile(health_times, 95)), 4),
                  "liveChunkP95Seconds": round(float(np.percentile(live_times, 95)), 4),
                  "fileP95SecondsIncludingRetries": round(float(np.percentile(file_times, 95)), 3),
                  "elapsedSeconds": round(time.perf_counter() - started, 3),
                  "crossSessionAccessDenied": True, "scope": "Local smoke load, not a production capacity guarantee"}
    finally:
        done.set()
        # Join bounded uploads and native capture threads before deleting their outputs.
        await asyncio.gather(*jobs, return_exceptions=True)
        if health_task is not None:
            await asyncio.gather(health_task, return_exceptions=True)
        cleanup_errors = []
        for client in clients:
            try:
                (await client.delete("/api/recordings")).raise_for_status()
                assert (await client.get("/api/recordings")).json()["recordings"] == []
            except Exception as error:
                cleanup_errors.append(error)
            finally:
                await client.aclose()
        await probe.aclose()
        if cleanup_errors:
            raise ExceptionGroup("Test recording cleanup failed", cleanup_errors)
    report["testRecordingsDeleted"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8006")
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/capacity/metrics.json"))
    parser.add_argument("--users", type=int, default=8)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.output, args.users))

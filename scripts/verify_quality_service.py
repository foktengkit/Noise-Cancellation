"""Check a running service against local inference using an explicit test clip."""

import argparse
import io
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import av
import httpx
import numpy as np
from websockets.sync.client import connect

from backend.denoise import read_audio_bytes, reduce_noise, write_wav_bytes


def encode_compressed(samples, rate, extension):
    target = io.BytesIO()
    with av.open(target, mode="w", format="mp3" if extension == "mp3" else "mp4") as container:
        stream = container.add_stream("mp3" if extension == "mp3" else "aac", rate=rate)
        stream.layout = "mono"
        stream.bit_rate = 64000
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="fltp", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return target.getvalue()


def verify(path, url, output, strength=1, hum_filter="off"):
    url = url.rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("An HTTP or HTTPS service URL is required.")
    origin = parsed.scheme + "://" + parsed.netloc
    source = read_audio_bytes(path.read_bytes(), path.name)
    if source.channels != 1:
        raise ValueError("Use a mono test clip for the microphone comparison.")
    rate, samples = source.sample_rate, source.samples[:, 0]
    report = {"fixture": path.name, "service": url, "sampleRate": rate, "strength": strength, "humFilter": hum_filter, "uploads": []}
    owned = []
    with httpx.Client(base_url=url, headers={"origin": origin}, timeout=60) as client:
        client.get("/api/session").raise_for_status()
        try:
            for extension in ("wav", "mp3", "m4a"):
                payload = write_wav_bytes(samples, rate) if extension == "wav" else encode_compressed(samples, rate, extension)
                filename = "quality-test." + extension
                decoded = read_audio_bytes(payload, filename)
                response = client.post("/api/denoise", files={"file": (filename, payload)}, data={"strength": str(strength), "humFilter": hum_filter})
                response.raise_for_status()
                result = response.json()
                owned.append(result["recordingId"])
                assert result["engine"] == "rnnoise"
                assert result["strength"] == strength and result["humFilter"] == hum_filter
                downloaded = client.get(result["denoisedUrl"])
                downloaded.raise_for_status()
                actual = read_audio_bytes(downloaded.content, "result.wav").samples
                expected = read_audio_bytes(write_wav_bytes(reduce_noise(decoded.samples, decoded.sample_rate, strength, hum_filter),
                                                           decoded.sample_rate), "expected.wav").samples
                np.testing.assert_allclose(actual, expected, atol=1 / 32768, rtol=0)
                report["uploads"].append({"format": extension, "samples": len(actual),
                                           "maxPcmError": float(np.max(np.abs(actual - expected))),
                                           "matchesLocalInference": True})
                client.delete("/api/recordings/" + owned[-1]).raise_for_status()
                owned.pop()
            cookie = "; ".join(name + "=" + value for name, value in client.cookies.items())
            socket_url = ("wss" if parsed.scheme == "https" else "ws") + "://" + parsed.netloc + "/api/live"
            returned = []
            with connect(socket_url, origin=origin, additional_headers={"Cookie": cookie},
                         open_timeout=15, max_size=131072) as socket:
                socket.send(json.dumps({"type": "start", "sampleRate": rate, "strength": strength, "humFilter": hum_filter}))
                ready = json.loads(socket.recv(timeout=15))
                assert ready["type"] == "ready" and ready["engine"] == "rnnoise", ready
                for start in range(0, len(samples), rate // 4):
                    socket.send(samples[start:start + rate // 4].astype("<f4").tobytes())
                    message = socket.recv(timeout=15)
                    if not isinstance(message, bytes):
                        raise AssertionError(message)
                    returned.append(np.frombuffer(message, dtype="<f4"))
                    # Respect admission limits; this is not a network latency benchmark.
                    time.sleep(0.06)
                socket.send(json.dumps({"type": "stop"}))
                while True:
                    message = socket.recv(timeout=15)
                    if isinstance(message, bytes):
                        returned.append(np.frombuffer(message, dtype="<f4"))
                    else:
                        result = json.loads(message)
                        assert result["type"] == "complete", result
                        owned.append(result["recordingId"])
                        break
            streamed = np.concatenate(returned)
            expected = reduce_noise(samples, rate, strength, hum_filter)[:, 0]
            np.testing.assert_allclose(streamed, expected, atol=1e-6, rtol=0)
            saved = client.get(result["denoisedUrl"])
            saved.raise_for_status()
            assert saved.content == write_wav_bytes(streamed, rate)
            report["live"] = {"samples": len(streamed), "maxFloatError": float(np.max(np.abs(streamed - expected))),
                              "savedMatchesReturnedAudio": True, "matchesLocalInference": True}
        finally:
            for recording in owned:
                client.delete("/api/recordings/" + recording).raise_for_status()
        remaining = client.get("/api/recordings")
        remaining.raise_for_status()
        assert remaining.json()["recordings"] == []
        report["testRecordingsDeleted"] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/quality-recheck/service.json"))
    parser.add_argument("--strength", type=float, default=1)
    parser.add_argument("--hum-filter", choices=("off", "50", "60"), default="off")
    args = parser.parse_args()
    verify(args.fixture, args.url, args.output, args.strength, args.hum_filter)

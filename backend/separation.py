"""Server-configured private ML transport, never a client-supplied URL."""

import logging
import http.client
import os
import time
import urllib.parse

from .denoise import AudioError, read_wav_bytes, write_wav_bytes


MAX_SEPARATION_SECONDS = 15
MODEL_RATE = 8000
MAX_RESPONSE_BYTES = MODEL_RATE * MAX_SEPARATION_SECONDS * 4 + 1024


def settings():
    url = os.environ.get("SEPARATION_URL", "").rstrip("/")
    key = os.environ.get("SEPARATION_API_KEY", "")
    if not url and not key:
        return None
    try:
        parsed = urllib.parse.urlsplit(url)
        parsed.port
    except ValueError as error:
        raise ValueError("Invalid private separation service configuration.") from error
    allow_http = os.environ.get("SEPARATION_ALLOW_HTTP", "0") == "1"
    if (parsed.scheme not in (("http", "https") if allow_http else ("https",)) or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or len(key) < 32 or not key.isascii() or not key.isprintable() or any(character.isspace() for character in key)):
        raise ValueError("Invalid private separation service configuration.")
    return url + "/v1/separate", key


def separate_audio(payload, duration):
    config = settings()
    if config is None:
        raise AudioError("The private separation service is not configured.")
    if not 0 < duration <= MAX_SEPARATION_SECONDS:
        raise AudioError("Speaker separation accepts at most 15 seconds per recording.")
    endpoint, key = config
    parsed = urllib.parse.urlsplit(endpoint)
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=130)
    deadline = time.monotonic() + 140
    try:
        # No environment proxy, redirects, cookies, or premature Connection: close.
        connection.request("POST", parsed.path, body=payload,
                           headers={"Authorization": "Bearer " + key, "Content-Type": "audio/wav"})
        with connection.getresponse() as response:
            if response.status != 200 or response.getheader("Content-Type", "").split(";", 1)[0] != "audio/wav":
                logging.getLogger(__name__).warning("Separation upstream response status=%s", response.status)
                raise AudioError("The separation service returned an invalid response.")
            data = bytearray()
            while chunk := response.read(min(65536, MAX_RESPONSE_BYTES + 1 - len(data))):
                data.extend(chunk)
                if len(data) > MAX_RESPONSE_BYTES or time.monotonic() > deadline:
                    raise AudioError("The separation response exceeded its limits.")
    except (http.client.HTTPException, OSError, TimeoutError) as error:
        logging.getLogger(__name__).warning("Separation transport failure: %s status=%s",
                                            type(error).__name__, getattr(error, "code", None))
        raise AudioError("The separation service is unavailable or timed out.") from error
    finally:
        connection.close()
    audio = read_wav_bytes(bytes(data))
    expected_frames = round(duration * MODEL_RATE)
    if audio.sample_rate != MODEL_RATE or audio.channels != 2 or abs(len(audio.samples) - expected_frames) > 16:
        raise AudioError("The separation service returned incompatible audio.")
    return [write_wav_bytes(audio.samples[:, index], MODEL_RATE) for index in range(2)]


def save_sources(store, owner, original, sources, metadata):
    results = []
    with store.lock:
        try:
            for index, source in enumerate(sources):
                results.append(store.save(owner, original, source, {
                    **metadata, "originalSampleRate": metadata["sampleRate"], "originalChannels": metadata["channels"],
                    "sampleRate": MODEL_RATE, "channels": 1, "engine": "sepformer", "sourceIndex": index + 1,
                    "strength": 1, "humFilter": "off"}))
        except BaseException:
            for result in results:
                store.delete(owner, result["recordingId"])
            raise
    return {"sources": results}

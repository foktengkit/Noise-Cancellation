from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import deque
from contextlib import ExitStack, asynccontextmanager, suppress
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import ClientDisconnect

from .admission import AdmissionRejected, FileAdmission
from .denoise import (DEFAULT_STRENGTH, HUM_FILTERS, MAX_AUDIO_SECONDS, MAX_SAMPLE_RATE,
                      AudioError, StreamingNoiseReducer, validate_hum_filter, write_wav_bytes)
from .recording import WavSpool
from .security import RateLimits, SecurityHeaders, same_origin
from . import separation
from .speech import verify_model
from .storage import MediaStore, PRIVATE_HEADERS, SESSION_COOKIE, SESSION_SECONDS, StorageFull
from .tls_probe import run_startup_tls_check
from .worker import process_file


BASE_DIR = Path(__file__).resolve().parent.parent
APP_VERSION = "0.5.4"
FRONTEND_DIR = BASE_DIR / "frontend"
MEDIA_DIR = Path(os.environ.get("MEDIA_DIR", str(BASE_DIR / "media")))
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
UPLOAD_SECONDS = 30
MEDIA_TTL_SECONDS = 60 * 60
LIVE_PROGRESS_SECONDS = 10
LIVE_GRACE_SECONDS = 15
LIVE_SEND_SECONDS = 5
MAX_LIVE_CHUNKS = 1024
IP_RATE_LIMITS = {"session": 120, "upload": 60, "live": 100}
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "0") == "1"


def cleanup_media() -> None:
    app.state.store.cleanup()


async def media_cleanup_loop():
    while True:
        try:
            await run_in_threadpool(cleanup_media)
        except (OSError, sqlite3.Error):
            logging.getLogger("uvicorn.error").warning("Media cleanup failed; retrying on the next interval.")
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(application):
    separation.settings()
    await run_in_threadpool(verify_model)
    application.state.store = MediaStore(
        MEDIA_DIR, MEDIA_TTL_SECONDS,
        max_bytes=int(os.environ.get("MAX_MEDIA_BYTES", 256 * 1024 * 1024)),
        owner_bytes=int(os.environ.get("MAX_OWNER_MEDIA_BYTES", 64 * 1024 * 1024)),
    )
    application.state.file_slots = asyncio.Semaphore(1)
    application.state.file_admission = FileAdmission()
    application.state.live_slots = asyncio.Semaphore(2)
    application.state.live_owners = set()
    application.state.rates = RateLimits()
    task = asyncio.create_task(media_cleanup_loop())
    tls_task = (asyncio.create_task(run_in_threadpool(run_startup_tls_check))
                if os.environ.get("TLS_STARTUP_CHECK") == "1" else None)
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if tls_task is not None:
            tls_task.cancel()
            with suppress(asyncio.CancelledError):
                await tls_task


app = FastAPI(title="Noise Control", version=APP_VERSION, lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(SecurityHeaders, secure=COOKIE_SECURE)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.environ.get(
    "ALLOWED_HOSTS", os.environ.get("RENDER_EXTERNAL_HOSTNAME", "localhost,127.0.0.1,[::1],testserver")
).split(","))


@app.exception_handler(HTTPException)
async def http_error(request, error):
    return JSONResponse({"detail": error.detail}, status_code=error.status_code,
                        headers={**PRIVATE_HEADERS, **(error.headers or {})})


@app.exception_handler(AdmissionRejected)
async def capacity_error(request, error):
    return JSONResponse({"detail": str(error)}, status_code=error.status,
                        headers={**PRIVATE_HEADERS, "Retry-After": error.retry_after})


def client_ip(connection) -> str:
    return connection.client.host if connection.client else "unknown"


def check_rate(connection, action: str, owner: str | None = None, limit: int = 10):
    rates = app.state.rates
    # A reverse proxy or NAT can legitimately serve many independent sessions.
    if not rates.allow(action + ":ip:" + client_ip(connection), IP_RATE_LIMITS.get(action, limit * 10)) or (
        owner and not rates.allow(action + ":owner:" + owner, limit)
    ):
        raise HTTPException(429, "Too many requests. Please wait a minute.", headers={"Retry-After": "60"})


async def require_owner(request: Request, required_origin: bool = True) -> str:
    if not same_origin(request.scope, COOKIE_SECURE, required=required_origin):
        raise HTTPException(403, "A same-origin request is required.")
    owner = await run_in_threadpool(app.state.store.owner, request.cookies.get(SESSION_COOKIE))
    if not owner:
        raise HTTPException(401, "Reload this page to establish a private session.")
    return owner


@app.get("/api/session")
async def session(request: Request):
    if not same_origin(request.scope, COOKIE_SECURE, required=False):
        raise HTTPException(403, "Cross-site session creation is not allowed.")
    check_rate(request, "session", limit=30)
    cookie = request.cookies.get(SESSION_COOKIE)
    owner = await run_in_threadpool(app.state.store.owner, cookie)
    response = JSONResponse({"status": "ok"}, headers=PRIVATE_HEADERS)
    if not owner:
        try:
            cookie = await run_in_threadpool(app.state.store.new_session)
        except StorageFull as exc:
            raise HTTPException(503, str(exc)) from exc
        response.set_cookie(SESSION_COOKIE, cookie, max_age=SESSION_SECONDS,
                            httponly=True, secure=COOKIE_SECURE or request.url.scheme == "https",
                            samesite="strict", path="/")
    return response


@app.get("/api/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/config")
def config():
    return {"maxUploadBytes": MAX_UPLOAD_BYTES, "maxDurationSeconds": MAX_AUDIO_SECONDS,
            "chunkSeconds": 0.25, "mediaRetentionSeconds": MEDIA_TTL_SECONDS,
            "engine": "rnnoise", "defaultStrength": DEFAULT_STRENGTH,
            "humFilters": HUM_FILTERS, "defaultHumFilter": "off",
            "separationEnabled": separation.settings() is not None,
            "maxSeparationDurationSeconds": separation.MAX_SEPARATION_SECONDS}


def validate_strength(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise AudioError("Noise reduction strength must be between 0 and 1.")
    return value


async def parse_upload(request: Request):
    limit = MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
    length = request.headers.get("content-length")
    if length is not None:
        if not length.isascii() or not length.isdigit() or len(length) > 12:
            raise HTTPException(400, "Invalid Content-Length.")
        if int(length) > limit:
            raise HTTPException(413, "The upload request is too large.")
    if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
        raise HTTPException(415, "Expected a multipart audio upload.")

    # Bound the entire stream before any multipart parsing, including chunked uploads.
    with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as body:
        total = 0
        async with asyncio.timeout(UPLOAD_SECONDS):
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit:
                    raise HTTPException(413, "The upload request is too large.")
                await run_in_threadpool(body.write, chunk)
        body.seek(0)

        async def bounded_body():
            while chunk := await run_in_threadpool(body.read, 64 * 1024):
                yield chunk

        try:
            form = await MultiPartParser(request.headers, bounded_body(), max_files=1,
                                         max_fields=2, max_part_size=1024).parse()
        except MultiPartException as exc:
            raise HTTPException(400, "Invalid upload form or too many parts.") from exc
        try:
            parts = form.multi_items()
            if any(key not in {"file", "strength", "humFilter"} for key, value in parts):
                raise HTTPException(400, "Unexpected upload field.")
            if sum(key == "file" for key, value in parts) != 1:
                raise HTTPException(400, "Exactly one audio file is required.")
            if any(sum(key == field for key, value in parts) > 1 for field in ("strength", "humFilter")):
                raise HTTPException(400, "Duplicate processing settings are not allowed.")
            file = form.get("file")
            if not isinstance(file, UploadFile) or isinstance(form.get("strength"), UploadFile):
                raise HTTPException(400, "Invalid audio upload fields.")
            strength = validate_strength(form.get("strength", str(DEFAULT_STRENGTH)))
            hum_filter = validate_hum_filter(form.get("humFilter", "off"))
            payload = await file.read(MAX_UPLOAD_BYTES + 1)
            if not payload:
                raise AudioError("Please upload a WAV, MP3, or M4A audio file.")
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "The upload is larger than 25 MB.")
            return payload, (file.filename or "")[:255], file.content_type, strength, hum_filter
        finally:
            await form.close()


async def processing_thread(function, *args):
    # Cancelling a request must not free its slot while native work still runs.
    task = asyncio.create_task(run_in_threadpool(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not task.cancelled():
            task.exception()
        raise


@app.post("/api/denoise")
async def denoise_audio(request: Request):
    owner = await require_owner(request)
    check_rate(request, "upload", owner, limit=6)
    app.state.store.pin(owner)
    try:
        async with app.state.file_admission.hold(owner, app.state.file_slots):
            payload, filename, content_type, strength, hum_filter = await parse_upload(request)
            original, denoised, metadata = await processing_thread(
                process_file, payload, filename, content_type, strength, hum_filter)
            result = await processing_thread(app.state.store.save, owner, original, denoised, metadata)
            return JSONResponse(result, headers=PRIVATE_HEADERS)
    except StorageFull as exc:
        raise HTTPException(507, str(exc)) from exc
    except (AudioError, ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(408, "The audio upload or processing timed out.") from exc
    except ClientDisconnect as exc:
        raise HTTPException(408, "The upload connection was interrupted.") from exc
    except OSError as exc:
        raise HTTPException(503, "Audio processing or storage is unavailable.") from exc
    finally:
        app.state.store.unpin(owner)


@app.post("/api/separate")
async def separate_recording(request: Request):
    owner = await require_owner(request)
    check_rate(request, "separate", owner, limit=2)
    if separation.settings() is None:
        raise HTTPException(503, "Speaker separation is not configured on this server.")
    app.state.store.pin(owner)
    try:
        async with app.state.file_admission.hold(owner, app.state.file_slots, "separate"):
            payload, filename, content_type, _, _ = await parse_upload(request)
            original, _, metadata = await processing_thread(process_file, payload, filename, content_type, 0, "off", "decode")
            duration = metadata["inputFrames"] / metadata["sampleRate"]
            try:
                sources = await processing_thread(separation.separate_audio, original, duration)
            except AudioError as error:
                raise HTTPException(503, "Speaker separation is unavailable. Please try again later.") from error
            result = await processing_thread(separation.save_sources, app.state.store, owner, original, sources, metadata)
            return JSONResponse(result, headers=PRIVATE_HEADERS)
    except StorageFull as error:
        raise HTTPException(507, str(error)) from error
    except (AudioError, ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    except TimeoutError as error:
        raise HTTPException(408, "The upload or decoding timed out.") from error
    except ClientDisconnect as error:
        raise HTTPException(408, "The upload connection was interrupted.") from error
    except OSError as error:
        raise HTTPException(503, "Audio processing or storage is unavailable.") from error
    finally:
        app.state.store.unpin(owner)


@app.api_route("/media/{filename}", methods=["GET", "HEAD"])
def private_audio(filename: str, request: Request):
    owner = app.state.store.owner(request.cookies.get(SESSION_COOKIE))
    result = app.state.store.locate(owner, filename)
    if result is None:
        raise HTTPException(404, "Audio not found or expired.")
    path, kind = result
    return FileResponse(path, media_type="audio/wav", filename=kind + ".wav",
                        content_disposition_type="inline", headers=PRIVATE_HEADERS)


@app.delete("/api/recordings/{recording_id}")
async def delete_recording(recording_id: str, request: Request):
    owner = await require_owner(request)
    check_rate(request, "manage", owner, limit=60)
    if not re.fullmatch(r"[0-9a-f]{32}", recording_id):
        raise HTTPException(404, "Recording not found.")
    if not await run_in_threadpool(app.state.store.delete, owner, recording_id):
        raise HTTPException(404, "Recording not found.")
    return Response(status_code=204, headers=PRIVATE_HEADERS)


@app.get("/api/recordings")
async def list_recordings(request: Request):
    owner = await require_owner(request, required_origin=False)
    check_rate(request, "list", owner, limit=60)
    result = await run_in_threadpool(app.state.store.list_recordings, owner)
    return JSONResponse(result, headers=PRIVATE_HEADERS)


@app.delete("/api/recordings")
async def clear_recordings(request: Request):
    owner = await require_owner(request)
    check_rate(request, "manage", owner, limit=60)
    await run_in_threadpool(app.state.store.clear_owner, owner)
    return Response(status_code=204, headers=PRIVATE_HEADERS)


def read_control(text):
    if len(text) > 1024:
        raise AudioError("The control message is too large.")
    message = json.loads(text)
    if not isinstance(message, dict):
        raise AudioError("Expected a JSON control object.")
    return message


def save_live(owner, original, cleaned, sample_rate, strength, hum_filter="off"):
    if original.frames != cleaned.frames:
        raise AudioError("The speech filter returned incomplete audio.")
    return app.state.store.save(owner, original.read_wav(), cleaned.read_wav(), {
        "sampleRate": sample_rate, "channels": 1,
        "durationSeconds": round(original.frames / sample_rate, 2),
        "strength": strength, "inputFormat": "microphone", "engine": "rnnoise", "humFilter": hum_filter})


async def send_live(websocket, method, payload, deadline):
    remaining = min(LIVE_SEND_SECONDS, deadline - time.monotonic())
    if remaining <= 0:
        raise asyncio.TimeoutError("Microphone connection timed out.")
    await asyncio.wait_for(getattr(websocket, method)(payload), timeout=remaining)


@app.websocket("/api/live")
async def live_audio(websocket: WebSocket):
    owner = await run_in_threadpool(app.state.store.owner, websocket.cookies.get(SESSION_COOKIE))
    if not owner or not same_origin(websocket.scope, COOKIE_SECURE):
        await websocket.close(code=1008)
        return
    try:
        check_rate(websocket, "live", owner, limit=10)
    except HTTPException:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    if owner in app.state.live_owners:
        await websocket.send_json({"type": "error", "detail": "Only one microphone session per client is allowed."})
        await websocket.close(code=1008)
        return
    app.state.live_owners.add(owner)
    app.state.store.pin(owner)
    acquired = False
    recordings = ExitStack()
    deadline = time.monotonic() + MAX_AUDIO_SECONDS + LIVE_GRACE_SECONDS
    try:
        try:
            await asyncio.wait_for(app.state.live_slots.acquire(), timeout=0.1)
            acquired = True
        except asyncio.TimeoutError:
            raise AudioError("The microphone server is busy. Please try again shortly.")
        initial = await asyncio.wait_for(websocket.receive(), timeout=min(15, max(0, deadline - time.monotonic())))
        if initial["type"] == "websocket.disconnect":
            return
        if not isinstance(initial.get("text"), str):
            raise AudioError("Start with a JSON control message.")
        start = read_control(initial["text"])
        sample_rate = start.get("sampleRate")
        if start.get("type") != "start" or type(sample_rate) is not int:
            raise AudioError("Start with a sample rate and noise reduction strength.")
        if not 8000 <= sample_rate <= MAX_SAMPLE_RATE:
            raise AudioError("Unsupported microphone sample rate.")
        strength = validate_strength(start.get("strength", DEFAULT_STRENGTH))
        hum_filter = validate_hum_filter(start.get("humFilter", "off"))
        reducer = StreamingNoiseReducer(sample_rate, hum_filter)
        original_recording = recordings.enter_context(WavSpool(sample_rate))
        cleaned_recording = recordings.enter_context(WavSpool(sample_rate))
        chunk_count = 0
        total_samples = 0
        last_audio = time.monotonic()
        messages = deque()
        await send_live(websocket, "send_json", {"type": "ready", "maxDurationSeconds": MAX_AUDIO_SECONDS,
                                               "engine": "rnnoise"}, deadline)
        while True:
            now = time.monotonic()
            remaining = min(deadline - now, last_audio + LIVE_PROGRESS_SECONDS - now)
            if remaining <= 0:
                raise AudioError("Microphone connection expired or stopped sending audio.")
            message = await asyncio.wait_for(websocket.receive(), timeout=remaining)
            if message["type"] == "websocket.disconnect":
                return
            now = time.monotonic()
            while messages and messages[0] <= now - 1:
                messages.popleft()
            if len(messages) >= 30:
                raise AudioError("Microphone message rate exceeded.")
            messages.append(now)
            payload = message.get("bytes")
            if payload is not None:
                if not payload or len(payload) % 4 or len(payload) > sample_rate * 4 // 2:
                    raise AudioError("Expected at most half a second of float32 mono audio.")
                if chunk_count >= MAX_LIVE_CHUNKS:
                    raise AudioError("Too many microphone chunks.")
                samples = np.frombuffer(payload, dtype="<f4").copy()
                if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1.01:
                    raise AudioError("Invalid microphone audio samples.")
                if total_samples + samples.size > sample_rate * MAX_AUDIO_SECONDS:
                    raise AudioError(f"Recording limit: {MAX_AUDIO_SECONDS} seconds.")
                cleaned = await processing_thread(reducer.process, samples, strength)
                total_samples += samples.size
                last_audio = time.monotonic()
                chunk_count += 1
                await processing_thread(original_recording.write, samples)
                await processing_thread(cleaned_recording.write, cleaned)
                await send_live(websocket, "send_bytes", cleaned.astype("<f4").tobytes(), deadline)
            else:
                control = read_control(message.get("text", ""))
                if control.get("type") == "strength":
                    strength = validate_strength(control.get("value"))
                elif control.get("type") == "stop":
                    if not chunk_count:
                        raise AudioError("No microphone audio was received.")
                    tail = await processing_thread(reducer.finish)
                    if tail.size:
                        await processing_thread(cleaned_recording.write, tail)
                        await send_live(websocket, "send_bytes", tail.astype("<f4").tobytes(), deadline)
                    result = await processing_thread(
                        save_live, owner, original_recording, cleaned_recording,
                        sample_rate, strength, hum_filter)
                    await send_live(websocket, "send_json", {"type": "complete", **result}, deadline)
                    await asyncio.wait_for(websocket.close(code=1000), timeout=LIVE_SEND_SECONDS)
                    return
                else:
                    raise AudioError("Unknown microphone control message.")
    except WebSocketDisconnect:
        pass
    except (AudioError, StorageFull, ValueError, TypeError, asyncio.TimeoutError) as exc:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
            async with asyncio.timeout(LIVE_SEND_SECONDS):
                await websocket.send_json({"type": "error", "detail": str(exc) or "Microphone connection timed out."})
                await websocket.close(code=1008)
    except OSError:
        with suppress(WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
            async with asyncio.timeout(LIVE_SEND_SECONDS):
                await websocket.send_json({"type": "error", "detail": "Unable to save the recording."})
                await websocket.close(code=1011)
    finally:
        try:
            recordings.close()
        finally:
            app.state.store.unpin(owner)
            app.state.live_owners.discard(owner)
            if acquired:
                app.state.live_slots.release()


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

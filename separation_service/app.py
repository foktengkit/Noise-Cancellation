"""Authenticated, nonpersistent ML service; no public audio URLs or upload UI."""

import asyncio
import logging
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from backend.security import SecurityHeaders
from backend.storage import PRIVATE_HEADERS
from backend.worker_security import PYTHON_FLAGS, worker_environment
from .model import MODEL_RATE, verify_assets
from .worker import MAX_SECONDS, PROCESS_SECONDS


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = Path(os.environ.get("SEPARATION_MODEL_DIR", str(ROOT / ".tools" / "separation-model")))
MAX_BODY_BYTES = 3 * 1024 * 1024
MAX_RESPONSE_BYTES = MODEL_RATE * MAX_SECONDS * 4 + 1024


@asynccontextmanager
async def lifespan(application):
    token = os.environ.get("SEPARATION_API_KEY", "")
    if len(token) < 32 or not token.isascii() or not token.isprintable() or any(character.isspace() for character in token):
        raise RuntimeError("A private SEPARATION_API_KEY of at least 32 characters is required.")
    await run_in_threadpool(verify_assets, MODEL_DIRECTORY)
    application.state.key = token
    application.state.slot = asyncio.Semaphore(1)
    yield


app = FastAPI(title="Noise Control Separation", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.add_middleware(SecurityHeaders, secure=os.environ.get("COOKIE_SECURE", "0") == "1")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.environ.get(
    "SEPARATION_ALLOWED_HOSTS", os.environ.get("RENDER_EXTERNAL_HOSTNAME", "127.0.0.1,localhost,testserver")
).split(","))


@app.exception_handler(HTTPException)
async def http_error(request, error):
    return JSONResponse({"detail": error.detail}, status_code=error.status_code, headers=PRIVATE_HEADERS)


@app.get("/health")
def health():
    return {"status": "ok"}


def run_worker(payload):
    with tempfile.TemporaryDirectory(prefix="noise-control-separation-") as directory:
        root = Path(directory)
        (root / "input.wav").write_bytes(payload)
        environment = worker_environment(root, threads=2)
        environment["SEPARATION_DEVICE"] = os.environ.get("SEPARATION_DEVICE", "cpu")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        completed = subprocess.run([sys.executable, *PYTHON_FLAGS, "-m", "separation_service.worker", directory, str(MODEL_DIRECTORY)],
                                   cwd=ROOT, env=environment, stdin=subprocess.DEVNULL, close_fds=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=PROCESS_SECONDS, **options)
        target = root / "sources.wav"
        if completed.returncode != 0 or not target.is_file() or target.stat().st_size > MAX_RESPONSE_BYTES:
            logging.getLogger(__name__).warning("Separation worker failed: exit=%s", completed.returncode)
            raise ValueError("Separation failed.")
        return target.read_bytes()


@app.post("/v1/separate")
async def separate(request: Request):
    provided = request.headers.get("authorization", "")
    if not secrets.compare_digest(provided.encode("utf-8"), ("Bearer " + app.state.key).encode("ascii")):
        raise HTTPException(401, "Service authentication required.")
    if request.headers.get("content-type", "").split(";", 1)[0] != "audio/wav":
        raise HTTPException(415, "Expected PCM WAV audio.")
    length = request.headers.get("content-length")
    if length is not None and (not length.isascii() or not length.isdigit() or len(length) > 9):
        raise HTTPException(400, "Invalid Content-Length.")
    if length is not None and int(length) > MAX_BODY_BYTES:
        raise HTTPException(413, "Separation input is too large.")
    try:
        await asyncio.wait_for(app.state.slot.acquire(), 0.1)
    except asyncio.TimeoutError:
        raise HTTPException(503, "The separation service is busy.")
    try:
        payload = bytearray()
        async with asyncio.timeout(30):
            async for chunk in request.stream():
                if len(payload) + len(chunk) > MAX_BODY_BYTES:
                    raise HTTPException(413, "Separation input is too large.")
                payload.extend(chunk)
        if not payload:
            raise HTTPException(400, "The separation clip is empty.")
        result = await run_in_threadpool(run_worker, bytes(payload))
        return Response(result, media_type="audio/wav", headers=PRIVATE_HEADERS)
    except (subprocess.TimeoutExpired, TimeoutError):
        raise HTTPException(408, "Separation timed out.")
    except (ValueError, OSError):
        raise HTTPException(503, "Unable to separate this recording.")
    finally:
        app.state.slot.release()

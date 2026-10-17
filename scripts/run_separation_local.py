"""Run both local services with an ephemeral shared credential and no public bind."""

import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

from separation_service.model import verify_assets


ROOT = Path(__file__).resolve().parent.parent


def available_port(start):
    for port in range(start, start + 20):
        with socket.socket() as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
                return port
            except OSError:
                pass
    raise RuntimeError("No local port is available.")


def main():
    ml_python = ROOT / (".venv-separation/Scripts/python.exe" if os.name == "nt" else ".venv-separation/bin/python")
    if not ml_python.is_file():
        raise RuntimeError("The optional .venv-separation environment is missing.")
    verify_assets(ROOT / ".tools" / "separation-model")
    web_port, ml_port = available_port(8002), available_port(8101)
    env = {**os.environ, "SEPARATION_API_KEY": secrets.token_urlsafe(32), "SEPARATION_ALLOW_HTTP": "1",
           "SEPARATION_URL": "http://127.0.0.1:" + str(ml_port), "COOKIE_SECURE": "0",
           "SEPARATION_ALLOWED_HOSTS": "127.0.0.1,localhost", "ALLOWED_HOSTS": "127.0.0.1,localhost",
           "SEPARATION_MODEL_DIR": str(ROOT / ".tools" / "separation-model"),
           "MEDIA_DIR": str(ROOT / ".tools" / "local-separation-media")}
    children, logs = [], []
    try:
        for python, application, port, name in ((ml_python, "separation_service.app:app", ml_port, "separation"),
                                                (sys.executable, "backend.app:app", web_port, "web-separation")):
            log = (ROOT / ("server." + name + ".log")).open("ab")
            logs.append(log)
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            children.append(subprocess.Popen([str(python), "-m", "uvicorn", application, "--host", "127.0.0.1",
                                               "--port", str(port), "--workers", "1", "--no-proxy-headers",
                                               "--ws-max-size", "131072", "--ws-max-queue", "4", "--no-access-log"],
                                              cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options))
        print("Noise Control: http://127.0.0.1:" + str(web_port), flush=True)
        print("Keep this window open. Press Ctrl+C to stop both services.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(.5)
        raise RuntimeError("A service stopped. See server.separation.log and server.web-separation.log.")
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    child.terminate()
                child.wait(timeout=15)
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()

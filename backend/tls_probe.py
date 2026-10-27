"""Opt-in, fixed-target TLS checks; results are restricted to operator logs."""

import http.client
import json
import logging
import os
from pathlib import Path
import re
import ssl
import subprocess
import sys
import tempfile

from .worker_security import PYTHON_FLAGS, worker_environment


SOCKET_SECONDS = 5
PROCESS_SECONDS = 25
MAX_REPORT_BYTES = 4096
SCOPE = "python.http.client.system_ca"
TARGETS = (
    ("control", "render.com", True),
    ("emudhra", "dv-g3b-emudhra.we-have-a-problem.com", False),
    ("emudhra_aaa", "dv-g3b-aaa-emudhra.we-have-a-problem.com", False),
)
# OpenSSL issuer/trust errors, not expiry, hostname errors, or generic TLS failures.
TRUST_ERRORS = frozenset({2, 18, 19, 20, 21, 23, 27, 28})
logger = logging.getLogger("uvicorn.error.tls")


def check_target(target):
    name, hostname, _ = target
    result = {"target": name, "outcome": "inconclusive"}
    connection = None
    accepted = False
    try:
        # Match the production separation client's default HTTPSConnection context.
        connection = http.client.HTTPSConnection(hostname, 443, timeout=SOCKET_SECONDS)
        context = connection._context
        if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
            return result
        connection.connect()
        accepted = True
        result["outcome"] = "accepted"
        connection.request("HEAD", "/", headers={"User-Agent": "Noise-Control-TLS-Check", "Connection": "close"})
        response = connection.getresponse()
        result["httpStatus"] = response.status
        response.close()
    except ssl.SSLCertVerificationError as error:
        code = getattr(error, "verify_code", None)
        if not accepted and code in TRUST_ERRORS:
            result.update(outcome="certificate_rejected", verifyCode=code)
    except (OSError, http.client.HTTPException):
        # A successful handshake is unsafe for a negative target even if HTTP fails.
        pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
    return result


def successful(checks):
    return len(checks) == len(TARGETS) and all(row["outcome"] == ("accepted" if target[2] else "certificate_rejected")
               for target, row in zip(TARGETS, checks))


def collect_checks():
    checks = [check_target(target) for target in TARGETS]
    return {"schema": 1, "scope": SCOPE, "passed": successful(checks), "checks": checks}


def probe_environment(directory):
    environment = worker_environment(directory)
    # Preserve trust configuration, but never credentials, proxies, or TLS key logging.
    for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "OPENSSL_CONF", "OPENSSL_MODULES"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def parse_report(payload):
    if not isinstance(payload, (bytes, str)) or len(payload) > MAX_REPORT_BYTES:
        raise ValueError("Oversized TLS check result")
    report = json.loads(payload)
    if (not isinstance(report, dict) or type(report.get("schema")) is not int or report.get("schema") != 1 or report.get("scope") != SCOPE
            or not isinstance(report.get("checks"), list) or len(report["checks"]) != len(TARGETS)):
        raise ValueError("Incomplete TLS check result")
    checks = report["checks"]
    for target, row in zip(TARGETS, checks):
        if (not isinstance(row, dict) or row.get("target") != target[0]
                or not isinstance(row.get("outcome"), str)
                or row.get("outcome") not in {"accepted", "certificate_rejected", "inconclusive"}
                or set(row) - {"target", "outcome", "verifyCode", "httpStatus"}):
            raise ValueError("Invalid TLS check result")
        if row["outcome"] == "certificate_rejected":
            if type(row.get("verifyCode")) is not int or row["verifyCode"] not in TRUST_ERRORS:
                raise ValueError("TLS rejection does not prove an issuer/trust failure")
        elif "verifyCode" in row:
            raise ValueError("Unexpected TLS verification details")
        if "httpStatus" in row and (type(row["httpStatus"]) is not int or not 100 <= row["httpStatus"] <= 599):
            raise ValueError("Invalid HTTP status")
        if "httpStatus" in row and row["outcome"] != "accepted":
            raise ValueError("HTTP response without an accepted TLS handshake")
    passed = successful(checks)
    if type(report.get("passed")) is not bool or report["passed"] != passed:
        raise ValueError("Inconsistent TLS check summary")
    return {"schema": 1, "scope": SCOPE, "passed": passed, "checks": checks}


def run_startup_tls_check():
    if os.environ.get("TLS_STARTUP_CHECK") != "1":
        return None
    report = {"schema": 1, "scope": SCOPE, "passed": False, "checks": [], "outcome": "inconclusive"}
    try:
        with tempfile.TemporaryDirectory(prefix="noise-tls-") as directory:
            result = subprocess.run(
                [sys.executable, *PYTHON_FLAGS, "-m", "backend.tls_probe", "--worker"],
                cwd=Path(__file__).resolve().parent.parent, env=probe_environment(directory),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                close_fds=True, timeout=PROCESS_SECONDS,
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        if result.returncode != 0:
            raise ValueError("TLS check process did not finish normally")
        report = parse_report(result.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    commit = os.environ.get("RENDER_GIT_COMMIT", "")
    instance = os.environ.get("RENDER_INSTANCE_ID", "")
    report.update(runtime="render" if os.environ.get("RENDER") == "true" else "local",
                  commit=commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "",
                  instance=instance if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", instance) else "")
    logger.log(logging.INFO if report["passed"] else logging.WARNING,
               "tls_startup_check %s", json.dumps(report, separators=(",", ":"), allow_nan=False))
    return report


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This module only supports its fixed-target worker mode.")
    print(json.dumps(collect_checks(), separators=(",", ":"), allow_nan=False))

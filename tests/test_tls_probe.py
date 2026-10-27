import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import backend.app as app
from backend import separation, tls_probe as probe


def rejected(code=20):
    error = ssl.SSLCertVerificationError(1, "private-sentinel")
    error.verify_code = code
    return error


def connection(error=None, status=200):
    result = Mock()
    result._context = ssl.create_default_context()
    result.connect.side_effect = error
    result.getresponse.return_value.status = status
    return result


def healthy_report():
    return {"schema": 1, "scope": probe.SCOPE, "passed": True, "checks": [
        {"target": "control", "outcome": "accepted", "httpStatus": 200},
        {"target": "emudhra", "outcome": "certificate_rejected", "verifyCode": 20},
        {"target": "emudhra_aaa", "outcome": "certificate_rejected", "verifyCode": 21}]}


class TlsProbeTests(unittest.TestCase):
    def test_uses_production_client_and_strict_default_context(self):
        self.assertIs(probe.http.client.HTTPSConnection, separation.http.client.HTTPSConnection)
        client = probe.http.client.HTTPSConnection("render.com", timeout=5)
        try:
            self.assertTrue(client._context.check_hostname)
            self.assertEqual(client._context.verify_mode, ssl.CERT_REQUIRED)
        finally:
            client.close()

    def test_control_and_both_trust_rejections_are_required(self):
        clients = [connection(), connection(rejected(20)), connection(rejected(21))]
        with patch.object(probe.http.client, "HTTPSConnection", side_effect=clients) as factory:
            result = probe.collect_checks()
        self.assertTrue(result["passed"])
        self.assertEqual([call.args[0] for call in factory.call_args_list], [target[1] for target in probe.TARGETS])
        for client in clients:
            client.close.assert_called_once()
        clients[0].request.assert_called_once_with("HEAD", "/", headers={
            "User-Agent": "Noise-Control-TLS-Check", "Connection": "close"})
        clients[1].request.assert_not_called()
        clients[2].request.assert_not_called()

    def test_http_error_or_redirect_still_means_bad_certificate_was_trusted(self):
        for status in (200, 301, 404, 500):
            clients = [connection(), connection(status=status), connection(rejected())]
            with self.subTest(status=status), patch.object(probe.http.client, "HTTPSConnection", side_effect=clients):
                result = probe.collect_checks()
                self.assertFalse(result["passed"])
                self.assertEqual(result["checks"][1]["outcome"], "accepted")
                self.assertEqual(clients[1].request.call_count, 1)

    def test_accepted_handshake_followed_by_http_failure_is_not_a_rejection(self):
        client = connection()
        client.request.side_effect = ConnectionResetError("private-sentinel")
        with patch.object(probe.http.client, "HTTPSConnection", return_value=client):
            result = probe.check_target(probe.TARGETS[1])
        self.assertEqual(result["outcome"], "accepted")

    def test_dns_timeout_and_generic_tls_failures_are_inconclusive(self):
        for error in (socket.gaierror("private-sentinel"), TimeoutError("private-sentinel"),
                      ConnectionRefusedError("private-sentinel"), ssl.SSLError("private-sentinel")):
            clients = [connection(), connection(error), connection(rejected())]
            with self.subTest(error=type(error).__name__), patch.object(probe.http.client, "HTTPSConnection", side_effect=clients):
                result = probe.collect_checks()
                self.assertFalse(result["passed"])
                self.assertEqual(result["checks"][1]["outcome"], "inconclusive")
                self.assertNotIn("private-sentinel", json.dumps(result))

    def test_expiry_hostname_and_unknown_errors_do_not_prove_ca_distrust(self):
        for code in (9, 10, 62, 64, 999):
            with self.subTest(code=code), patch.object(probe.http.client, "HTTPSConnection", return_value=connection(rejected(code))):
                self.assertEqual(probe.check_target(probe.TARGETS[1])["outcome"], "inconclusive")

    def test_blocked_positive_control_cannot_pass(self):
        with patch.object(probe.http.client, "HTTPSConnection", side_effect=[connection(rejected()) for _ in range(3)]):
            self.assertFalse(probe.collect_checks()["passed"])

    def test_insecure_context_does_not_connect(self):
        client = connection()
        client._context.check_hostname = False
        with patch.object(probe.http.client, "HTTPSConnection", return_value=client):
            self.assertEqual(probe.check_target(probe.TARGETS[1])["outcome"], "inconclusive")
        client.connect.assert_not_called()
        client.close.assert_called_once()

    def test_child_environment_preserves_trust_settings_but_not_secrets_or_key_logs(self):
        values = {"SEPARATION_API_KEY": "private-sentinel", "RENDER_API_KEY": "private-sentinel",
                  "HTTPS_PROXY": "private-sentinel", "SSLKEYLOGFILE": "private-sentinel",
                  "SSL_CERT_FILE": "custom-ca.pem", "SSL_CERT_DIR": "custom-ca",
                  "OPENSSL_CONF": "custom-openssl.cnf", "OPENSSL_MODULES": "custom-modules"}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, values):
            environment = probe.probe_environment(directory)
        for name in ("SEPARATION_API_KEY", "RENDER_API_KEY", "HTTPS_PROXY", "SSLKEYLOGFILE"):
            self.assertNotIn(name, environment)
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "OPENSSL_CONF", "OPENSSL_MODULES"):
            self.assertEqual(environment[name], values[name])

    def test_disabled_by_default_without_process_or_log(self):
        for value in ("", "0", "true"):
            with self.subTest(value=value), patch.dict(os.environ, {"TLS_STARTUP_CHECK": value}), \
                    patch.object(probe.subprocess, "run") as run:
                self.assertIsNone(probe.run_startup_tls_check())
                run.assert_not_called()

    def test_subprocess_is_bounded_and_log_contains_no_raw_environment(self):
        result = Mock(returncode=0, stdout=json.dumps(healthy_report()).encode())
        values = {"TLS_STARTUP_CHECK": "1", "RENDER": "true", "RENDER_GIT_COMMIT": "a" * 40,
                  "RENDER_INSTANCE_ID": "srv-instance", "SEPARATION_API_KEY": "private-sentinel"}
        with patch.dict(os.environ, values), patch.object(probe.subprocess, "run", return_value=result) as run, \
                self.assertLogs("uvicorn.error.tls", level="INFO") as logs:
            report = probe.run_startup_tls_check()
        self.assertTrue(report["passed"])
        self.assertEqual(report["runtime"], "render")
        self.assertEqual(report["commit"], "a" * 40)
        self.assertEqual(run.call_args.kwargs["timeout"], probe.PROCESS_SECONDS)
        self.assertTrue(run.call_args.kwargs["close_fds"])
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("private-sentinel", json.dumps(report) + "".join(logs.output))

    def test_child_timeout_failure_and_invalid_results_never_pass(self):
        values = [subprocess.TimeoutExpired("private-sentinel", 25), OSError("private-sentinel"),
                  Mock(returncode=1, stdout=b""), Mock(returncode=0, stdout=b"private-sentinel"),
                  Mock(returncode=0, stdout=b"x" * (probe.MAX_REPORT_BYTES + 1))]
        for value in values:
            kwargs = {"side_effect": value} if isinstance(value, Exception) else {"return_value": value}
            with self.subTest(value=type(value).__name__), patch.dict(os.environ, {"TLS_STARTUP_CHECK": "1"}), \
                    patch.object(probe.subprocess, "run", **kwargs), self.assertLogs("uvicorn.error.tls", level="WARNING") as logs:
                result = probe.run_startup_tls_check()
            self.assertFalse(result["passed"])
            self.assertNotIn("private-sentinel", json.dumps(result) + "".join(logs.output))

    def test_incomplete_mismatched_and_forged_child_summaries_are_rejected(self):
        invalid = [{}, {**healthy_report(), "checks": []}, {**healthy_report(), "passed": False}]
        wrong_target = healthy_report()
        wrong_target["checks"][1]["target"] = "arbitrary-host"
        invalid.append(wrong_target)
        wrong_rejection = healthy_report()
        wrong_rejection["checks"][1]["verifyCode"] = 10
        invalid.append(wrong_rejection)
        extra_data = healthy_report()
        extra_data["checks"][1]["secret"] = "private-sentinel"
        invalid.append(extra_data)
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                probe.parse_report(json.dumps(data).encode())

    def test_malformed_or_unexpected_verification_metadata_is_rejected(self):
        for outcome, code in (([], 20), ("accepted", "private-sentinel"),
                              ("certificate_rejected", []), ("certificate_rejected", True)):
            result = healthy_report()
            result["checks"][1].update(outcome=outcome, verifyCode=code)
            with self.subTest(outcome=outcome, code=code), self.assertRaises(ValueError):
                probe.parse_report(json.dumps(result).encode())
        self.assertFalse(probe.successful([]))

    def test_startup_is_opt_in_and_has_no_public_diagnostic_route(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "MEDIA_DIR", Path(directory)), \
                patch.dict(os.environ, {"TLS_STARTUP_CHECK": "0"}), patch.object(app, "run_startup_tls_check") as run:
            with TestClient(app.app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertEqual(client.get("/api/tls-check").status_code, 404)
            run.assert_not_called()

    def test_enabled_startup_runs_once_off_the_request_thread(self):
        called = threading.Event()
        def check():
            called.set()
            return healthy_report()
        with tempfile.TemporaryDirectory() as directory, patch.object(app, "MEDIA_DIR", Path(directory)), \
                patch.dict(os.environ, {"TLS_STARTUP_CHECK": "1"}), patch.object(app, "run_startup_tls_check", side_effect=check) as run:
            with TestClient(app.app) as client:
                self.assertEqual(client.get("/api/health").status_code, 200)
                self.assertTrue(called.wait(2))
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()

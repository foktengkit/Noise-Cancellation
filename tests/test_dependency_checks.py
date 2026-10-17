from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from scripts import check_dependencies as scanner


ROOT = Path(__file__).resolve().parents[1]


class DependencyCheckTests(unittest.TestCase):
    def test_inventory_covers_both_locks_including_platform_specific_versions(self):
        packages = scanner.inventory()
        self.assertGreater(len(packages), 40)
        self.assertTrue(any(row["name"] == "torch" for row in packages))
        self.assertTrue(any(row["name"] == "hf-xet" for row in packages))
        self.assertFalse(any("+" in row["version"] for row in packages))
        self.assertEqual(len(packages), len({(row["name"],row["version"]) for row in packages}))

    def test_unpinned_or_malformed_inventory_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in scanner.LOCKFILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("package>=1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                scanner.inventory(root)

    def query(self, response):
        packages = [{"name":"example", "version":"1.0"}]
        with patch.object(scanner.urllib.request, "urlopen", return_value=io.BytesIO(json.dumps(response).encode())) as call:
            result = scanner.check(packages)
        sent = json.loads(call.call_args.args[0].data)
        self.assertEqual(sent, {"queries":[{"package":{"name":"example","ecosystem":"PyPI"},"version":"1.0"}]})
        return result

    def test_clean_response_and_vulnerable_response_are_distinguished(self):
        self.assertEqual(self.query({"results":[{}]})["findings"], [])
        self.assertEqual(self.query({"results":[{"vulns":[{"id":"TEST-ADVISORY"}]}]})["findings"][0]["advisories"], ["TEST-ADVISORY"])

    def test_incomplete_paginated_and_invalid_responses_fail_closed(self):
        for response in ({}, {"results":[]}, {"results":[None]}, {"results":[{"next_page_token":"more"}]},
                         {"results":[{"error":"upstream failed"}]},
                         {"results":[{"vulns":[{}]}]}, {"results":[{"vulns":None}]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.query(response)

    def test_known_vulnerability_returns_a_failure_exit(self):
        with patch.object(scanner,"inventory",return_value=[]), \
                patch.object(scanner,"check",return_value={"findings":[{"advisories":["TEST"]}]}), redirect_stdout(io.StringIO()):
            self.assertEqual(scanner.main(), 1)

    def test_security_workflow_and_update_configuration_are_present(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8"))
        events = workflow.get("on", workflow.get(True))
        self.assertIn("schedule", events)
        self.assertIn("pull_request", events)
        self.assertEqual(workflow["permissions"], {"contents":"read"})
        self.assertIn("python -m scripts.check_dependencies", [s.get("run") for s in workflow["jobs"]["dependencies"]["steps"]])
        updates = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))
        self.assertEqual({(row["package-ecosystem"],row["directory"]) for row in updates["updates"]},
                         {("pip","/"),("pip","/separation_service"),("github-actions","/")})


if __name__ == "__main__":
    unittest.main()

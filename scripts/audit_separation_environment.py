"""Query OSV with package versions only; no audio, credentials or file paths."""

import importlib.metadata
import json
from pathlib import Path
import urllib.request


def main():
    packages = sorted(({"name": distribution.metadata["Name"], "version": distribution.version}
                       for distribution in importlib.metadata.distributions()), key=lambda row: row["name"].lower())
    queries = [{"package": {"name": row["name"], "ecosystem": "PyPI"},
                "version": row["version"].split("+", 1)[0]} for row in packages]
    request = urllib.request.Request("https://api.osv.dev/v1/querybatch", json.dumps({"queries": queries}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        results = json.load(response)["results"]
    if len(results) != len(packages):
        raise ValueError("OSV returned an incomplete inventory response.")
    findings = [{**package, "advisories": result["vulns"]} for package, result in zip(packages, results) if result.get("vulns")]
    report = {"source": "https://api.osv.dev/v1/querybatch", "packagesChecked": len(packages), "findings": findings,
              "limitations": "Installed Python packages only; excludes OS, native libraries and uninstalled platform dependencies."}
    output = Path("test-artifacts/separation/dependency-audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

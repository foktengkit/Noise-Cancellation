"""Check every pinned Python version against OSV without installing the locks."""

import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

from packaging.utils import canonicalize_name
from pip_requirements_parser import RequirementsFile


ROOT = Path(__file__).resolve().parents[1]
LOCKFILES = ("requirements.lock", "separation_service/requirements.lock")
ENDPOINT = "https://api.osv.dev/v1/querybatch"


def inventory(root=ROOT):
    packages = set()
    for filename in LOCKFILES:
        parsed = RequirementsFile.from_file(str(root / filename))
        if parsed.invalid_lines or not parsed.requirements:
            raise ValueError("A dependency lock could not be parsed completely.")
        for entry in parsed.requirements:
            requirement = entry.req
            if requirement is None or requirement.url:
                raise ValueError("Only exact package versions can be checked.")
            specs = list(requirement.specifier)
            if len(specs) != 1 or specs[0].operator != "==" or "*" in specs[0].version:
                raise ValueError("Every dependency must use an exact version.")
            # CPU wheel local tags are not part of the upstream advisory version.
            packages.add((canonicalize_name(requirement.name), specs[0].version.split("+", 1)[0]))
    return [{"name": name, "version": version} for name, version in sorted(packages)]


def check(packages):
    queries = [{"package": {"name": row["name"], "ecosystem": "PyPI"}, "version": row["version"]}
               for row in packages]
    request = urllib.request.Request(ENDPOINT, json.dumps({"queries": queries}).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(2 * 1024 * 1024 + 1)
    if len(payload) > 2 * 1024 * 1024:
        raise ValueError("The advisory response exceeded its size limit.")
    data = json.loads(payload)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list) or len(data["results"]) != len(packages):
        raise ValueError("The advisory response was incomplete.")
    findings = []
    for package, result in zip(packages, data["results"]):
        if not isinstance(result, dict) or "error" in result or result.get("next_page_token"):
            raise ValueError("The advisory response was incomplete or paginated.")
        vulnerabilities = result.get("vulns", [])
        if not isinstance(vulnerabilities, list) or any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
                                                      or not row["id"] for row in vulnerabilities):
            raise ValueError("The advisory response contained invalid entries.")
        if vulnerabilities:
            findings.append({**package, "advisories": [row["id"] for row in vulnerabilities]})
    return {"packagesChecked": len(packages), "findings": findings}


def main():
    try:
        result = check(inventory())
    except (ValueError, OSError, urllib.error.URLError):
        print("Dependency check failed; no clean result is available.", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 1 if result["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

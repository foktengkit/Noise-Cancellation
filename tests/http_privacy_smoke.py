"""Verify a running service with synthetic audio and independent HTTP sessions."""

import argparse
import json
from urllib.parse import urlsplit

import httpx
import numpy as np

from backend.denoise import write_wav_bytes


def run(url):
    url = url.rstrip("/")
    parsed = urlsplit(url)
    origin = parsed.scheme + "://" + parsed.netloc
    headers = {"origin": origin}
    with httpx.Client(base_url=url, headers=headers, timeout=60) as owner, \
            httpx.Client(base_url=url, headers=headers, timeout=15) as other, \
            httpx.Client(base_url=url, timeout=15) as anonymous:
        for client in [owner, other]:
            response = client.get("/api/session")
            response.raise_for_status()
        payload = write_wav_bytes(np.zeros((160, 1), dtype=np.float32), 8000)
        response = owner.post("/api/denoise", files={"file": ("privacy-smoke.wav", payload)},
                              data={"strength": "0"})
        response.raise_for_status()
        result = response.json()
        endpoint = "/api/recordings/" + result["recordingId"]
        try:
            for variant in ["originalUrl", "denoisedUrl"]:
                path = result[variant]
                own = owner.get(path)
                assert own.status_code == 200 and own.content.startswith(b"RIFF")
                assert "no-store" in own.headers.get("cache-control", "")
                assert owner.get(path, headers={"range": "bytes=0-15"}).status_code == 206
                for client in [other, anonymous]:
                    for method, extra in [("GET", {}), ("HEAD", {}),
                                          ("GET", {"range": "bytes=0-15"}),
                                          ("GET", {"if-none-match": own.headers["etag"]})]:
                        response = client.request(method, path, headers=extra)
                        assert response.status_code == 404, (method, response.status_code)
            assert other.delete(endpoint).status_code == 404
            assert owner.get(result["originalUrl"]).status_code == 200
            assert owner.delete(endpoint).status_code == 204
            for variant in ["originalUrl", "denoisedUrl"]:
                assert owner.get(result[variant]).status_code == 404
        finally:
            owner.delete(endpoint)
    print(json.dumps({"ownerDownload": 200, "ownerRange": 206, "foreignGetHeadRangeConditional": 404,
                      "foreignDelete": 404, "ownerDelete": 204, "afterDelete": 404}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    run(parser.parse_args().url)

from __future__ import annotations

import time
import re
from collections import deque
from urllib.parse import urlsplit

from starlette.datastructures import Headers


class RateLimits:
    def __init__(self):
        self.buckets = {}
        self.next_sweep = 0

    def allow(self, key: str, limit: int, seconds: float = 60) -> bool:
        now = time.monotonic()
        if now >= self.next_sweep:
            for old_key, bucket in list(self.buckets.items()):
                if not bucket or bucket[-1] <= now - 60:
                    del self.buckets[old_key]
            self.next_sweep = now + 60
        if key not in self.buckets:
            if len(self.buckets) >= 4096:
                return False
            self.buckets[key] = deque()
        bucket = self.buckets[key]
        while bucket and bucket[0] <= now - seconds:
            bucket.popleft()
        if len(bucket) >= limit:
            return False
        bucket.append(now)
        return True


def same_origin(scope, secure: bool, required: bool = True) -> bool:
    headers = Headers(scope=scope)
    origin = headers.get("origin")
    if not origin:
        return not required and headers.get("sec-fetch-site") not in {"cross-site", "same-site"}
    try:
        parsed = urlsplit(origin)
        scheme = "https" if secure or scope["scheme"] in {"https", "wss"} else "http"
        return (parsed.scheme == scheme and parsed.netloc.lower() == headers.get("host", "").lower()
                and not parsed.path and not parsed.query and not parsed.fragment
                and parsed.username is None and parsed.password is None)
    except ValueError:
        return False


class SecurityHeaders:
    def __init__(self, app, secure=False):
        self.app = app
        self.secure = secure

    async def __call__(self, scope, receive, send):
        async def secured_send(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                host = Headers(scope=scope).get("host", "")
                sockets = f"ws://{host} wss://{host}" if re.fullmatch(r"[A-Za-z0-9.\[\]:-]+", host) else ""
                policy = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                          "img-src 'self' data:; font-src 'self'; "
                          f"connect-src 'self' {sockets}; media-src 'self' blob:; "
                          "worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
                          "frame-ancestors 'none'; form-action 'self'")
                headers.extend([
                    (b"content-security-policy", policy.encode("ascii")),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                    (b"permissions-policy", b"microphone=(self), camera=(), geolocation=()"),
                ])
                if self.secure or scope.get("scheme") == "https":
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)
        await self.app(scope, receive, secured_send)

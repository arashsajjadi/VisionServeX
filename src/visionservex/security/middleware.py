"""FastAPI middleware components."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from visionservex.config import LimitsConfig
from visionservex.runtime.monitor import metrics
from visionservex.utils.ids import request_id
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Response]]):
        rid = request.headers.get("X-Request-Id") or request_id()
        request.state.request_id = rid
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:  # pragma: no cover - defensive
            metrics.error("unhandled")
            _log.exception("unhandled error during request %s", rid)
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": rid,
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "internal server error",
                        "hint": "see server logs",
                        "details": {},
                    },
                },
            )
        response.headers["X-Request-Id"] = rid
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.observe("http_latency_ms", elapsed_ms)
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose declared body exceeds the configured cap.

    This is a defense-in-depth complement to upload validation; ASGI servers
    may also need their own limits configured.
    """

    def __init__(self, app, limits: LimitsConfig) -> None:
        super().__init__(app)
        self.limits = limits

    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Response]]):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                size = int(cl)
            except ValueError:
                return _err(
                    request,
                    400,
                    "BAD_CONTENT_LENGTH",
                    "invalid content-length",
                    "send a valid integer",
                )
            if size > self.limits.max_upload_bytes:
                return _err(
                    request,
                    413,
                    "REQUEST_TOO_LARGE",
                    f"request body {size} bytes exceeds max_upload_bytes {self.limits.max_upload_bytes}",
                    "send a smaller image or raise the limit in config",
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token bucket per remote address."""

    def __init__(self, app, limits: LimitsConfig) -> None:
        super().__init__(app)
        self.limits = limits
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable[..., Awaitable[Response]]):
        if self.limits.rate_limit_per_minute <= 0:
            return await call_next(request)
        key = request.client.host if request.client else "anonymous"
        now = time.monotonic()
        window = 60.0
        bucket = self._buckets[key]
        while bucket and now - bucket[0] > window:
            bucket.popleft()
        if len(bucket) >= self.limits.rate_limit_per_minute:
            metrics.increment("requests_rate_limited")
            return _err(
                request,
                429,
                "RATE_LIMITED",
                "request rate exceeded",
                "wait a minute or raise rate_limit_per_minute",
            )
        bucket.append(now)
        return await call_next(request)


def _err(request: Request, status: int, code: str, message: str, hint: str) -> JSONResponse:
    rid = getattr(request.state, "request_id", request_id())
    return JSONResponse(
        status_code=status,
        content={
            "request_id": rid,
            "error": {
                "code": code,
                "message": message,
                "hint": hint,
                "details": {},
            },
        },
    )


__all__ = [
    "BodySizeLimitMiddleware",
    "RateLimitMiddleware",
    "RequestContextMiddleware",
]

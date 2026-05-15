"""Async request scheduler with per-model concurrency, queueing, and backpressure.

Goals:
* Two or more simultaneous requests are handled safely.
* Avoid loading the same model twice unnecessarily (delegated to ModelCache).
* Bound per-model concurrency so GPUs don't OOM under load.
* Apply backpressure (HTTP 503) instead of unbounded queueing.
* Honor a request timeout.
* Cancel work cleanly on shutdown.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from visionservex.config import get_settings
from visionservex.runtime.monitor import metrics
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class BackpressureError(RuntimeError):
    """Raised when the scheduler refuses to accept more work."""


class RequestTimeoutError(asyncio.TimeoutError):
    pass


@dataclass
class _ModelSlot:
    semaphore: asyncio.Semaphore
    inflight: int = 0
    queued: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RequestScheduler:
    """Coordinates concurrent inference work across models."""

    def __init__(
        self,
        *,
        per_model_concurrency: int | None = None,
        queue_size: int | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        settings = get_settings()
        self.per_model_concurrency = per_model_concurrency or settings.runtime.per_model_concurrency
        self.queue_size = queue_size or settings.runtime.queue_size
        self.request_timeout_s = request_timeout_s or settings.runtime.request_timeout_s
        self._slots: dict[str, _ModelSlot] = {}
        self._lock = threading.Lock()
        self._total_inflight = 0
        self._global_lock = asyncio.Lock()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_inflight": self._total_inflight,
                "queue_size": self.queue_size,
                "per_model_concurrency": self.per_model_concurrency,
                "request_timeout_s": self.request_timeout_s,
                "models": {
                    mid: {"inflight": slot.inflight, "queued": slot.queued}
                    for mid, slot in self._slots.items()
                },
            }

    def _get_slot(self, model_id: str) -> _ModelSlot:
        with self._lock:
            slot = self._slots.get(model_id)
            if slot is None:
                slot = _ModelSlot(semaphore=asyncio.Semaphore(self.per_model_concurrency))
                self._slots[model_id] = slot
            return slot

    @asynccontextmanager
    async def reserve(self, model_id: str):
        """Acquire a per-model slot or raise :class:`BackpressureError`."""
        slot = self._get_slot(model_id)
        with self._lock:
            if slot.queued + slot.inflight >= self.queue_size:
                metrics.increment("requests_rejected_backpressure")
                raise BackpressureError(
                    f"model {model_id!r} queue is full ({slot.queued}+{slot.inflight} ≥ {self.queue_size})"
                )
            slot.queued += 1
            self._total_inflight += 1
        try:
            await asyncio.wait_for(slot.semaphore.acquire(), timeout=self.request_timeout_s)
        except asyncio.TimeoutError as exc:
            with self._lock:
                slot.queued -= 1
                self._total_inflight -= 1
            metrics.increment("requests_timed_out")
            raise RequestTimeoutError("timed out waiting for model slot") from exc

        with self._lock:
            slot.queued -= 1
            slot.inflight += 1

        try:
            yield
        finally:
            slot.semaphore.release()
            with self._lock:
                slot.inflight -= 1
                self._total_inflight -= 1

    async def run(
        self,
        model_id: str,
        fn: Callable[[], Awaitable[Any] | Any],
    ) -> Any:
        """Run ``fn`` under a per-model slot, off the event loop."""
        async with self.reserve(model_id):
            loop = asyncio.get_running_loop()
            start = time.perf_counter()
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await asyncio.wait_for(fn(), timeout=self.request_timeout_s)
                else:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, fn),
                        timeout=self.request_timeout_s,
                    )
            except asyncio.TimeoutError as exc:
                metrics.increment("requests_timed_out")
                raise RequestTimeoutError("inference timed out") from exc
            finally:
                latency_ms = (time.perf_counter() - start) * 1000.0
                metrics.observe("latency_ms", latency_ms)
                metrics.increment("requests_total")
            return result


_default: RequestScheduler | None = None
_lock = threading.Lock()


def get_scheduler() -> RequestScheduler:
    global _default
    if _default is None:
        with _lock:
            if _default is None:
                _default = RequestScheduler()
    return _default


__all__ = [
    "BackpressureError",
    "RequestScheduler",
    "RequestTimeoutError",
    "get_scheduler",
]

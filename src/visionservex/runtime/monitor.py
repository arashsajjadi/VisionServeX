"""Lightweight in-process metrics."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any


class MetricsRegistry:
    def __init__(self, *, history_size: int = 1024) -> None:
        self._lock = threading.RLock()
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._observations: dict[str, deque[float]] = {}
        self._history_size = history_size
        self._started_at = time.time()
        self._errors: dict[str, int] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            dq = self._observations.setdefault(name, deque(maxlen=self._history_size))
            dq.append(float(value))

    def error(self, code: str) -> None:
        with self._lock:
            self._errors[code] = self._errors.get(code, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            obs = {}
            for k, dq in self._observations.items():
                values = sorted(dq)
                if not values:
                    continue
                obs[k] = {
                    "n": len(values),
                    "min": values[0],
                    "p50": values[len(values) // 2],
                    "p90": values[min(len(values) - 1, int(len(values) * 0.9))],
                    "p99": values[min(len(values) - 1, int(len(values) * 0.99))],
                    "max": values[-1],
                }
            return {
                "uptime_s": time.time() - self._started_at,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "observations": obs,
                "errors": dict(self._errors),
            }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._observations.clear()
            self._errors.clear()
            self._started_at = time.time()


metrics = MetricsRegistry()


__all__ = ["MetricsRegistry", "metrics"]

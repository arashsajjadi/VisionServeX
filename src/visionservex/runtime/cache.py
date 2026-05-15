"""Model cache with LRU eviction and idle unload.

The cache holds at most ``max_loaded_models`` :class:`VisionModel` instances
in memory. Asking for a model that is not loaded loads it; asking for one
that is loaded returns the cached instance. When the cache is full the
least-recently-used model is unloaded.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional

from visionservex.config import get_settings
from visionservex.utils.logging import get_logger

if False:  # type-checking only; avoid runtime circular import
    from visionservex.core.model import VisionModel  # noqa: F401

_log = get_logger(__name__)


class ModelCache:
    def __init__(self, max_loaded: Optional[int] = None) -> None:
        settings = get_settings()
        self.max_loaded = max_loaded or settings.runtime.max_loaded_models
        self.idle_unload_s = settings.runtime.model_idle_unload_s
        self._lock = threading.RLock()
        self._items: OrderedDict[str, _CacheEntry] = OrderedDict()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items.keys())

    def info(self) -> list[dict]:
        with self._lock:
            return [e.info() for e in self._items.values()]

    def get(self, model_id: str) -> "VisionModel":
        from visionservex.core.model import VisionModel as _VisionModel
        with self._lock:
            entry = self._items.get(model_id)
            if entry is not None:
                self._items.move_to_end(model_id)
                entry.touch()
                return entry.model

            self._evict_if_needed()
            model = _VisionModel(model_id)
            model.warmup()
            entry = _CacheEntry(model)
            self._items[model_id] = entry
            self._items.move_to_end(model_id)
            _log.info("loaded model %s on device=%s", model_id, model.device)
            return model

    def unload(self, model_id: str) -> bool:
        with self._lock:
            entry = self._items.pop(model_id, None)
            if entry is None:
                return False
            try:
                entry.model.unload()
            except Exception:  # pragma: no cover - engine-specific
                _log.exception("error unloading %s", model_id)
            _log.info("unloaded model %s", model_id)
            return True

    def sweep(self) -> int:
        """Drop models that have been idle longer than the configured TTL."""
        with self._lock:
            now = time.monotonic()
            to_remove = [
                mid for mid, entry in self._items.items()
                if self.idle_unload_s > 0 and (now - entry.last_used) > self.idle_unload_s
            ]
            for mid in to_remove:
                self.unload(mid)
            return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            for mid in list(self._items):
                self.unload(mid)

    def _evict_if_needed(self) -> None:
        while len(self._items) >= self.max_loaded:
            oldest_id, _entry = next(iter(self._items.items()))
            self.unload(oldest_id)


class _CacheEntry:
    __slots__ = ("model", "loaded_at", "last_used")

    def __init__(self, model: "VisionModel") -> None:
        self.model = model
        self.loaded_at = time.monotonic()
        self.last_used = self.loaded_at

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def info(self) -> dict:
        return {
            "model_id": self.model.entry.id,
            "task": self.model.entry.task,
            "device": self.model.device,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "engine": self.model.entry.engine,
        }


_default: ModelCache | None = None
_default_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = ModelCache()
    return _default


__all__ = ["ModelCache", "get_model_cache"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""High-level ``VisionModel`` API.

A ``VisionModel`` is the friendly facade users interact with. It hides
engine selection, device choice, weight download, and result construction
behind a single object.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

from visionservex.config import get_settings
from visionservex.core.results import BaseResult
from visionservex.engines.base import BaseEngine, MissingDependencyError
from visionservex.engines.registry import build_engine
from visionservex.registry import ModelEntry, default_registry
from visionservex.runtime.device import resolve_device
from visionservex.runtime.downloads import (
    DownloadError,
    cached_path,
    download,
    is_cached,
)
from visionservex.utils.images import open_safe
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class VisionModel:
    """User-facing wrapper around an engine and a registry entry.

    Example::

        model = VisionModel("mock-detect")
        result = model.predict("image.jpg")
        result.save("out.jpg")

    Auto-pull example (Python only; server is a separate config path)::

        model = VisionModel("grounding-dino-tiny", auto_pull=True)
        result = model.predict("image.jpg", prompts=["cat", "dog"])
    """

    def __init__(
        self,
        model_id: str,
        *,
        task: str | None = None,
        device: str | None = None,
        precision: str | None = None,
        auto_pull: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.entry: ModelEntry = default_registry().get(model_id)
        if task is not None and task != self.entry.task:
            raise ValueError(
                f"model {model_id!r} is registered for task {self.entry.task!r}, "
                f"not {task!r}"
            )

        chosen_device = resolve_device(
            preference=device or self.settings.runtime.device_preference,
            supported=self.entry.supported_devices,
        )
        chosen_precision = self._resolve_precision(precision, chosen_device)
        self.device = chosen_device
        self.precision = chosen_precision
        self.auto_pull = auto_pull

        self.engine: BaseEngine = build_engine(self.entry)
        self._loaded = False
        self._cache_path: str | None = None
        self._model_loaded_from: str | None = None

    # ------- properties -------

    def _resolve_precision(self, precision: str | None, device: str) -> str:
        pref = precision or self.settings.runtime.precision_preference
        if pref == "auto":
            if device == "cpu":
                return "fp32"
            return "fp16" if "fp16" in self.entry.supported_precisions else "fp32"
        if pref in self.entry.supported_precisions:
            return pref
        return self.entry.supported_precisions[0]

    # ------- lifecycle -------

    def warmup(self) -> None:
        """Load weights eagerly and run a tiny dummy inference if supported."""
        self._ensure_loaded()
        self.engine.warmup()

    def unload(self) -> None:
        if self._loaded:
            self.engine.unload()
            self._loaded = False

    def _ensure_weights(self) -> None:
        """Ensure local weights exist; pull if allowed."""
        if self.entry.download_type == "synthetic":
            return
        if is_cached(self.entry):
            cp = cached_path(self.entry)
            self._cache_path = str(cp) if cp else None
            self._model_loaded_from = "cache"
            return
        if not self.auto_pull:
            return  # engine.load() will surface a clearer error if it needs weights
        try:
            path = download(self.entry)
            self._cache_path = str(path)
            self._model_loaded_from = self.entry.download_type
        except DownloadError as exc:
            raise RuntimeError(
                f"could not auto-pull weights for {self.entry.id!r}: {exc}"
            ) from exc

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._ensure_weights()
            self.engine.load(device=self.device, precision=self.precision)
            self._loaded = True

    def __enter__(self) -> VisionModel:
        self._ensure_loaded()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()

    # ------- inference -------

    def predict(
        self,
        image: Image.Image | bytes | str | Path,
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> BaseResult:
        self._ensure_loaded()
        pil = self._coerce_image(image)
        start = time.perf_counter()
        result = self.engine.predict(pil, prompts=prompts, **kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        result.model_id = self.entry.id
        result.task = self.entry.task
        result.device = self.device
        result.precision = self.precision
        result.backend = getattr(self.engine, "backend_label", self.entry.backend) or self.entry.backend
        result.model_loaded_from = self._model_loaded_from
        result.cache_path = self._cache_path
        result.image_size = pil.size
        result._image = pil
        return result

    def batch_predict(
        self,
        images: Iterable[Image.Image | bytes | str | Path],
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> list[BaseResult]:
        results: list[BaseResult] = []
        for img in images:
            results.append(self.predict(img, prompts=prompts, **kwargs))
        return results

    def stream(
        self,
        images: Iterable[Image.Image | bytes | str | Path],
        **kwargs: Any,
    ):
        for img in images:
            yield self.predict(img, **kwargs)

    # ------- introspection / utilities -------

    def info(self) -> dict[str, Any]:
        return {
            "id": self.entry.id,
            "task": self.entry.task,
            "device": self.device,
            "precision": self.precision,
            "engine": self.entry.engine,
            "backend": self.entry.backend,
            "status": self.entry.status,
            "implementation_status": self.entry.implementation_status,
            "license": self.entry.license,
            "loaded": self._loaded,
            "cache_path": self._cache_path,
            "auto_pull": self.auto_pull,
        }

    def export(self, format: str, output_path: str | Path) -> Path:
        return self.engine.export(format, Path(output_path))

    def benchmark(self, image: Image.Image | bytes | str | Path, *, n: int = 5) -> dict[str, Any]:
        pil = self._coerce_image(image)
        self._ensure_loaded()
        self.engine.warmup()
        latencies = []
        for _ in range(max(1, n)):
            t0 = time.perf_counter()
            self.engine.predict(pil)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        latencies.sort()
        return {
            "n": len(latencies),
            "p50_ms": latencies[len(latencies) // 2],
            "p90_ms": latencies[min(len(latencies) - 1, int(len(latencies) * 0.9))],
            "p99_ms": latencies[-1],
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "device": self.device,
            "model_id": self.entry.id,
            "backend": self.entry.backend,
        }

    # ------- helpers -------

    def _coerce_image(self, image: Image.Image | bytes | str | Path) -> Image.Image:
        limits = self.settings.limits
        if isinstance(image, Image.Image):
            w, h = image.size
            if w > limits.max_image_dim or h > limits.max_image_dim:
                raise ValueError(f"image dim {w}x{h} exceeds {limits.max_image_dim}")
            if w * h > limits.max_image_pixels:
                raise ValueError("image pixel area exceeds max_image_pixels")
            return image.convert("RGB") if image.mode != "RGB" else image
        if isinstance(image, (bytes, bytearray)):
            return open_safe(bytes(image), max_pixels=limits.max_image_pixels, max_dim=limits.max_image_dim)
        if isinstance(image, (str, Path)):
            return open_safe(Path(image), max_pixels=limits.max_image_pixels, max_dim=limits.max_image_dim)
        raise TypeError(f"unsupported image input type: {type(image).__name__}")


__all__ = ["VisionModel"]

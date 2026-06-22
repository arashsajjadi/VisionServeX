# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Batch inference telemetry + true-forward-batch verification (v3.22.0).

Two responsibilities:

1. **Honesty enforcement.** :func:`verify_true_forward_batch` counts how many
   times an engine's underlying model ``forward`` is called while running
   ``predict_batch`` on N>1 images. A genuine tensor batch calls ``forward``
   exactly ONCE. An engine that loops over single-image ``predict`` calls it N
   times. This is the mechanism that makes ``supports_true_batch=True`` a claim
   that can FAIL a test, not a self-assertion.

2. **Measured metadata.** :func:`run_batch_with_telemetry` runs a batch and
   returns a :class:`BatchTelemetry` carrying the per-stage timings, GPU util
   (sampled via NVML), and VRAM peaks that populate the ``/infer-batch`` response.
   Nothing here claims "GPU saturation" — it reports what NVML and torch.cuda
   measured.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from visionservex.runtime.gpu_lifecycle import (
    clear_torch_cuda_cache,
    force_gc,
    get_gpu_memory_state,
)


# --------------------------------------------------------------------------- #
# NVML utilization sampler
# --------------------------------------------------------------------------- #
class _NvmlSampler:
    """Background sampler of GPU utilization + free VRAM via NVML.

    Safe no-op if pynvml/NVML is unavailable. Samples on a thread because a
    batch forward can complete in <50 ms — too fast for a single poll.
    """

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self._utils: list[int] = []
        self._free_mb: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._pynvml = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception:
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    def _loop(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        while not self._stop.is_set():
            try:
                u = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                m = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._utils.append(int(u.gpu))
                self._free_mb.append(m.free / (1024**2))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> _NvmlSampler:
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict[str, Any]:
        return {
            "gpu_util_avg": (
                round(sum(self._utils) / len(self._utils), 1) if self._utils else None
            ),
            "gpu_util_peak": (max(self._utils) if self._utils else None),
            "vram_free_min_mb": (round(min(self._free_mb), 1) if self._free_mb else None),
            "nvml_samples": len(self._utils),
        }


# --------------------------------------------------------------------------- #
# Telemetry record (mirrors the /infer-batch response contract)
# --------------------------------------------------------------------------- #
@dataclass
class BatchTelemetry:
    batch_mode: str = "internal_loop"
    true_batch_supported: bool = False
    true_forward_batch: bool = False
    internal_loop: bool = True
    requested_batch_size: int = 0
    actual_batch_size: int = 0
    microbatch_size: int = 0
    batch_trajectory: list[int] = field(default_factory=list)
    preprocess_ms: float = 0.0
    forward_ms: float = 0.0
    postprocess_ms: float = 0.0
    mask_polygon_ms: float = 0.0
    nms_ms: float = 0.0
    decode_ms: float = 0.0
    encode_ms: float = 0.0
    total_ms: float = 0.0
    per_image_timings: list[dict[str, float]] = field(default_factory=list)
    gpu_util_avg: float | None = None
    gpu_util_peak: int | None = None
    vram_used_peak_mb: float = 0.0
    vram_reserved_peak_mb: float = 0.0
    vram_free_min_mb: float | None = None
    oom_recovered: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Forward-call counting (the anti-fake mechanism)
# --------------------------------------------------------------------------- #
def count_forward_calls(model_obj: Any, fn: Callable[[], Any]) -> tuple[Any, int]:
    """Run ``fn`` while counting calls to ``model_obj.forward``.

    Returns ``(fn_result, n_forward_calls)``. Restores the original forward.
    """
    if model_obj is None or not hasattr(model_obj, "forward"):
        return fn(), -1
    calls = {"n": 0}
    orig = model_obj.forward

    def _wrapped(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return orig(*a, **k)

    model_obj.forward = _wrapped  # type: ignore[assignment]
    try:
        result = fn()
    finally:
        model_obj.forward = orig  # type: ignore[assignment]
    return result, calls["n"]


def verify_true_forward_batch(engine: Any, images: list[Any], **kwargs: Any) -> dict[str, Any]:
    """Empirically verify whether ``engine.predict_batch`` is a TRUE forward batch.

    A true tensor batch calls the underlying model ``forward`` EXACTLY ONCE for
    N>1 images. A hidden internal loop calls it N times. This is what makes the
    ``supports_true_batch`` flag falsifiable.

    Returns a dict with ``forward_calls``, ``n_images``, ``is_true_forward_batch``
    and ``claimed`` (the engine's self-declared flag) so a test can assert the
    claim matches reality.
    """
    n = len(images)
    model_obj = getattr(engine, "_model", None) or getattr(engine, "_rfdetr_model", None)
    claimed = bool(getattr(engine, "supports_true_batch", False))
    if model_obj is None or not hasattr(model_obj, "forward"):
        return {
            "forward_calls": -1,
            "n_images": n,
            "is_true_forward_batch": False,
            "claimed": claimed,
            "verifiable": False,
        }
    _, calls = count_forward_calls(model_obj, lambda: engine.predict_batch(images, **kwargs))
    return {
        "forward_calls": calls,
        "n_images": n,
        "is_true_forward_batch": bool(calls == 1 and n > 1),
        "claimed": claimed,
        "verifiable": True,
        "claim_matches_reality": (claimed == (calls == 1 and n > 1)) if n > 1 else None,
    }


# --------------------------------------------------------------------------- #
# Telemetry-wrapped batch run
# --------------------------------------------------------------------------- #
def run_batch_with_telemetry(
    model: Any,
    images: list[Any],
    *,
    prompts: list[str] | None = None,
    **kwargs: Any,
) -> tuple[list[Any], BatchTelemetry]:
    """Run ``model.batch_predict`` (or engine.predict_batch) with measured telemetry.

    ``model`` may be a ``VisionModel`` (uses ``batch_predict``/``supports_true_batch``)
    or a raw engine (uses ``predict_batch``). Never asserts "saturation" — only
    reports NVML + torch.cuda measurements.
    """
    tel = BatchTelemetry(requested_batch_size=len(images), actual_batch_size=len(images))
    tel.microbatch_size = len(images)
    tel.batch_trajectory = [len(images)]

    is_vision_model = hasattr(model, "batch_predict") and hasattr(model, "supports_true_batch")
    tel.true_batch_supported = bool(getattr(model, "supports_true_batch", False))

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # type: ignore

    start = time.perf_counter()
    with _NvmlSampler() as sampler:
        try:
            if is_vision_model:
                results = model.batch_predict(images, prompts=prompts, **kwargs)
            else:
                results = model.predict_batch(images, prompts=prompts, **kwargs)
        except Exception as exc:  # OOM or other — surface honestly
            if torch is not None and "out of memory" in str(exc).lower():
                clear_torch_cuda_cache()
                force_gc()
                tel.oom_recovered = False
                tel.fallback_reason = f"OOM at batch_size={len(images)}: {str(exc)[:160]}"
            else:
                tel.fallback_reason = str(exc)[:200]
            raise
    tel.total_ms = round((time.perf_counter() - start) * 1000.0, 3)

    # Aggregate per-stage timings + batch_mode from result metadata (engine-set).
    if results:
        md0 = getattr(results[0], "metadata", {}) or {}
        tel.batch_mode = md0.get("batch_mode", "internal_loop")
        tel.true_forward_batch = bool(md0.get("true_forward_batch", False))
        tel.internal_loop = bool(md0.get("internal_loop", not tel.true_forward_batch))
        for r in results:
            md = getattr(r, "metadata", {}) or {}
            tel.per_image_timings.append(
                {
                    "preprocess_ms": float(md.get("preprocess_ms", 0.0)),
                    "forward_ms": float(md.get("forward_ms", 0.0)),
                    "postprocess_ms": float(md.get("postprocess_ms", 0.0)),
                }
            )
        tel.preprocess_ms = round(sum(t["preprocess_ms"] for t in tel.per_image_timings), 3)
        tel.forward_ms = round(sum(t["forward_ms"] for t in tel.per_image_timings), 3)
        tel.postprocess_ms = round(sum(t["postprocess_ms"] for t in tel.per_image_timings), 3)

    nv = sampler.summary()
    tel.gpu_util_avg = nv["gpu_util_avg"]
    tel.gpu_util_peak = nv["gpu_util_peak"]
    tel.vram_free_min_mb = nv["vram_free_min_mb"]

    mem = get_gpu_memory_state("batch_peak")
    tel.vram_used_peak_mb = round(mem.max_allocated_mb, 1)
    tel.vram_reserved_peak_mb = round(mem.reserved_mb, 1)
    return results, tel


__all__ = [
    "BatchTelemetry",
    "count_forward_calls",
    "run_batch_with_telemetry",
    "verify_true_forward_batch",
]

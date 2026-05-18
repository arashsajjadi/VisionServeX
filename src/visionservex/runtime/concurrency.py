# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.18.0: concurrency profile + shared-model request benchmark.

`build_concurrency_profile()` is GPU-aware. The profile maps the active
``gpu_profile`` (from :mod:`runtime.gpu_profile`) to recommended worker
counts and a structured policy. The benchmark runs N concurrent requests
through a single loaded model and reports throughput / latency / VRAM.
The runner refuses to spawn multiple heavy-model processes — concurrency
is request-level, not model-level, by design.
"""

from __future__ import annotations

import contextlib
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ConcurrencyProfile",
    "build_concurrency_profile",
    "run_concurrency_benchmark",
]


@dataclass
class ConcurrencyProfile:
    """Result of :func:`build_concurrency_profile`."""

    gpu_name: str
    gpu_profile: str
    total_vram_gb: float
    recommended_small_model_workers: int
    recommended_medium_model_workers: int
    recommended_heavy_model_workers: int
    recommended_cpu_workers: int
    max_safe_concurrent_requests: int
    policy: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_name": self.gpu_name,
            "gpu_profile": self.gpu_profile,
            "total_vram_gb": round(self.total_vram_gb, 2),
            "recommended_small_model_workers": self.recommended_small_model_workers,
            "recommended_medium_model_workers": self.recommended_medium_model_workers,
            "recommended_heavy_model_workers": self.recommended_heavy_model_workers,
            "recommended_cpu_workers": self.recommended_cpu_workers,
            "max_safe_concurrent_requests": self.max_safe_concurrent_requests,
            "policy": dict(self.policy),
            "warnings": list(self.warnings),
        }


def build_concurrency_profile() -> ConcurrencyProfile:
    """Build a concurrency profile from the active GPU profile."""
    from visionservex.runtime.gpu_profile import detect_gpu_profile

    gpu = detect_gpu_profile()
    profile = gpu.profile
    warnings_: list[str] = list(gpu.notes)

    if profile in ("h100_colab", "desktop_32gb_plus") or profile == "a100_colab":
        small, medium, heavy, cpu = 4, 2, 1, 1
        max_concurrent = 8
    elif profile == "desktop_24gb_fast":
        small, medium, heavy, cpu = 3, 2, 1, 1
        max_concurrent = 4
    elif profile in ("desktop_16gb_fast", "l4_colab"):
        small, medium, heavy, cpu = 2, 1, 1, 1
        max_concurrent = 2
    elif profile == "t4_colab":
        small, medium, heavy, cpu = 1, 1, 1, 1
        max_concurrent = 2
        warnings_.append("T4 has tight VRAM; concurrency > 1 may OOM on medium/heavy models.")
    elif profile == "cpu_only":
        small, medium, heavy, cpu = 1, 1, 1, 2
        max_concurrent = 1
        warnings_.append("No CUDA: model concurrency is single-process / single-stream.")
    else:
        small, medium, heavy, cpu = 1, 1, 1, 1
        max_concurrent = 1
        warnings_.append(f"Unknown profile {profile!r}; defaulting to single-worker.")

    policy = {
        "small_models_parallel": small > 1,
        "medium_models_parallel": medium > 1,
        "heavy_models_parallel": heavy > 1,
        "sidecars_parallel": False,
    }
    return ConcurrencyProfile(
        gpu_name=gpu.gpu_name,
        gpu_profile=profile,
        total_vram_gb=gpu.total_vram_gb,
        recommended_small_model_workers=small,
        recommended_medium_model_workers=medium,
        recommended_heavy_model_workers=heavy,
        recommended_cpu_workers=cpu,
        max_safe_concurrent_requests=max_concurrent,
        policy=policy,
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Concurrency benchmark
# ---------------------------------------------------------------------------


def run_concurrency_benchmark(
    *,
    model_id: str,
    image_paths: list[Path],
    device: str,
    require_gpu: bool,
    concurrency_levels: list[int],
    request_mode: str,
    sample_gpu: bool = False,
    gpu_sample_interval: float = 0.5,
) -> dict[str, Any]:
    """Run shared-model concurrency benchmark.

    The model loads once. For each concurrency level we dispatch up to
    ``len(image_paths)`` requests through a thread pool that all share the
    same loaded model. Latency, throughput, and optional GPU sampling are
    reported per level.

    ``request_mode`` accepts ``"shared-model"`` (the supported mode) or
    ``"separate-process"`` (returns SHARED_MODEL_CONCURRENCY_NOT_SUPPORTED
    because the current package does not isolate models per process safely
    in a single benchmark call).
    """
    if request_mode not in ("shared-model",):
        return {
            "status": "expected_blocker",
            "code": "SHARED_MODEL_CONCURRENCY_NOT_SUPPORTED",
            "request_mode": request_mode,
            "message": (
                "v2.18.0 supports request_mode=shared-model only. "
                "separate-process concurrency requires a multi-process orchestrator "
                "that isolates VRAM; this is on the v2.19 roadmap."
            ),
            "model_id": model_id,
            "runs": [],
            "summary": {},
        }

    from PIL import Image as _PIL

    from visionservex.core.model import VisionModel
    from visionservex.engines.base import MissingDependencyError
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired
    from visionservex.runtime.gpu_profile import detect_gpu_profile
    from visionservex.runtime.persistent_benchmark import GpuUtilizationSampler

    gpu_info = detect_gpu_profile()

    # Load model once
    try:
        model = VisionModel(model_id, device=device)
        model._ensure_loaded()
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        return {
            "status": "expected_blocker",
            "code": "DEPENDENCY_REQUIRED",
            "model_id": model_id,
            "errors": [str(exc)[:300]],
            "runs": [],
            "summary": {},
        }
    except Exception as exc:
        return {
            "status": "failed",
            "code": "MODEL_LOAD_FAILED",
            "model_id": model_id,
            "errors": [str(exc)[:300]],
            "runs": [],
            "summary": {},
        }

    device_actual = str(getattr(model, "device", device))
    if require_gpu and not device_actual.startswith("cuda"):
        with contextlib.suppress(Exception):
            model.close()
        return {
            "status": "failed",
            "code": "GPU_REQUIRED_NOT_USED",
            "model_id": model_id,
            "device_requested": device,
            "device_actual": device_actual,
            "errors": [
                f"--require-gpu set but device_actual={device_actual}; refusing concurrency benchmark."
            ],
            "runs": [],
            "summary": {},
        }

    runs: list[dict[str, Any]] = []
    images = [_PIL.open(p).convert("RGB") for p in image_paths]
    n_requests = len(images)

    try:
        for level in concurrency_levels:
            if level < 1:
                continue
            sampler = GpuUtilizationSampler(interval=gpu_sample_interval) if sample_gpu else None
            if sampler is not None:
                sampler.start()

            latencies: list[float] = []
            errors: list[str] = []
            lock = threading.Lock()

            def _worker(
                idx: int,
                _latencies: list[float] = latencies,
                _errors: list[str] = errors,
                _lock: threading.Lock = lock,
            ) -> None:
                t0 = time.perf_counter()
                try:
                    model.predict(images[idx % n_requests], threshold=0.001)
                    with _lock:
                        _latencies.append((time.perf_counter() - t0) * 1000.0)
                except Exception as exc:
                    with _lock:
                        _errors.append(str(exc)[:200])

            t_total = time.perf_counter()
            # Dispatch up to n_requests in waves of `level` concurrent threads.
            remaining = n_requests
            i = 0
            while remaining > 0:
                wave = min(level, remaining)
                batch = [threading.Thread(target=_worker, args=(i + j,)) for j in range(wave)]
                for t in batch:
                    t.start()
                for t in batch:
                    t.join()
                i += wave
                remaining -= wave
            wall_seconds = time.perf_counter() - t_total

            n_success = len(latencies)
            n_failed = n_requests - n_success
            lat_sorted = sorted(latencies)

            def _pct(xs: list[float], q: float) -> float:
                if not xs:
                    return 0.0
                return float(xs[min(len(xs) - 1, int(len(xs) * q))])

            row = {
                "model_id": model_id,
                "concurrency": level,
                "request_mode": request_mode,
                "n_requests": n_requests,
                "n_success": n_success,
                "n_failed": n_failed,
                "device_actual": device_actual,
                "load_count": 1,
                "wall_seconds": round(wall_seconds, 3),
                "throughput_req_per_sec": round(n_success / wall_seconds, 2)
                if wall_seconds > 0
                else 0.0,
                "latency_ms_p50": round(_pct(lat_sorted, 0.50), 2),
                "latency_ms_p95": round(_pct(lat_sorted, 0.95), 2),
                "latency_ms_p99": round(_pct(lat_sorted, 0.99), 2),
                "latency_ms_mean": round(statistics.fmean(latencies) if latencies else 0.0, 2),
                "errors": errors[:5],
                "warnings": [],
            }

            if sampler is not None:
                sampler.stop()
                summary = sampler.summary()
                row["gpu_utilization_mean"] = summary["utilization_mean"]
                row["gpu_utilization_p95"] = summary["utilization_p95"]
                row["vram_peak_gb"] = summary["vram_used_peak_gb"]
                row["warnings"].extend(summary.get("warnings", []))
            else:
                row["gpu_utilization_mean"] = None
                row["gpu_utilization_p95"] = None
                row["vram_peak_gb"] = None

            runs.append(row)
    finally:
        with contextlib.suppress(Exception):
            model.close()

    # Summary
    if not runs:
        return {
            "status": "failed",
            "code": "NO_RUNS",
            "model_id": model_id,
            "runs": [],
            "summary": {},
        }
    safe_runs = [r for r in runs if r["n_failed"] == 0]
    best_throughput = max(runs, key=lambda r: r["throughput_req_per_sec"])
    summary = {
        "best_safe_concurrency": (max((r["concurrency"] for r in safe_runs), default=0)),
        "best_throughput_concurrency": best_throughput["concurrency"],
        "best_throughput_req_per_sec": best_throughput["throughput_req_per_sec"],
        "resource_blocked_runs": 0,
        "failed_runs": sum(1 for r in runs if r["n_failed"] > 0),
    }
    return {
        "status": "ok" if all(r["n_failed"] == 0 for r in runs) else "partial",
        "code": "OK",
        "gpu_name": gpu_info.gpu_name,
        "gpu_profile": gpu_info.profile,
        "model_id": model_id,
        "request_mode": request_mode,
        "runs": runs,
        "summary": summary,
    }

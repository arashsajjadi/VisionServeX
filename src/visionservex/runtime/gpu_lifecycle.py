# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""GPU VRAM lifecycle manager.

Prevents stepwise VRAM accumulation during repeated model loads and benchmarks.

Problem solved:
    Without explicit cleanup, each model load adds GPU memory that the CUDA
    allocator retains in its pool. After five models you may have filled VRAM
    even though only one model is logically "loaded."

Correct cleanup sequence
------------------------
1. Delete all Python references to the model, processor, and tensors.
2. Run Python garbage collection (triggers __del__ on reference-counted objects).
3. If CUDA is available:
   a. Synchronize the current stream (flush async work).
   b. Empty the CUDA allocator cache (returns memory to the OS pool).
   c. Collect inter-process CUDA handles.
   d. Reset peak memory statistics.

Safe limits
-----------
- Never kills GUI processes (gnome-shell, kwin, Xorg, ...).
- Never resets GPU device (no torch.cuda.reset_device or driver-level reset).
- Never terminates non-VisionServeX processes.
- Reports memory state without acting on it unless asked.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryState:
    """Snapshot of GPU memory at a point in time."""

    label: str
    timestamp: float = field(default_factory=time.time)
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    max_allocated_mb: float = 0.0
    cuda_available: bool = False
    device_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "timestamp": round(self.timestamp, 3),
            "allocated_mb": round(self.allocated_mb, 1),
            "reserved_mb": round(self.reserved_mb, 1),
            "max_allocated_mb": round(self.max_allocated_mb, 1),
            "cuda_available": self.cuda_available,
            "device_name": self.device_name,
        }

    def growth_vs(self, other: MemoryState) -> float:
        """MB growth in allocated memory since `other`."""
        return self.allocated_mb - other.allocated_mb

    def reserved_growth_vs(self, other: MemoryState) -> float:
        """MB growth in reserved memory since `other`."""
        return self.reserved_mb - other.reserved_mb


def get_gpu_memory_state(label: str = "") -> MemoryState:
    """Snapshot current GPU memory. Returns zeroed state if CUDA unavailable."""
    state = MemoryState(label=label)
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return state
        state.cuda_available = True
        state.device_name = torch.cuda.get_device_name(0)
        state.allocated_mb = torch.cuda.memory_allocated() / (1024**2)
        state.reserved_mb = torch.cuda.memory_reserved() / (1024**2)
        state.max_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        pass
    return state


def get_process_gpu_memory() -> dict[str, Any]:
    """Return per-process GPU memory from nvidia-smi (if available)."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        rows = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                rows.append(
                    {
                        "pid": parts[0],
                        "process_name": parts[1],
                        "used_memory_mb": float(parts[2]) if parts[2].isdigit() else 0,
                    }
                )
        return {"available": True, "processes": rows}
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:100]}


def clear_torch_cuda_cache() -> None:
    """Run the full CUDA cleanup sequence (does NOT delete model objects).

    Each CUDA call is wrapped independently so that a pending CUDA error in
    one call (e.g. OOM from a previous stream) does not prevent subsequent
    cleanup steps from running.
    """
    import contextlib

    gc.collect()
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return
        with contextlib.suppress(Exception):
            torch.cuda.synchronize()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def unload_model_object(model: Any) -> None:
    """Best-effort unload of a model object, moving weights to CPU first.

    Handles HF Transformers models, rfdetr models, and VisionModel wrappers.
    """
    try:
        # VisionModel wrapper
        if hasattr(model, "unload"):
            model.unload()
            return
    except Exception:
        pass

    # Move to CPU to free VRAM before deleting
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass

    # HF model: explicitly delete sub-modules
    for attr_name in ("_model", "model", "encoder", "decoder", "backbone", "processor"):
        try:
            sub = getattr(model, attr_name, None)
            if sub is not None:
                try:
                    if hasattr(sub, "cpu"):
                        sub.cpu()
                except Exception:
                    pass
                delattr(model, attr_name)
                del sub
        except Exception:
            pass


def release_inference_context(*objects: Any) -> None:
    """Delete inference objects and flush GPU caches.

    Usage::

        release_inference_context(model, processor, inputs, outputs, result)
    """
    import contextlib

    for obj in objects:
        with contextlib.suppress(Exception):
            unload_model_object(obj)
        with contextlib.suppress(Exception):
            del obj
    force_gc()
    clear_torch_cuda_cache()


def force_gc() -> None:
    """Force Python garbage collection (three generations)."""
    for _ in range(3):
        gc.collect()


def maybe_cuda_ipc_collect() -> None:
    """Collect CUDA IPC handles if available (safe no-op if not)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    except Exception:
        pass


def memory_checkpoint(label: str = "") -> MemoryState:
    """Record a GPU memory checkpoint. Returns current state."""
    return get_gpu_memory_state(label)


def assert_memory_returned_to_baseline(
    baseline: MemoryState,
    current: MemoryState,
    *,
    max_growth_mb: float = 256.0,
    check_reserved: bool = False,
) -> dict[str, Any]:
    """Compare current vs baseline memory. Returns result dict.

    Does NOT raise — returns ``ok`` or ``warning`` so callers can decide.
    """
    alloc_growth = current.growth_vs(baseline)
    reserved_growth = current.reserved_growth_vs(baseline)

    result: dict[str, Any] = {
        "baseline_allocated_mb": baseline.allocated_mb,
        "current_allocated_mb": current.allocated_mb,
        "allocated_growth_mb": round(alloc_growth, 1),
        "reserved_growth_mb": round(reserved_growth, 1),
        "max_growth_mb": max_growth_mb,
    }

    if alloc_growth > max_growth_mb:
        result["status"] = "warning"
        result["message"] = (
            f"GPU allocated memory grew by {alloc_growth:.1f} MB "
            f"(limit {max_growth_mb:.0f} MB). "
            "VRAM may not be fully released. Run 'visionservex gpu cleanup-cache'."
        )
    elif check_reserved and reserved_growth > max_growth_mb * 2:
        result["status"] = "info"
        result["message"] = (
            f"GPU reserved (CUDA cache) grew by {reserved_growth:.1f} MB. "
            "This is normal for the CUDA allocator; allocated memory is within limits."
        )
    else:
        result["status"] = "ok"
        result["message"] = (
            f"Memory returned to near-baseline (growth: {alloc_growth:.1f} MB allocated, "
            f"{reserved_growth:.1f} MB reserved)."
        )

    return result


def recommend_process_restart_if_needed(
    baseline: MemoryState, current: MemoryState, *, threshold_mb: float = 1024.0
) -> str | None:
    """Return a restart recommendation message if memory growth is severe, else None."""
    growth = current.growth_vs(baseline)
    if growth > threshold_mb:
        return (
            f"VRAM grew by {growth:.1f} MB since baseline. "
            "This may indicate a leak that persists across Python objects. "
            "Consider restarting the Python process or using --isolate-process mode."
        )
    return None


def cleanup_gpu_after_model(
    model: Any | None = None,
    *extra_objects: Any,
    verbose: bool = False,
) -> MemoryState:
    """Standard cleanup after a model run. Returns post-cleanup memory state.

    Performs the full cleanup sequence:
    1. Unload model.
    2. Delete extra objects.
    3. Python GC.
    4. CUDA cache flush.
    """
    before = get_gpu_memory_state("before_cleanup")

    import contextlib

    if model is not None:
        with contextlib.suppress(Exception):
            unload_model_object(model)

    for obj in extra_objects:
        with contextlib.suppress(Exception):
            del obj

    force_gc()
    clear_torch_cuda_cache()

    after = get_gpu_memory_state("after_cleanup")

    if verbose:
        freed = before.allocated_mb - after.allocated_mb
        print(
            f"[gpu_lifecycle] freed {freed:.1f} MB allocated | {before.reserved_mb:.1f} → {after.reserved_mb:.1f} MB reserved"
        )

    return after


__all__ = [
    "MemoryState",
    "assert_memory_returned_to_baseline",
    "cleanup_gpu_after_model",
    "clear_torch_cuda_cache",
    "force_gc",
    "get_gpu_memory_state",
    "get_process_gpu_memory",
    "maybe_cuda_ipc_collect",
    "memory_checkpoint",
    "recommend_process_restart_if_needed",
    "release_inference_context",
    "unload_model_object",
]

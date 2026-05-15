# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Device detection, sanity-checking, and selection.

Preference order for ``device=auto``:
  Linux/Windows: CUDA (highest free VRAM, passes sanity) → ROCm → DirectML → CPU
  macOS:          MPS (passes sanity) → CPU

A device "sanity check" allocates a tiny tensor and runs a small operation.
If that fails, the device is marked unavailable and the next candidate is tried.
This catches libnvrtc/runtime/library mismatches before they surface at
inference time.

Multi-GPU: all CUDA devices are probed; the one with the most free VRAM
(and passing the sanity check) is chosen for ``auto``.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DeviceInfo:
    """A device VisionServeX may use for inference."""

    name: str
    available: bool
    detail: str = ""
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    capability: str | None = None
    sanity_ok: bool | None = None  # None = not yet checked; True/False = result
    sanity_error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "total_vram_gb": (round(self.total_vram_gb, 2) if self.total_vram_gb else None),
            "free_vram_gb": (round(self.free_vram_gb, 2) if self.free_vram_gb else None),
            "capability": self.capability,
            "sanity_ok": self.sanity_ok,
            "sanity_error": self.sanity_error,
            "extras": self.extras,
        }


# ------------------------------------------------------------------ sanity ---


def _cuda_sanity(device_idx: int) -> tuple[bool, str]:
    """Try to allocate and multiply a tiny CUDA tensor."""
    try:
        import torch  # type: ignore

        x = torch.ones(3, 3, device=f"cuda:{device_idx}")
        _ = (x @ x).sum().item()
        return True, ""
    except Exception as exc:
        short = str(exc).split("\n")[0][:200]
        return False, short


def _mps_sanity() -> tuple[bool, str]:
    try:
        import torch  # type: ignore

        x = torch.ones(3, 3, device="mps")
        _ = (x + x).sum().item()
        return True, ""
    except Exception as exc:
        return False, str(exc)[:200]


# ------------------------------------------------------------------ probes ---


def _all_cuda_devices() -> list[DeviceInfo]:
    """Return a DeviceInfo for every CUDA device, or one failure entry."""
    try:
        import torch  # type: ignore
    except Exception:
        smi = _nvidia_smi_summary()
        if smi:
            return [
                DeviceInfo(
                    name="cuda",
                    available=False,
                    detail=f"GPU present ({smi}) but torch not installed. "
                    "Install: pip install 'visionservex[torch]'",
                )
            ]
        return [DeviceInfo(name="cuda", available=False, detail="torch not installed")]

    try:
        if not torch.cuda.is_available():
            return [
                DeviceInfo(name="cuda", available=False, detail="torch reports CUDA unavailable")
            ]
    except Exception as exc:
        return [
            DeviceInfo(
                name="cuda",
                available=False,
                detail=f"torch.cuda.is_available() raised: {exc!s:.80}",
            )
        ]

    count = torch.cuda.device_count()
    devices: list[DeviceInfo] = []
    for idx in range(count):
        try:
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory / (1024**3)
            free = total
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
                free = free_bytes / (1024**3)
                total = total_bytes / (1024**3)
            except Exception:
                pass
            cap = f"{props.major}.{props.minor}"
            sanity_ok, sanity_err = _cuda_sanity(idx)
            tag = f"cuda:{idx}" if count > 1 else "cuda"
            devices.append(
                DeviceInfo(
                    name=tag,
                    available=sanity_ok,
                    detail=f"{name} (cap {cap})"
                    + ("" if sanity_ok else f" [broken: {sanity_err[:60]}]"),
                    total_vram_gb=total,
                    free_vram_gb=free,
                    capability=cap,
                    sanity_ok=sanity_ok,
                    sanity_error=sanity_err or None,
                    extras={"index": idx, "count": count, "gpu_name": name},
                )
            )
        except Exception as exc:
            short = str(exc).split("\n")[0][:120]
            devices.append(
                DeviceInfo(
                    name=f"cuda:{idx}" if count > 1 else "cuda",
                    available=False,
                    detail=f"CUDA runtime error: {short}. Using CPU fallback.",
                    sanity_ok=False,
                    sanity_error=short,
                )
            )
    return devices


def _nvidia_smi_summary() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=4,
        )
        lines = (out.stdout or "").strip().splitlines()
        return lines[0].strip() if lines else None
    except Exception:
        return None


def _best_cuda() -> DeviceInfo | None:
    """Return the most capable, healthy CUDA device."""
    devices = _all_cuda_devices()
    healthy = [d for d in devices if d.available and d.sanity_ok is not False]
    if not healthy:
        return None
    # Prefer highest free VRAM; break ties by device index (lower = preferred)
    return max(healthy, key=lambda d: (d.free_vram_gb or 0.0, -(d.extras.get("index", 0))))


def _mps_info() -> DeviceInfo:
    if platform.system() != "Darwin":
        return DeviceInfo(name="mps", available=False, detail="not macOS")
    try:
        import torch  # type: ignore

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            sanity_ok, sanity_err = _mps_sanity()
            return DeviceInfo(
                name="mps",
                available=sanity_ok,
                detail="Apple MPS" + ("" if sanity_ok else f" [sanity failed: {sanity_err[:60]}]"),
                sanity_ok=sanity_ok,
                sanity_error=sanity_err or None,
            )
    except Exception:
        pass
    return DeviceInfo(name="mps", available=False, detail="MPS not available")


def _rocm_info() -> DeviceInfo:
    if os.environ.get("VISIONSERVEX_FORCE_ROCM") == "1":
        return DeviceInfo(name="rocm", available=True, detail="ROCm forced via env")
    try:
        import torch  # type: ignore

        hip = getattr(torch.version, "hip", None)
        if hip:
            return DeviceInfo(name="rocm", available=True, detail=f"ROCm/HIP {hip}")
    except Exception:
        pass
    return DeviceInfo(name="rocm", available=False, detail="ROCm not detected")


def _directml_info() -> DeviceInfo:
    if platform.system() != "Windows":
        return DeviceInfo(name="directml", available=False, detail="not Windows")
    try:
        import torch_directml  # type: ignore  # noqa: F401

        return DeviceInfo(name="directml", available=True, detail="torch_directml installed")
    except Exception:
        return DeviceInfo(name="directml", available=False, detail="torch_directml not installed")


def _cpu_info() -> DeviceInfo:
    return DeviceInfo(
        name="cpu",
        available=True,
        sanity_ok=True,
        detail=f"{platform.processor() or 'CPU'} ({os.cpu_count()} cores)",
    )


# ----------------------------------------------------------------- public ----


def available_devices() -> list[DeviceInfo]:
    """Return all devices, including per-GPU entries for multi-GPU systems."""
    cuda_devices = _all_cuda_devices()
    return [
        _cpu_info(),
        *cuda_devices,
        _mps_info(),
        _rocm_info(),
        _directml_info(),
    ]


def best_device(supported: Iterable[str] | None = None) -> DeviceInfo:
    """Pick the fastest healthy device, optionally restricted to ``supported``."""
    allowed = {s.lower() for s in supported} if supported is not None else None

    def _ok(d: DeviceInfo) -> bool:
        if not d.available:
            return False
        if d.sanity_ok is False:
            return False
        if allowed is None:
            return True
        base = d.name.split(":")[0].lower()
        return base in allowed or d.name.lower() in allowed

    # Try CUDA first (best GPU by free VRAM)
    best_cu = _best_cuda()
    if best_cu and _ok(best_cu):
        return best_cu

    # macOS MPS
    mps = _mps_info()
    if _ok(mps):
        return mps

    # ROCm
    rocm = _rocm_info()
    if _ok(rocm):
        return rocm

    # DirectML
    dml = _directml_info()
    if _ok(dml):
        return dml

    return _cpu_info()


def device_benchmark(device_name: str, *, quick: bool = True) -> dict[str, Any]:
    """Run a small synthetic benchmark and return timing + throughput."""
    results: dict[str, Any] = {"device": device_name, "ok": False}
    try:
        import torch  # type: ignore

        dev = torch.device(device_name)
        size = 256 if quick else 1024
        a = torch.randn(size, size, device=dev)
        b = torch.randn(size, size, device=dev)
        n = 5 if quick else 20
        t0 = time.perf_counter()
        for _ in range(n):
            _ = (a @ b).sum().item()
        elapsed = (time.perf_counter() - t0) / n
        results.update(
            {
                "ok": True,
                "matrix_size": size,
                "avg_ms": round(elapsed * 1000, 2),
                "throughput_gflops": round(2 * size**3 / elapsed / 1e9, 2),
            }
        )
    except Exception as exc:
        results["error"] = str(exc)[:200]
    return results


def resolve_device(
    *,
    preference: str,
    supported: Iterable[str],
) -> str:
    """Pick a device, honoring user preference, model support, and sanity."""
    supported_set = {s.lower() for s in supported}
    pref = (preference or "auto").lower()
    base_pref = pref.split(":", 1)[0]

    if pref == "auto":
        bd = best_device(supported=supported_set)
        return bd.name
    else:
        # Check the explicitly requested device passes sanity
        all_devs = {d.name: d for d in available_devices()}
        info = all_devs.get(pref) or all_devs.get(base_pref)
        if info and info.available and info.sanity_ok is not False:
            return pref if ":" in pref else base_pref
        # Fallback if user's choice is broken but base_pref matches something
        return "cpu"


__all__ = [
    "DeviceInfo",
    "available_devices",
    "best_device",
    "device_benchmark",
    "resolve_device",
]

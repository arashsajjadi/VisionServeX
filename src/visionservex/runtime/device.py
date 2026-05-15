# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Device detection and selection.

Supports CUDA (NVIDIA), MPS (Apple Silicon), CPU. ROCm and DirectML are
acknowledged but not auto-detected unless the user opts in.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class DeviceInfo:
    """A device VisionServeX may run inference on."""

    name: str
    available: bool
    detail: str = ""
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    capability: str | None = None  # e.g. "8.9" for compute capability
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
            "total_vram_gb": (round(self.total_vram_gb, 2) if self.total_vram_gb else None),
            "free_vram_gb": (round(self.free_vram_gb, 2) if self.free_vram_gb else None),
            "capability": self.capability,
            "extras": self.extras,
        }


def _cuda_info() -> DeviceInfo:
    try:
        import torch  # type: ignore
    except Exception:
        nvidia_smi = _nvidia_smi_summary()
        if nvidia_smi:
            return DeviceInfo(
                name="cuda",
                available=False,
                detail=f"GPU present ({nvidia_smi}) but torch is not installed. "
                       "Install with `pip install 'visionservex[torch]'`.",
            )
        return DeviceInfo(name="cuda", available=False, detail="torch not installed")

    try:
        if not torch.cuda.is_available():
            return DeviceInfo(name="cuda", available=False, detail="torch reports CUDA unavailable")
        idx = torch.cuda.current_device()
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
        return DeviceInfo(
            name="cuda",
            available=True,
            detail=f"{name} (cap {cap})",
            total_vram_gb=total,
            free_vram_gb=free,
            capability=cap,
            extras={"index": idx, "count": torch.cuda.device_count()},
        )
    except Exception as exc:  # pragma: no cover - hardware-specific
        return DeviceInfo(name="cuda", available=False, detail=f"torch.cuda probe failed: {exc}")


def _nvidia_smi_summary() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=4,
        )
        line = (out.stdout or "").strip().splitlines()
        if line:
            return line[0].strip()
    except Exception:
        return None
    return None


def _mps_info() -> DeviceInfo:
    if platform.system() != "Darwin":
        return DeviceInfo(name="mps", available=False, detail="not macOS")
    try:
        import torch  # type: ignore
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return DeviceInfo(
                name="mps",
                available=True,
                detail="Apple Metal Performance Shaders",
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
        detail=f"{platform.processor() or 'CPU'} ({os.cpu_count()} cores)",
    )


def available_devices() -> list[DeviceInfo]:
    return [
        _cpu_info(),
        _cuda_info(),
        _mps_info(),
        _rocm_info(),
        _directml_info(),
    ]


def best_device(supported: Iterable[str] | None = None) -> DeviceInfo:
    """Pick the most capable available device, optionally restricted to a set."""
    devices = available_devices()
    allowed = {s.lower() for s in supported} if supported else None
    for candidate in ("cuda", "mps", "rocm", "directml", "cpu"):
        info = next((d for d in devices if d.name == candidate), None)
        if info is None or not info.available:
            continue
        if allowed is not None and candidate not in allowed:
            continue
        return info
    return next(d for d in devices if d.name == "cpu")


def resolve_device(
    *,
    preference: str,
    supported: Iterable[str],
) -> str:
    """Pick a device honoring user preference and model support."""
    supported_set = {s.lower() for s in supported}
    devices = {d.name: d for d in available_devices()}
    pref = (preference or "auto").lower()

    # Handle indexed CUDA preference (e.g. "cuda:0")
    base_pref = pref.split(":", 1)[0]

    if pref == "auto":
        for candidate in ("cuda", "mps", "rocm", "directml", "cpu"):
            if candidate in supported_set and devices[candidate].available:
                return candidate
        return "cpu"

    info = devices.get(base_pref)
    if info and info.available:
        # Preserve specific index if user requested e.g. "cuda:1"
        return pref if ":" in pref else base_pref
    return "cpu"


__all__ = [
    "DeviceInfo",
    "available_devices",
    "best_device",
    "resolve_device",
]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""System information collection for diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a base dep but be defensive
    psutil = None  # type: ignore[assignment]


@dataclass
class CPUInfo:
    architecture: str
    logical_cores: int
    physical_cores: int | None
    brand: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "brand": self.brand,
        }


@dataclass
class MemoryInfo:
    total_gb: float
    available_gb: float
    percent_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_gb": round(self.total_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "percent_used": round(self.percent_used, 1),
        }


@dataclass
class DiskInfo:
    path: str
    total_gb: float
    free_gb: float
    percent_used: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "total_gb": round(self.total_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "percent_used": round(self.percent_used, 1),
        }


@dataclass
class SystemInfo:
    os_name: str
    os_release: str
    os_version: str
    machine: str
    python_version: str
    package_version: str
    package_install_path: str
    cache_path: str
    cpu: CPUInfo
    memory: MemoryInfo
    disk: DiskInfo
    is_64bit: bool
    is_linux: bool
    is_macos: bool
    is_windows: bool
    is_apple_silicon: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": {
                "name": self.os_name,
                "release": self.os_release,
                "version": self.os_version,
                "machine": self.machine,
                "is_linux": self.is_linux,
                "is_macos": self.is_macos,
                "is_windows": self.is_windows,
                "is_apple_silicon": self.is_apple_silicon,
            },
            "python": self.python_version,
            "package_version": self.package_version,
            "package_install_path": self.package_install_path,
            "cache_path": self.cache_path,
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disk": self.disk.to_dict(),
            "is_64bit": self.is_64bit,
            "extras": self.extras,
        }


def _cpu_info() -> CPUInfo:
    logical = os.cpu_count() or 1
    physical: int | None = None
    if psutil is not None:
        try:
            physical = psutil.cpu_count(logical=False) or None
        except Exception:  # pragma: no cover
            physical = None
    return CPUInfo(
        architecture=platform.machine(),
        logical_cores=logical,
        physical_cores=physical,
        brand=platform.processor() or "",
    )


def _memory_info() -> MemoryInfo:
    if psutil is None:  # pragma: no cover
        return MemoryInfo(total_gb=0.0, available_gb=0.0, percent_used=0.0)
    vm = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=vm.total / (1024**3),
        available_gb=vm.available / (1024**3),
        percent_used=float(vm.percent),
    )


def _disk_info(path: Path | str) -> DiskInfo:
    try:
        usage = shutil.disk_usage(str(path))
    except Exception:
        return DiskInfo(path=str(path), total_gb=0.0, free_gb=0.0, percent_used=0.0)
    total = usage.total / (1024**3)
    free = usage.free / (1024**3)
    used_pct = ((usage.total - usage.free) / usage.total * 100.0) if usage.total else 0.0
    return DiskInfo(path=str(path), total_gb=total, free_gb=free, percent_used=used_pct)


def collect() -> SystemInfo:
    """Collect system information for diagnostics."""
    from visionservex import __version__
    from visionservex.config import get_settings

    settings = get_settings()
    cache_path = Path(settings.cache.cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    osname = platform.system().lower()
    machine = platform.machine().lower()
    is_apple_silicon = osname == "darwin" and machine in {"arm64", "aarch64"}

    install_path = Path(__file__).resolve().parents[1]

    return SystemInfo(
        os_name=platform.system(),
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        package_version=__version__,
        package_install_path=str(install_path),
        cache_path=str(cache_path),
        cpu=_cpu_info(),
        memory=_memory_info(),
        disk=_disk_info(cache_path if cache_path.exists() else cache_path.parent),
        is_64bit=sys.maxsize > 2**32,
        is_linux=osname == "linux",
        is_macos=osname == "darwin",
        is_windows=osname == "windows",
        is_apple_silicon=is_apple_silicon,
    )


def probe_dependencies() -> dict[str, dict[str, Any]]:
    """Probe optional dependencies relevant to VisionServeX backends."""
    items = [
        ("torch", "pip install 'visionservex[torch]'"),
        ("torchvision", "pip install 'visionservex[torch]'"),
        ("transformers", "pip install 'visionservex[hf]'"),
        ("huggingface_hub", "pip install 'visionservex[hf]'"),
        ("safetensors", "pip install 'visionservex[hf]'"),
        ("onnxruntime", "pip install 'visionservex[onnx]'"),
        ("fastapi", "pip install 'visionservex[server]'"),
        ("uvicorn", "pip install 'visionservex[server]'"),
        ("mmengine", "pip install openmim && mim install mmengine mmcv"),
        ("mmcv", "pip install openmim && mim install mmcv"),
        ("mmpose", "mim install mmpose"),
        ("mmdet", "mim install mmdet"),
        ("mmrotate", "mim install mmrotate"),
        ("mmpretrain", "mim install mmpretrain"),
        ("sam2", "pip install 'visionservex[sam2]' and follow upstream sam2 install"),
        ("cv2", "pip install opencv-python-headless"),
    ]
    out: dict[str, dict[str, Any]] = {}
    for name, hint in items:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            out[name] = {"installed": True, "version": version, "hint": ""}
        except Exception:
            out[name] = {"installed": False, "version": None, "hint": hint}
    return out


__all__ = [
    "SystemInfo",
    "CPUInfo",
    "MemoryInfo",
    "DiskInfo",
    "collect",
    "probe_dependencies",
]

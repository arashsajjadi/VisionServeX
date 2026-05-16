# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Central resource guard — prevents RAM/VRAM/CPU/disk exhaustion during tests and model loads.

Environment overrides (all default to safe values):
  VISIONSERVEX_ALLOW_HEAVY_TESTS=1          — skip all resource pre-checks for heavy tests
  VISIONSERVEX_RUN_REAL_MODEL_TESTS=1       — enable real_model marker tests
  VISIONSERVEX_RUN_GPU_TESTS=1              — enable gpu marker tests
  VISIONSERVEX_RUN_DOWNLOAD_TESTS=1         — enable download marker tests
  VISIONSERVEX_RUN_SIDECAR_TESTS=1          — enable sidecar marker tests
  VISIONSERVEX_RUN_BENCHMARK_TESTS=1        — enable benchmark marker tests
  VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1       — enable disk_heavy marker tests
  VISIONSERVEX_ALLOW_CONCURRENT_PYTEST=1    — skip concurrent-pytest lockout
  VISIONSERVEX_MAX_TEST_WORKERS=N           — max worker processes (default: 1)
  VISIONSERVEX_MIN_FREE_RAM_GB=8            — minimum free RAM required (GB)
  VISIONSERVEX_MIN_FREE_VRAM_GB=2           — minimum free VRAM required (GB)
  VISIONSERVEX_MIN_FREE_DISK_GB=10          — minimum free disk required (GB)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

# ---------------------------------------------------------------------------
# Constants / environment overrides
# ---------------------------------------------------------------------------

_ENV = os.environ.get

ALLOW_HEAVY = _ENV("VISIONSERVEX_ALLOW_HEAVY_TESTS", "0") == "1"
RUN_REAL_MODEL = _ENV("VISIONSERVEX_RUN_REAL_MODEL_TESTS", "0") == "1"
RUN_GPU = _ENV("VISIONSERVEX_RUN_GPU_TESTS", "0") == "1"
RUN_DOWNLOAD = _ENV("VISIONSERVEX_RUN_DOWNLOAD_TESTS", "0") == "1"
RUN_SIDECAR = _ENV("VISIONSERVEX_RUN_SIDECAR_TESTS", "0") == "1"
RUN_BENCHMARK = _ENV("VISIONSERVEX_RUN_BENCHMARK_TESTS", "0") == "1"
RUN_DISK_HEAVY = _ENV("VISIONSERVEX_RUN_DISK_HEAVY_TESTS", "0") == "1"
ALLOW_CONCURRENT_PYTEST = _ENV("VISIONSERVEX_ALLOW_CONCURRENT_PYTEST", "0") == "1"

try:
    MAX_TEST_WORKERS: int = int(_ENV("VISIONSERVEX_MAX_TEST_WORKERS", "1"))
except ValueError:
    MAX_TEST_WORKERS = 1

try:
    MIN_FREE_RAM_GB: float = float(_ENV("VISIONSERVEX_MIN_FREE_RAM_GB", "8"))
except ValueError:
    MIN_FREE_RAM_GB = 8.0

try:
    MIN_FREE_VRAM_GB: float = float(_ENV("VISIONSERVEX_MIN_FREE_VRAM_GB", "2"))
except ValueError:
    MIN_FREE_VRAM_GB = 2.0

try:
    MIN_FREE_DISK_GB: float = float(_ENV("VISIONSERVEX_MIN_FREE_DISK_GB", "10"))
except ValueError:
    MIN_FREE_DISK_GB = 10.0

# Maximum allowed RAM usage % before refusing tests
RAM_MAX_USAGE_PCT = 80.0
# Default pytest lockfile path
PYTEST_LOCK_PATH = Path(os.environ.get("VISIONSERVEX_PYTEST_LOCK", "/tmp/visionservex_pytest.lock"))
# Repo root (used to scope process kills)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SystemMemoryState:
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_gb: float = 0.0
    used_pct: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_pct: float = 0.0


@dataclass
class GpuMemoryState:
    cuda_available: bool = False
    device_name: str = ""
    total_mb: float = 0.0
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_gb: float = 0.0
    free_gb: float = 0.0


@dataclass
class DiskState:
    path: str = "/"
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    used_pct: float = 0.0


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: str
    memory_mb: float
    is_pytest: bool = False
    is_visionservex: bool = False


@dataclass
class ResourceBudget:
    """Snapshot of all monitored resources."""

    memory: SystemMemoryState = field(default_factory=SystemMemoryState)
    gpu: GpuMemoryState = field(default_factory=GpuMemoryState)
    disk: DiskState = field(default_factory=DiskState)
    cpu_pct: float = 0.0
    cpu_count: int = 1
    pytest_processes: list[ProcessInfo] = field(default_factory=list)
    visionservex_workers: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "ram_available_gb": round(self.memory.available_gb, 2),
            "ram_used_pct": round(self.memory.used_pct, 1),
            "swap_used_pct": round(self.memory.swap_pct, 1),
            "vram_free_gb": round(self.gpu.free_gb, 2),
            "disk_free_gb": round(self.disk.free_gb, 2),
            "cpu_pct": round(self.cpu_pct, 1),
            "cpu_count": self.cpu_count,
            "other_pytest_pids": [p.pid for p in self.pytest_processes if p.pid != os.getpid()],
            "visionservex_workers": self.visionservex_workers,
        }


# ---------------------------------------------------------------------------
# State readers
# ---------------------------------------------------------------------------


def get_system_memory_state() -> SystemMemoryState:
    """Read current system RAM and swap usage."""
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return SystemMemoryState(
        total_gb=vm.total / 1024**3,
        available_gb=vm.available / 1024**3,
        used_gb=vm.used / 1024**3,
        used_pct=vm.percent,
        swap_total_gb=sw.total / 1024**3,
        swap_used_gb=sw.used / 1024**3,
        swap_pct=sw.percent,
    )


def get_gpu_memory_state() -> GpuMemoryState:
    """Read current GPU VRAM. Returns zeroed state when CUDA unavailable."""
    state = GpuMemoryState()
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return state
        state.cuda_available = True
        state.device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        state.total_mb = props.total_memory / 1024**2
        state.total_gb = state.total_mb / 1024
        state.allocated_mb = torch.cuda.memory_allocated() / 1024**2
        state.reserved_mb = torch.cuda.memory_reserved() / 1024**2
        state.free_mb = state.total_mb - state.reserved_mb
        state.free_gb = state.free_mb / 1024
    except Exception:
        pass
    return state


def get_disk_state(path: str | Path = "/") -> DiskState:
    """Read disk usage for the given path."""
    p = str(path)
    try:
        usage = psutil.disk_usage(p)
        return DiskState(
            path=p,
            total_gb=usage.total / 1024**3,
            used_gb=usage.used / 1024**3,
            free_gb=usage.free / 1024**3,
            used_pct=usage.percent,
        )
    except Exception:
        return DiskState(path=p)


def get_process_tree() -> list[ProcessInfo]:
    """Return information about running Python/pytest processes."""
    results: list[ProcessInfo] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            cmdline_parts = info.get("cmdline") or []
            cmdline = " ".join(str(c) for c in cmdline_parts).lower()
            if not ("python" in name or "pytest" in name or "python" in cmdline):
                continue
            mem_mb = 0.0
            mem_info = info.get("memory_info")
            if mem_info:
                mem_mb = mem_info.rss / 1024**2
            is_pytest = "pytest" in cmdline or "-m pytest" in cmdline
            is_vsrv = "visionservex" in cmdline or str(_REPO_ROOT).lower() in cmdline
            results.append(
                ProcessInfo(
                    pid=info["pid"],
                    name=name,
                    cmdline=cmdline[:200],
                    memory_mb=round(mem_mb, 1),
                    is_pytest=is_pytest,
                    is_visionservex=is_vsrv,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return results


def _collect_resource_budget(disk_path: str | Path = ".") -> ResourceBudget:
    mem = get_system_memory_state()
    gpu = get_gpu_memory_state()
    disk = get_disk_state(disk_path)
    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
    except Exception:
        cpu_pct = 0.0
    cpu_count = psutil.cpu_count(logical=True) or 1
    procs = get_process_tree()
    pytest_procs = [p for p in procs if p.is_pytest]
    vsrv_workers = sum(1 for p in procs if p.is_visionservex)
    return ResourceBudget(
        memory=mem,
        gpu=gpu,
        disk=disk,
        cpu_pct=cpu_pct,
        cpu_count=cpu_count,
        pytest_processes=pytest_procs,
        visionservex_workers=vsrv_workers,
    )


# ---------------------------------------------------------------------------
# Guard assertions
# ---------------------------------------------------------------------------


class ResourceGuardError(RuntimeError):
    """Raised when a resource safety threshold is violated."""


def refuse_if_other_pytest_running() -> None:
    """Raise if another pytest process (different PID) is already running."""
    if ALLOW_CONCURRENT_PYTEST:
        return
    procs = get_process_tree()
    my_pid = os.getpid()
    others = [p for p in procs if p.is_pytest and p.pid != my_pid]
    if others:
        pids = ", ".join(str(p.pid) for p in others)
        raise ResourceGuardError(
            f"Another VisionServeX test run is already active (PID {pids}). "
            "Use 'visionservex dev kill-tests' if this is stale, "
            "or set VISIONSERVEX_ALLOW_CONCURRENT_PYTEST=1 to override."
        )


def refuse_if_ram_above_threshold(max_pct: float = RAM_MAX_USAGE_PCT) -> None:
    """Raise if RAM usage exceeds max_pct percent."""
    mem = get_system_memory_state()
    if mem.used_pct > max_pct:
        raise ResourceGuardError(
            f"System RAM usage is {mem.used_pct:.1f}% (limit {max_pct:.0f}%). "
            f"Only {mem.available_gb:.1f} GB free of {mem.total_gb:.1f} GB total. "
            "Free memory before running heavy tests. "
            "Set VISIONSERVEX_MIN_FREE_RAM_GB to adjust."
        )


def refuse_if_vram_above_threshold(required_vram_gb: float = 0.0) -> None:
    """Raise if free VRAM is insufficient (includes desktop reserve)."""
    gpu = get_gpu_memory_state()
    if not gpu.cuda_available:
        return
    needed = required_vram_gb + MIN_FREE_VRAM_GB
    if gpu.free_gb < needed:
        raise ResourceGuardError(
            f"Insufficient free VRAM: {gpu.free_gb:.1f} GB free on {gpu.device_name}, "
            f"need {needed:.1f} GB ({required_vram_gb:.1f} model + {MIN_FREE_VRAM_GB:.1f} reserve). "
            "Unload other GPU applications first. "
            "Set VISIONSERVEX_MIN_FREE_VRAM_GB to adjust the reserve."
        )


def refuse_if_disk_free_below_threshold(path: str | Path = ".") -> None:
    """Raise if disk free space is below MIN_FREE_DISK_GB."""
    disk = get_disk_state(path)
    if disk.free_gb < MIN_FREE_DISK_GB:
        raise ResourceGuardError(
            f"Insufficient disk space: {disk.free_gb:.1f} GB free at '{path}', "
            f"need at least {MIN_FREE_DISK_GB:.1f} GB. "
            "Clean disk before running tests. "
            "Set VISIONSERVEX_MIN_FREE_DISK_GB to adjust."
        )


def assert_safe_to_start_test(disk_path: str | Path = ".") -> ResourceBudget:
    """Check all resource thresholds before starting any test.

    Returns the current ResourceBudget so callers can log it.
    """
    if ALLOW_HEAVY:
        return _collect_resource_budget(disk_path)
    refuse_if_other_pytest_running()
    refuse_if_ram_above_threshold()
    refuse_if_disk_free_below_threshold(disk_path)
    return _collect_resource_budget(disk_path)


def assert_safe_to_start_model_load(required_vram_gb: float = 2.0) -> ResourceBudget:
    """Check GPU and RAM resources before loading a real model."""
    if ALLOW_HEAVY:
        return _collect_resource_budget()
    refuse_if_ram_above_threshold()
    refuse_if_vram_above_threshold(required_vram_gb)
    refuse_if_disk_free_below_threshold()
    return _collect_resource_budget()


def assert_safe_to_start_benchmark() -> ResourceBudget:
    """Check resources before starting a benchmark (RAM + VRAM + disk)."""
    if ALLOW_HEAVY:
        return _collect_resource_budget()
    refuse_if_ram_above_threshold()
    refuse_if_vram_above_threshold()
    refuse_if_disk_free_below_threshold()
    return _collect_resource_budget()


def enforce_resource_budget() -> ResourceBudget:
    """Check all resources and return current budget (non-raising summary)."""
    return _collect_resource_budget()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup_after_test() -> None:
    """Best-effort cleanup after each test: GC + CUDA cache flush."""
    for _ in range(3):
        gc.collect()
    try:
        import contextlib

        import torch  # type: ignore

        if torch.cuda.is_available():
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


def kill_only_own_child_processes() -> list[int]:
    """Kill only child processes of the current process. Never kills system processes."""
    killed: list[int] = []
    my_proc = psutil.Process(os.getpid())
    try:
        children = my_proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return killed
    for child in children:
        try:
            child.terminate()
            killed.append(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Give them a moment, then SIGKILL stragglers
    time.sleep(0.5)
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


# ---------------------------------------------------------------------------
# Pytest lockfile
# ---------------------------------------------------------------------------


def _lock_data() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "command": " ".join(sys.argv[:5]),
        "started": time.time(),
        "repo": str(_REPO_ROOT),
    }


def _is_stale_lock(data: dict[str, Any]) -> bool:
    pid = data.get("pid", -1)
    if not isinstance(pid, int):
        return True
    try:
        return not psutil.pid_exists(pid)
    except Exception:
        return True


def acquire_pytest_lock() -> None:
    """Create the pytest lockfile for this process. Raises if another is active."""
    if ALLOW_CONCURRENT_PYTEST:
        return
    if PYTEST_LOCK_PATH.exists():
        try:
            data = json.loads(PYTEST_LOCK_PATH.read_text())
            if not _is_stale_lock(data):
                pid = data.get("pid", "?")
                cmd = data.get("command", "?")
                raise ResourceGuardError(
                    f"Another VisionServeX test run is already active (PID {pid}, cmd: {cmd}). "
                    "Use 'visionservex dev kill-tests' if this is stale, "
                    "or set VISIONSERVEX_ALLOW_CONCURRENT_PYTEST=1 to override."
                )
        except (json.JSONDecodeError, OSError):
            pass  # Corrupt/unreadable lock — treat as stale
    import contextlib

    with contextlib.suppress(OSError):
        PYTEST_LOCK_PATH.write_text(json.dumps(_lock_data()))


def release_pytest_lock() -> None:
    """Remove the pytest lockfile if it belongs to this process."""
    if not PYTEST_LOCK_PATH.exists():
        return
    try:
        data = json.loads(PYTEST_LOCK_PATH.read_text())
        if data.get("pid") == os.getpid():
            PYTEST_LOCK_PATH.unlink(missing_ok=True)
    except (json.JSONDecodeError, OSError):
        pass


def kill_visionservex_tests() -> list[int]:
    """Kill pytest processes running inside the VisionServeX repo. Repo-scoped only."""
    repo_str = str(_REPO_ROOT).lower()
    killed: list[int] = []
    my_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = proc.info
            if info["pid"] == my_pid:
                continue
            cmdline = " ".join(str(c) for c in (info.get("cmdline") or [])).lower()
            name = (info.get("name") or "").lower()
            if ("pytest" in cmdline or "pytest" in name) and repo_str in cmdline:
                proc.terminate()
                killed.append(info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Stale lock cleanup
    if PYTEST_LOCK_PATH.exists():
        try:
            data = json.loads(PYTEST_LOCK_PATH.read_text())
            if _is_stale_lock(data) or data.get("pid") in killed:
                PYTEST_LOCK_PATH.unlink(missing_ok=True)
        except (json.JSONDecodeError, OSError):
            PYTEST_LOCK_PATH.unlink(missing_ok=True)
    return killed


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------


def print_resource_report(label: str = "") -> None:
    """Print a concise resource usage summary to stdout."""
    budget = _collect_resource_budget()
    tag = f"[{label}] " if label else ""
    print(
        f"\n{tag}=== VisionServeX Resource Report ===\n"
        f"  RAM  : {budget.memory.used_gb:.1f}/{budget.memory.total_gb:.1f} GB used "
        f"({budget.memory.used_pct:.1f}%) | "
        f"{budget.memory.available_gb:.1f} GB free\n"
        f"  Swap : {budget.memory.swap_used_gb:.1f}/{budget.memory.swap_total_gb:.1f} GB "
        f"({budget.memory.swap_pct:.1f}%)\n"
        f"  CPU  : {budget.cpu_pct:.1f}% | {budget.cpu_count} logical cores\n"
        f"  Disk : {budget.disk.free_gb:.1f} GB free at {budget.disk.path!r}\n"
    )
    if budget.gpu.cuda_available:
        print(
            f"  VRAM : {budget.gpu.free_gb:.2f}/{budget.gpu.total_gb:.2f} GB free "
            f"({budget.gpu.device_name})\n"
            f"         allocated {budget.gpu.allocated_mb:.1f} MB | "
            f"reserved {budget.gpu.reserved_mb:.1f} MB\n"
        )
    else:
        print("  VRAM : CUDA not available\n")
    other_pytest = [p for p in budget.pytest_processes if p.pid != os.getpid()]
    if other_pytest:
        pids = ", ".join(str(p.pid) for p in other_pytest)
        print(f"  WARN : Other pytest processes running: PID {pids}\n")
    print(f"  Min free RAM required  : {MIN_FREE_RAM_GB:.1f} GB")
    print(f"  Min free VRAM required : {MIN_FREE_VRAM_GB:.1f} GB")
    print(f"  Min free disk required : {MIN_FREE_DISK_GB:.1f} GB")
    print("=" * 40)


__all__ = [
    "ALLOW_CONCURRENT_PYTEST",
    "ALLOW_HEAVY",
    "MAX_TEST_WORKERS",
    "MIN_FREE_DISK_GB",
    "MIN_FREE_RAM_GB",
    "MIN_FREE_VRAM_GB",
    "PYTEST_LOCK_PATH",
    "RAM_MAX_USAGE_PCT",
    "RUN_BENCHMARK",
    "RUN_DISK_HEAVY",
    "RUN_DOWNLOAD",
    "RUN_GPU",
    "RUN_REAL_MODEL",
    "RUN_SIDECAR",
    "DiskState",
    "GpuMemoryState",
    "ProcessInfo",
    "ResourceBudget",
    "ResourceGuardError",
    "SystemMemoryState",
    "acquire_pytest_lock",
    "assert_safe_to_start_benchmark",
    "assert_safe_to_start_model_load",
    "assert_safe_to_start_test",
    "cleanup_after_test",
    "enforce_resource_budget",
    "get_disk_state",
    "get_gpu_memory_state",
    "get_process_tree",
    "get_system_memory_state",
    "kill_only_own_child_processes",
    "kill_visionservex_tests",
    "print_resource_report",
    "refuse_if_disk_free_below_threshold",
    "refuse_if_other_pytest_running",
    "refuse_if_ram_above_threshold",
    "refuse_if_vram_above_threshold",
    "release_pytest_lock",
]

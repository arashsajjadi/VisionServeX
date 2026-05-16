#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Diagnose current system resources and identify potential problems.

Usage: python scripts/diagnose_resources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from visionservex.runtime.resource_guard import (
        MIN_FREE_DISK_GB,
        MIN_FREE_RAM_GB,
        MIN_FREE_VRAM_GB,
        RAM_MAX_USAGE_PCT,
        get_disk_state,
        get_gpu_memory_state,
        get_process_tree,
        get_system_memory_state,
        print_resource_report,
    )

    print_resource_report("diagnose")

    warnings: list[str] = []

    mem = get_system_memory_state()
    if mem.used_pct > RAM_MAX_USAGE_PCT:
        warnings.append(
            f"RAM usage {mem.used_pct:.1f}% exceeds limit {RAM_MAX_USAGE_PCT:.0f}%. "
            f"Free at least {mem.used_gb - (mem.total_gb * RAM_MAX_USAGE_PCT / 100):.1f} GB."
        )
    if mem.available_gb < MIN_FREE_RAM_GB:
        warnings.append(
            f"Free RAM {mem.available_gb:.1f} GB below minimum {MIN_FREE_RAM_GB:.1f} GB."
        )

    gpu = get_gpu_memory_state()
    if gpu.cuda_available and gpu.free_gb < MIN_FREE_VRAM_GB:
        warnings.append(f"Free VRAM {gpu.free_gb:.2f} GB below minimum {MIN_FREE_VRAM_GB:.1f} GB.")

    disk = get_disk_state(ROOT)
    if disk.free_gb < MIN_FREE_DISK_GB:
        warnings.append(f"Free disk {disk.free_gb:.1f} GB below minimum {MIN_FREE_DISK_GB:.1f} GB.")

    procs = get_process_tree()
    import os

    other_pytest = [p for p in procs if p.is_pytest and p.pid != os.getpid()]
    if other_pytest:
        pids = [p.pid for p in other_pytest]
        warnings.append(
            f"Other pytest processes running: {pids}. "
            "Run 'python scripts/kill_visionservex_tests.py' to clean up."
        )

    if warnings:
        print("\n[WARNINGS]")
        for w in warnings:
            print(f"  !! {w}")
        return 1
    else:
        print("\n[OK] All resource checks passed. Safe to run tests.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

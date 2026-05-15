# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 01 — check that VisionServeX can run on your machine.

This is exactly what `visionservex doctor` shows, accessed from Python.

    python examples/beginner/01_check_device.py
"""

from __future__ import annotations

import json

from visionservex.runtime.device import available_devices, best_device
from visionservex.runtime.recommendations import first_beginner_pick
from visionservex.utils.system import collect, probe_dependencies


def main() -> None:
    print("=== System ===")
    sys_info = collect()
    print(json.dumps(sys_info.to_dict()["os"], indent=2))
    mem = sys_info.memory.to_dict()
    print(f"Memory: {mem['total_gb']} GB total, {mem['available_gb']} GB available")
    disk = sys_info.disk.to_dict()
    print(f"Cache disk: {disk['free_gb']} GB free of {disk['total_gb']} GB")

    print("\n=== Devices ===")
    for d in available_devices():
        info = d.to_dict()
        ok = "yes" if info["available"] else "no "
        vram = ""
        if info.get("total_vram_gb"):
            vram = f" — {info['total_vram_gb']} GB VRAM"
        print(f"  [{ok}] {info['name']:9s} {info['detail']}{vram}")

    best = best_device()
    print(f"\nBest available device: {best.name}")

    print("\n=== Dependencies (only matter for real backends) ===")
    for name, info in probe_dependencies().items():
        mark = "ok" if info["installed"] else "no"
        print(f"  [{mark}] {name:18s} {info.get('version') or '-'}")

    pick = first_beginner_pick(task="detect")
    print(f"\nFirst recommended detection model: {pick.id if pick else '(none)'}")
    print("\nNext step:")
    print("  visionservex pull mock-detect")
    print("  visionservex predict mock-detect examples/images/simple_shapes.jpg --save out.jpg")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Kill VisionServeX pytest processes running inside this repo only.

Usage: python scripts/kill_visionservex_tests.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from visionservex.runtime.resource_guard import kill_visionservex_tests

    killed = kill_visionservex_tests()
    if killed:
        print(f"Killed PIDs: {killed}")
    else:
        print("No active VisionServeX test processes found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

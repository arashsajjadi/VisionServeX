#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick safe test runner — never runs heavy/model/GPU/download tests.

Usage: python scripts/test_quick_safe.py [extra pytest args...]
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

QUICK_MARKERS = (
    "not slow and not real_model and not gpu and not network "
    "and not sidecar and not release and not benchmark "
    "and not memory and not disk_heavy and not download"
)


def main() -> int:
    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        acquire_pytest_lock,
        cleanup_after_test,
        print_resource_report,
        refuse_if_other_pytest_running,
        release_pytest_lock,
    )

    print_resource_report("before-quick")
    try:
        refuse_if_other_pytest_running()
    except ResourceGuardError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    acquire_pytest_lock()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        QUICK_MARKERS,
        "--tb=short",
        "--maxfail=1",
        "--durations=20",
        "--no-header",
        "tests/",
        *sys.argv[1:],
    ]

    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode
    finally:
        release_pytest_lock()
        cleanup_after_test()
        print_resource_report("after-quick")


if __name__ == "__main__":
    sys.exit(main())

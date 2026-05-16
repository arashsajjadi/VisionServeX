#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Targeted safe test runner — runs specific file/keyword with resource guard.

Usage: python scripts/test_targeted_safe.py tests/test_foo.py [extra pytest args...]
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
    if len(sys.argv) < 2:
        print("Usage: test_targeted_safe.py <test-path-or-keyword> [extra-args...]")
        return 1

    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        acquire_pytest_lock,
        assert_safe_to_start_test,
        cleanup_after_test,
        print_resource_report,
        refuse_if_other_pytest_running,
        release_pytest_lock,
    )

    try:
        refuse_if_other_pytest_running()
        assert_safe_to_start_test()
    except ResourceGuardError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print_resource_report("before-targeted")
    acquire_pytest_lock()
    target = sys.argv[1]
    extra = sys.argv[2:]
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
        target,
        *extra,
    ]

    try:
        result = subprocess.run(cmd, cwd=str(ROOT))
        return result.returncode
    finally:
        release_pytest_lock()
        cleanup_after_test()
        print_resource_report("after-targeted")


if __name__ == "__main__":
    sys.exit(main())

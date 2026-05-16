#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Full release test runner — checks all resources before starting.

Usage: python scripts/test_release_safe.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from visionservex.runtime.resource_guard import (
        ResourceGuardError,
        acquire_pytest_lock,
        assert_safe_to_start_test,
        cleanup_after_test,
        print_resource_report,
        release_pytest_lock,
    )

    try:
        assert_safe_to_start_test()
    except ResourceGuardError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print_resource_report("before-full-release")
    acquire_pytest_lock()
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "--maxfail=5",
        "--durations=30",
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
        print_resource_report("after-full-release")


if __name__ == "__main__":
    sys.exit(main())

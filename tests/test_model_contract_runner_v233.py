# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: contract test runner module surface."""

from __future__ import annotations


def test_contract_runner_imports() -> None:
    from visionservex.runtime.contract_runner import (
        run_contract_matrix,
    )

    assert callable(run_contract_matrix)


def test_contract_cli_help() -> None:
    import subprocess
    import sys
    from pathlib import Path

    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "models", "contract-test", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).parent.parent),
    )
    assert proc.returncode == 0
    import re

    # Strip ANSI escape codes before checking
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    assert "--include" in clean_output or "include" in clean_output
    assert "--device" in clean_output or "device" in clean_output

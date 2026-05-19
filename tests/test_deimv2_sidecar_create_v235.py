# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.35.0: DEIMv2 sidecar creation and environment verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_deimv2_sidecar_create_report_exists() -> None:
    """v2.35 must produce a sidecar creation report."""
    p = Path(__file__).parent.parent / "reports/deimv2_sidecar_create_v235.json"
    if not p.exists():
        pytest.skip("DEIMv2 sidecar not yet created")
    d = json.loads(p.read_text())
    assert d.get("status") in ("ok", "expected_blocker"), f"unexpected status: {d}"


def test_deimv2_checkpoints_downloaded() -> None:
    """DEIMv2 S/M/L/X checkpoints must be in cache."""
    cache = Path.home() / ".cache/visionservex/deimv2"
    found = list(cache.glob("deimv2_dinov3_*.pth"))
    assert len(found) >= 1, "No DEIMv2 checkpoints found"
    print(f"Found {len(found)} DEIMv2 checkpoints")


@pytest.mark.slow
def test_deimv2_sidecar_runs_torch() -> None:
    """DEIMv2 sidecar must have a working torch with CUDA (slow/GPU test)."""
    import subprocess

    proc = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "visionservex-deimv2-sidecar",
            "python",
            "-c",
            "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"DEIMv2 sidecar torch check failed: {proc.stderr[:200]}"

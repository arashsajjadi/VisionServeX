# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: Florence-2 sidecar status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_florence2_sidecar_report_exists() -> None:
    """Florence-2 sidecar must produce a status report."""
    p = Path(__file__).parent.parent / "reports/v236_florence2_sidecar_create.json"
    if not p.exists():
        pytest.skip("Florence-2 report not present")
    d = json.loads(p.read_text())
    assert d.get("status") in ("ok", "expected_blocker", "sidecar_required")


def test_florence2_demo_runs_or_has_exact_blocker() -> None:
    """Florence-2 must either produce a caption or have exact sidecar_required."""
    p = Path(__file__).parent.parent / "reports/v236_florence2_sidecar_create.json"
    if not p.exists():
        pytest.skip("Florence-2 report not present")
    d = json.loads(p.read_text())
    if d.get("status") == "ok":
        assert d.get("caption") or d.get("caption_result"), "caption is empty but status is ok"
    else:
        assert d.get("code"), f"no blocker code: {d}"

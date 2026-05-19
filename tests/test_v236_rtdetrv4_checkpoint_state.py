# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: RT-DETRv4 checkpoint state documentation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rtdetrv4_checkpoint_state_exists() -> None:
    p = Path(__file__).parent.parent / "reports/v236_rtdetrv4_checkpoint_state.json"
    if not p.exists():
        pytest.skip("RT-DETRv4 checkpoint state not present")
    d = json.loads(p.read_text())
    assert "rows" in d or "summary" in d


def test_rtdetrv4_all_variants_have_manual_checkpoint_info() -> None:
    p = Path(__file__).parent.parent / "reports/v236_rtdetrv4_checkpoint_state.json"
    if not p.exists():
        pytest.skip("RT-DETRv4 checkpoint state not present")
    d = json.loads(p.read_text())
    for r in d.get("rows", []):
        mid = r.get("model_id", "")
        if "rtdetrv4" in mid:
            assert r.get("final_state") in (
                "manual_checkpoint_required",
                "benchmarked",
                "contract_passed",
            ), f"{mid}: unexpected state {r.get('final_state')}"
            if r.get("final_state") == "manual_checkpoint_required":
                assert r.get("blocker_code") == "MANUAL_CHECKPOINT_REQUIRED", (
                    f"{mid}: expected MANUAL_CHECKPOINT_REQUIRED, got {r.get('blocker_code')}"
                )

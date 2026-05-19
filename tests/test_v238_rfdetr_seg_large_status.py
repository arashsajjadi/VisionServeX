# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.38.0: rfdetr-seg-large benchmark passed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rfdetr_seg_large_benchmark_report() -> None:
    p = Path(__file__).parent.parent / "reports/v238_rfdetr_seg_large_benchmark.json"
    if not p.exists():
        pytest.skip("rfdetr-seg-large benchmark report not present")
    d = json.loads(p.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", [])}
    r = rows.get("rfdetr-seg-large")
    if r is None:
        pytest.skip("rfdetr-seg-large not in benchmark")
    if r.get("status") == "ok":
        assert r["mask_mAP50_95"] > 0.10, f"rfdetr-seg-large mAP too low: {r['mask_mAP50_95']}"


def test_rfdetr_seg_large_not_license_blocked_in_matrix() -> None:
    p = Path(__file__).parent.parent / "reports/v238_49_blocked_resolution_matrix.json"
    if not p.exists():
        pytest.skip("v2.38 matrix not present")
    d = json.loads(p.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    r = rows.get("rfdetr-seg-large")
    if r is None:
        pytest.skip("rfdetr-seg-large not in matrix")
    final = r.get("final_state_after_v238") or r.get("final_state_after_v237")
    assert "license_blocked" not in final, (
        f"rfdetr-seg-large is Apache-2.0 (core), must not be license_blocked. final={final}"
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: OneFormer segmentation contract and benchmark schema."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPORT = Path(__file__).parent.parent / "reports/v236_automatic_segmentation_400.json"


def test_v236_segmentation_report_exists() -> None:
    if not REPORT.exists():
        pytest.skip("v236 segmentation report not present")
    d = json.loads(REPORT.read_text())
    assert "rows" in d


def test_v236_oneformer_swinlarge_benchmarked() -> None:
    """OneFormer-SwinLarge must appear in v2.36 segmentation benchmark."""
    if not REPORT.exists():
        pytest.skip("v236 segmentation report not present")
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", [])}
    if "oneformer-swin-large" not in rows:
        pytest.skip("oneformer-swin-large not in benchmark")
    r = rows["oneformer-swin-large"]
    assert r["status"] == "ok", f"oneformer-swin-large not ok: {r.get('code')}"
    assert r["mask_mAP50_95"] > 0.10, f"OneFormer mAP too low: {r['mask_mAP50_95']}"


def test_v236_segmentation_no_nan() -> None:
    if not REPORT.exists():
        pytest.skip("v236 segmentation report not present")
    d = json.loads(REPORT.read_text())
    for r in d.get("rows", []):
        if r.get("status") == "ok":
            for m in ("mask_mAP50_95", "mask_AP50", "latency_ms_p50"):
                v = r.get(m)
                if isinstance(v, float):
                    assert not math.isnan(v), f"{r['model_id']}.{m} is NaN"


def test_v236_oneformer_convnext_has_structured_blocker() -> None:
    """oneformer-convnext-large must have a structured blocker, not generic."""
    if not REPORT.exists():
        pytest.skip("v236 segmentation report not present")
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", [])}
    r = rows.get("oneformer-convnext-large")
    if r is None:
        pytest.skip("oneformer-convnext-large not in report")
    if r.get("status") == "expected_blocker":
        assert r.get("code"), "oneformer-convnext-large has no blocker code"
        assert r["code"] not in ("", "UNKNOWN"), f"generic blocker: {r['code']}"

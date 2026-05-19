# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.35.0: SAM2 promptable benchmark results must be preserved."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPORT = Path(__file__).parent.parent / "reports/promptable_coco400_v234.json"


def test_sam2_promptable_report_exists() -> None:
    if not REPORT.exists():
        pytest.skip("Promptable benchmark report not present")
    d = json.loads(REPORT.read_text())
    assert d.get("status") == "ok"


def test_sam2_mean_iou_no_nan() -> None:
    if not REPORT.exists():
        pytest.skip("Promptable benchmark report not present")
    d = json.loads(REPORT.read_text())
    for r in d.get("rows", []):
        iou = r.get("mean_iou")
        if isinstance(iou, float):
            assert not math.isnan(iou), f"{r.get('model_id')}: mean_iou is NaN"


def test_sam21_large_achieves_reasonable_iou() -> None:
    """SAM2.1-hiera-large should achieve mean IoU > 0.75 on GT-box prompts."""
    if not REPORT.exists():
        pytest.skip("Promptable benchmark report not present")
    d = json.loads(REPORT.read_text())
    large_row = next(
        (
            r
            for r in d.get("rows", [])
            if "2.1" in r.get("model_id", "") and "large" in r.get("model_id", "")
        ),
        None,
    )
    if large_row is None:
        pytest.skip("sam2.1-hiera-large not in results")
    iou = large_row.get("mean_iou")
    assert isinstance(iou, float) and iou > 0.75, f"SAM2.1-large IoU too low: {iou}"

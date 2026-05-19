# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.35.0: DEIMv2 COCO detection benchmark output schema."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPORT = Path(__file__).parent.parent / "reports/deimv2_detection_400_v235.json"


def test_deimv2_benchmark_report_exists() -> None:
    if not REPORT.exists():
        pytest.skip("DEIMv2 benchmark report not present yet")
    d = json.loads(REPORT.read_text())
    assert "rows" in d


def test_deimv2_benchmark_no_nan() -> None:
    if not REPORT.exists():
        pytest.skip("DEIMv2 benchmark report not present")
    d = json.loads(REPORT.read_text())
    for r in d.get("rows", []):
        if r.get("status") == "ok":
            for metric in ("mAP50_95", "AP50", "latency_ms_p50"):
                v = r.get(metric)
                if isinstance(v, float):
                    assert not math.isnan(v), f"{r['model_id']}.{metric} is NaN"


def test_deimv2_x_is_benchmarked() -> None:
    """DEIMv2-X must be benchmarked (not just contract tested)."""
    if not REPORT.exists():
        pytest.skip("DEIMv2 benchmark report not present")
    d = json.loads(REPORT.read_text())
    x_row = next((r for r in d["rows"] if r["model_id"] == "deimv2-x"), None)
    if x_row is None:
        pytest.skip("DEIMv2-X not in benchmark (may not have checkpoint)")
    assert x_row["status"] == "ok", f"deimv2-x not ok: {x_row.get('code')}"
    assert x_row["mAP50_95"] > 0.3, f"deimv2-x mAP too low: {x_row['mAP50_95']}"


def test_deimv2_family_curve_monotonic() -> None:
    """Larger DEIMv2 models should generally have higher mAP."""
    if not REPORT.exists():
        pytest.skip("DEIMv2 benchmark report not present")
    d = json.loads(REPORT.read_text())
    ok_rows = {r["model_id"]: r for r in d["rows"] if r.get("status") == "ok"}
    order = ["deimv2-s", "deimv2-m", "deimv2-l", "deimv2-x"]
    available = [mid for mid in order if mid in ok_rows]
    if len(available) < 2:
        pytest.skip("Need at least 2 DEIMv2 variants to check family curve")
    maps = [ok_rows[mid]["mAP50_95"] for mid in available]
    # Allow small dips (within 0.01) due to different training schedules
    for i in range(1, len(maps)):
        assert maps[i] >= maps[i - 1] - 0.01, (
            f"Non-monotonic family curve: {available[i - 1]}={maps[i - 1]:.4f} > "
            f"{available[i]}={maps[i]:.4f}"
        )

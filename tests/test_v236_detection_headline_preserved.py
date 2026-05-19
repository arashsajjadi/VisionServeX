# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: detection headline from v2.35 must be preserved."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_libreyolo_dfine_x_headline_in_leaderboard() -> None:
    """libreyolo-dfine-x = 0.5030 must remain the detection leader."""
    report = Path(__file__).parent.parent / "reports/libreyolo_detection_400_v235.json"
    if not report.exists():
        pytest.skip("LibreYOLO detection report not present")
    d = json.loads(report.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", []) if r.get("status") == "ok"}
    assert "libreyolo-dfine-x" in rows, "libreyolo-dfine-x not in leaderboard"
    mAP = rows["libreyolo-dfine-x"]["mAP50_95"]
    assert mAP > 0.49, f"libreyolo-dfine-x mAP regressed: {mAP}"


def test_libreyolo_dfine_x_beats_yolo26x() -> None:
    """The headline claim: libreyolo-dfine-x > yolo26x.pt = 0.4894."""
    report = Path(__file__).parent.parent / "reports/libreyolo_detection_400_v235.json"
    yolo26x_mAP = 0.4894  # from v2.27 benchmark
    if not report.exists():
        pytest.skip("LibreYOLO detection report not present")
    d = json.loads(report.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", []) if r.get("status") == "ok"}
    libre_dfine_x = rows.get("libreyolo-dfine-x", {}).get("mAP50_95", 0)
    assert libre_dfine_x > yolo26x_mAP, (
        f"Headline broken: libreyolo-dfine-x ({libre_dfine_x}) did not beat yolo26x ({yolo26x_mAP})"
    )


def test_deimv2_x_in_leaderboard() -> None:
    """DEIMv2-X benchmark result from v2.35 must be preserved."""
    report = Path(__file__).parent.parent / "reports/deimv2_detection_400_v235.json"
    if not report.exists():
        pytest.skip("DEIMv2 detection report not present")
    d = json.loads(report.read_text())
    rows = {r["model_id"]: r for r in d.get("rows", []) if r.get("status") == "ok"}
    assert "deimv2-x" in rows, "deimv2-x not in leaderboard"
    assert rows["deimv2-x"]["mAP50_95"] > 0.40

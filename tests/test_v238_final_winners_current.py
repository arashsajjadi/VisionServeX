# SPDX-License-Identifier: Apache-2.0
"""v2.38.0: final_winners.json must reflect current results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_final_winners_detection_current() -> None:
    p = Path(__file__).parent.parent / "notebook/99_final_report/reports/final_winners.json"
    if not p.exists():
        pytest.skip("final_winners.json not present")
    d = json.loads(p.read_text())
    # Detection winner must be libreyolo-dfine-x (not yolo26x — stale)
    assert "libreyolo-dfine-x" in d.get("detection_winner_overall", "")


def test_final_winners_segmentation_current() -> None:
    p = Path(__file__).parent.parent / "notebook/99_final_report/reports/final_winners.json"
    if not p.exists():
        pytest.skip("final_winners.json not present")
    d = json.loads(p.read_text())
    # VisionServeX seg winner must be oneformer-swin-large from v2.36
    assert "oneformer-swin-large" in d.get("auto_segmentation_winner_visionservex", "")


def test_final_winners_promptable_current() -> None:
    p = Path(__file__).parent.parent / "notebook/99_final_report/reports/final_winners.json"
    if not p.exists():
        pytest.skip("final_winners.json not present")
    d = json.loads(p.read_text())
    assert "sam2.1-hiera-large" in d.get("promptable_segmentation_winner", "")

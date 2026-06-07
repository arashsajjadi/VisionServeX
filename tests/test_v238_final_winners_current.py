# SPDX-License-Identifier: Apache-2.0
"""final_winners.json must reflect the current v3 core/external schema and must
never name a restricted (non-commercial / AGPL) model as a commercial-safe core
winner. (v3-prep: EdgeSAM S-Lab non-commercial must not be a core winner.)"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FW = Path(__file__).parent.parent / "notebook/99_final_report/reports/final_winners.json"

# Models that must NEVER appear as a commercial-safe core winner.
_RESTRICTED = {
    "edgesam",
    "fastsam-s",
    "fastsam-x",
    "yolo-world",
    "yolo11x.pt",
    "yolo26x.pt",
    "yolov8x.pt",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
    "totalsegmentator",
}


def _load():
    if not _FW.exists():
        pytest.skip("final_winners.json not present")
    return json.loads(_FW.read_text())


def test_final_winners_uses_core_external_schema() -> None:
    d = _load()
    for key in (
        "detection_core_winner",
        "auto_segmentation_core_winner",
        "promptable_segmentation_core_winner",
    ):
        assert key in d, f"final_winners missing v3 schema key {key}"


def test_detection_headline_names_real_core_winner() -> None:
    d = _load()
    # The real 400-image detection core winner (libreyolo-dfine-x) is recorded in
    # the headline; the computed core winner must be a permissive core model.
    assert "libreyolo-dfine-x" in d.get("detection_headline_core", "")
    assert d.get("detection_core_winner") not in _RESTRICTED


def test_no_restricted_model_is_a_core_winner() -> None:
    d = _load()
    for key in (
        "detection_core_winner",
        "auto_segmentation_core_winner",
        "promptable_segmentation_core_winner",
    ):
        assert d.get(key) not in _RESTRICTED, (
            f"{key}={d.get(key)!r} is a restricted/non-commercial model — "
            "must not be a commercial-safe core winner"
        )


def test_promptable_core_winner_is_commercial_safe() -> None:
    d = _load()
    # EdgeSAM (S-Lab non-commercial) was wrongly the core promptable winner; the
    # commercial-safe winner must be a permissive SAM-family model.
    winner = d.get("promptable_segmentation_core_winner", "")
    assert winner != "edgesam"
    assert winner in {
        "efficientsam",
        "mobilesam",
        "hq-sam",
        "sam2.1-hiera-large",
        "sam2-hiera-large",
        "sam-vit-huge",
        "medsam",
        "no_benchmark_data",
    }, f"unexpected promptable core winner: {winner!r}"

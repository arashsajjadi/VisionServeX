# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: yolo*-seg.pt and rfdetr-seg-nano/small must never be
``expected_blocker`` when the segmentation leaderboard says
``benchmark_passed``."""

from __future__ import annotations

import json
from pathlib import Path

LEDGER = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.json"
)

SEGMENTATION_BENCHMARK_IDS = {
    "yolo26x-seg.pt",
    "yolo11x-seg.pt",
    "yolov8x-seg.pt",
    "yolo11l-seg.pt",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
}


def test_ledger_exists() -> None:
    assert LEDGER.exists(), f"{LEDGER} missing — run reconcile-model-states first"


def test_segmentation_benchmark_rows_not_expected_blocker() -> None:
    rows = {r["model_id"]: r for r in json.loads(LEDGER.read_text())["rows"]}
    for mid in SEGMENTATION_BENCHMARK_IDS:
        if mid not in rows:
            continue
        fs = rows[mid].get("final_state", "")
        assert fs not in (
            "expected_blocker",
            "stub",
            "blocked",
            "",
        ), f"{mid}: stale segmentation final_state {fs!r}"


def test_yolo26x_seg_must_not_be_expected_blocker() -> None:
    rows = {r["model_id"]: r for r in json.loads(LEDGER.read_text())["rows"]}
    if "yolo26x-seg.pt" in rows:
        assert rows["yolo26x-seg.pt"]["final_state"] == "benchmark_passed", (
            "yolo26x-seg.pt is the segmentation winner — must be benchmark_passed, "
            f"not {rows['yolo26x-seg.pt']['final_state']!r}"
        )


def test_rfdetr_seg_nano_small_must_not_be_expected_blocker() -> None:
    rows = {r["model_id"]: r for r in json.loads(LEDGER.read_text())["rows"]}
    for mid in ("rfdetr-seg-nano", "rfdetr-seg-small"):
        if mid not in rows:
            continue
        assert rows[mid]["final_state"] == "benchmark_passed", (
            f"{mid}: must be benchmark_passed when the segmentation leaderboard contains it"
        )

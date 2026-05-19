# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: yolo*-seg.pt and rfdetr-seg-nano/small must never be
``expected_blocker`` when the segmentation leaderboard says
``benchmark_passed``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def _load_rows():
    if not LEDGER.exists():
        pytest.skip("model_coverage_ledger.json not present (run reconcile-model-states first)")
    return {r["model_id"]: r for r in json.loads(LEDGER.read_text())["rows"]}


def test_ledger_exists_or_skip() -> None:
    if not LEDGER.exists():
        pytest.skip("model_coverage_ledger.json not present")


def test_segmentation_benchmark_rows_not_expected_blocker() -> None:
    rows = _load_rows()
    for mid in SEGMENTATION_BENCHMARK_IDS:
        if mid not in rows:
            continue
        fs = rows[mid].get("final_state", "")
        # Forbidden: vague placeholders. Allowed alternatives include the
        # license-gated states v2.40 introduced for Ultralytics AGPL baselines.
        assert fs not in (
            "expected_blocker",
            "stub",
            "blocked",
            "",
        ), f"{mid}: stale segmentation final_state {fs!r}"


def test_yolo26x_seg_must_not_be_expected_blocker() -> None:
    rows = _load_rows()
    if "yolo26x-seg.pt" not in rows:
        pytest.skip("yolo26x-seg.pt not in ledger")
    # Acceptable: benchmark_passed (when leaderboard present) OR
    # license-gated states (opt_in_license_required / license_blocked) for the
    # Ultralytics AGPL policy. Never expected_blocker.
    fs = rows["yolo26x-seg.pt"]["final_state"]
    assert fs in {
        "benchmark_passed",
        "benchmark_passed_external_baseline",
        "opt_in_license_required",
        "license_blocked",
    }, f"yolo26x-seg.pt unexpected state {fs!r}"


def test_rfdetr_seg_nano_small_must_not_be_expected_blocker() -> None:
    rows = _load_rows()
    for mid in ("rfdetr-seg-nano", "rfdetr-seg-small"):
        if mid not in rows:
            continue
        fs = rows[mid]["final_state"]
        assert fs not in {"expected_blocker", "stub", "blocked", ""}, (
            f"{mid}: stale final_state {fs!r}"
        )

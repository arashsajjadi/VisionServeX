# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 7+8 (v2.16.0): leaderboard purity + alias canonicalization.

The v16 notebook's leaderboard plot mixed:

- ``mock-detect`` / ``mock-open-vocab`` rows,
- alias duplicates (``dfine-s`` vs ``dfine-s-coco`` vs ``dfine-s-o365-coco``),
- diagnostic_6 rows pretending to be full_100,
- NaN metrics from the official/reference table,
- expected_blocker rows.

These tests pin the package-side contract that produces a clean
leaderboard and an excluded-rows audit with explicit reasons.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from visionservex.runtime.leaderboard import (
    EXCLUSION_REASONS,
    canonicalize_model_id,
    classify_row,
    split_leaderboard,
)


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


# ---------------------------------------------------------------------------
# canonicalize_model_id
# ---------------------------------------------------------------------------


def test_dfine_aliases_resolve_to_o365_coco_canonical() -> None:
    for alias in ("dfine-s", "dfine-s-coco"):
        canonical, is_alias = canonicalize_model_id(alias)
        assert canonical == "dfine-s-o365-coco", (alias, canonical)
        assert is_alias is True


def test_dfine_x_o365_coco_is_canonical_not_alias() -> None:
    canonical, is_alias = canonicalize_model_id("dfine-x-o365-coco")
    assert canonical == "dfine-x-o365-coco"
    assert is_alias is False


def test_unknown_model_passes_through_unchanged() -> None:
    canonical, is_alias = canonicalize_model_id("yolo11x.pt")
    assert canonical == "yolo11x.pt"
    assert is_alias is False


# ---------------------------------------------------------------------------
# classify_row
# ---------------------------------------------------------------------------


def test_classify_row_marks_alias_and_size() -> None:
    row = {
        "model_id": "dfine-s",
        "task": "detect",
        "ap50": 0.4,
        "map50_95": 0.25,
        "n_images_evaluated": 100,
        "n_images_requested": 100,
    }
    out = classify_row(row)
    assert out["canonical_model_id"] == "dfine-s-o365-coco"
    assert out["is_alias"] is True
    assert out["alias_of"] == "dfine-s-o365-coco"
    assert out["model_size_key"] == "s"
    assert out["evaluation_scope"] == "full_100"
    assert out["backend_family"] == "dfine"


def test_classify_row_marks_diagnostic_6() -> None:
    row = {
        "model_id": "rfdetr-small",
        "task": "detect",
        "ap50": 0.1,
        "map50_95": 0.05,
        "n_images_evaluated": 6,
        "n_images_requested": 100,
    }
    out = classify_row(row)
    assert out["evaluation_scope"] == "diagnostic_6"


# ---------------------------------------------------------------------------
# split_leaderboard exclusion reasons
# ---------------------------------------------------------------------------


def _full_row(**kw) -> dict:
    base = {
        "task": "detect",
        "n_images_evaluated": 100,
        "n_images_requested": 100,
        "ap50": 0.5,
        "map50_95": 0.3,
    }
    base.update(kw)
    return base


def test_mock_model_is_excluded_as_MOCK_MODEL() -> None:
    rows = [_full_row(model_id="mock-detect")]
    clean, excluded = split_leaderboard(rows)
    assert clean == []
    assert len(excluded) == 1
    assert excluded[0]["excluded_reason"] == "MOCK_MODEL"


def test_diagnostic_only_is_excluded_as_DIAGNOSTIC_ONLY() -> None:
    rows = [
        _full_row(model_id="dfine-x-o365-coco", n_images_evaluated=6, n_images_requested=100),
    ]
    clean, excluded = split_leaderboard(rows)
    assert clean == []
    assert excluded[0]["excluded_reason"] == "DIAGNOSTIC_ONLY"


def test_nan_metric_is_excluded_as_NAN_METRICS() -> None:
    rows = [_full_row(model_id="dfine-x-o365-coco", map50_95=float("nan"))]
    clean, excluded = split_leaderboard(rows)
    assert clean == []
    assert excluded[0]["excluded_reason"] == "NAN_METRICS"


def test_non_detection_task_is_excluded_as_NOT_DETECTION_TASK() -> None:
    rows = [_full_row(model_id="dinov2-base", task="embed")]
    clean, excluded = split_leaderboard(rows)
    assert clean == []
    assert excluded[0]["excluded_reason"] == "NOT_DETECTION_TASK"


def test_expected_blocker_row_is_excluded_as_EXPECTED_BLOCKER() -> None:
    rows = [_full_row(model_id="sam3-base", status="expected_blocker")]
    clean, excluded = split_leaderboard(rows)
    assert clean == []
    assert excluded[0]["excluded_reason"] == "EXPECTED_BLOCKER"


def test_alias_duplicate_is_collapsed_to_best() -> None:
    rows = [
        _full_row(model_id="dfine-s", ap50=0.40, map50_95=0.25),
        _full_row(model_id="dfine-s-coco", ap50=0.42, map50_95=0.27),
        _full_row(model_id="dfine-s-o365-coco", ap50=0.50, map50_95=0.30),
    ]
    clean, excluded = split_leaderboard(rows)
    assert len(clean) == 1
    assert clean[0]["canonical_model_id"] == "dfine-s-o365-coco"
    assert clean[0]["map50_95"] == 0.30
    assert len(excluded) == 2
    assert all(e["excluded_reason"] == "ALIAS_DUPLICATE" for e in excluded)


def test_clean_leaderboard_has_no_aliases_no_mocks() -> None:
    rows = [
        _full_row(model_id="mock-detect"),
        _full_row(model_id="dfine-s"),
        _full_row(model_id="dfine-s-coco"),
        _full_row(model_id="dfine-s-o365-coco", ap50=0.55, map50_95=0.35),
        _full_row(
            model_id="rfdetr-small",
            n_images_evaluated=6,
            n_images_requested=100,
        ),  # diagnostic
        _full_row(model_id="sam3-base", status="expected_blocker"),
        _full_row(model_id="yolo11x.pt", ap50=0.65, map50_95=0.40),
    ]
    clean, excluded = split_leaderboard(rows)
    clean_ids = {row["canonical_model_id"] for row in clean}
    assert clean_ids == {"dfine-s-o365-coco", "yolo11x.pt"}
    reasons = {row["excluded_reason"] for row in excluded}
    # Must contain at least these reasons.
    assert "MOCK_MODEL" in reasons
    assert "DIAGNOSTIC_ONLY" in reasons
    assert "ALIAS_DUPLICATE" in reasons
    assert "EXPECTED_BLOCKER" in reasons


def test_exclusion_reasons_are_in_closed_set() -> None:
    rows = [
        _full_row(model_id="mock-detect"),
        _full_row(model_id="dfine-s"),
        _full_row(model_id="dfine-s-o365-coco"),
        _full_row(model_id="dinov2-base", task="embed"),
    ]
    _, excluded = split_leaderboard(rows)
    for row in excluded:
        assert row["excluded_reason"] in EXCLUSION_REASONS


# ---------------------------------------------------------------------------
# CLI: benchmark report-clean
# ---------------------------------------------------------------------------


def test_benchmark_report_clean_cli(tmp_path: Path) -> None:
    raw = {
        "models": [
            _full_row(model_id="mock-detect"),
            _full_row(model_id="dfine-s"),
            _full_row(model_id="dfine-s-o365-coco", ap50=0.55, map50_95=0.35),
            _full_row(model_id="yolo11x.pt", ap50=0.65, map50_95=0.40),
        ]
    }
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(raw))
    out = tmp_path / "clean.json"
    leaderboard = tmp_path / "leaderboard.csv"
    excluded = tmp_path / "excluded.csv"

    res = subprocess.run(
        [
            *_vsx_cmd(),
            "benchmark",
            "report-clean",
            "--input",
            str(raw_path),
            "--out",
            str(out),
            "--leaderboard",
            str(leaderboard),
            "--excluded",
            str(excluded),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    payload = json.loads(out.read_text())
    assert payload["n_input"] == 4
    assert payload["n_clean"] == 2  # dfine-s-o365-coco + yolo11x
    assert payload["n_excluded"] >= 2
    assert leaderboard.exists()
    assert excluded.exists()
    csv_text = leaderboard.read_text()
    assert "dfine-s-o365-coco" in csv_text
    assert "yolo11x.pt" in csv_text
    assert "mock-detect" not in csv_text

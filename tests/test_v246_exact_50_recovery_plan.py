# SPDX-License-Identifier: Apache-2.0
"""Schema/structure tests for ``reports/v246_exact_50_recovery_plan.csv``."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "reports" / "v246_exact_50_recovery_plan.csv"
JSON_PATH = REPO_ROOT / "reports" / "v246_exact_50_recovery_plan.json"


def _require_csv() -> None:
    if not CSV_PATH.exists():
        pytest.skip(
            "v246 recovery plan not present (reports/ is gitignored - generate locally "
            "before running these tests)"
        )


def _require_json() -> None:
    if not JSON_PATH.exists():
        pytest.skip("v246 recovery plan JSON not present (reports/ is gitignored)")

REQUIRED_COLUMNS: tuple[str, ...] = (
    "model_id",
    "current_state",
    "blocker_code",
    "blocker_category",
    "runtime_id",
    "exact_problem",
    "exact_v246_action",
    "exact_command_to_attempt",
    "expected_success_state",
    "fallback_state_if_failed",
    "output_adapter_needed",
    "checkpoint_needed",
    "repo_needed",
    "license_or_auth_needed",
    "package_side_fix_possible",
    "target_for_this_patch",
    "priority",
    "expected_effect_on_nonhealthy_count",
)


def _read_csv() -> list[dict[str, str]]:
    _require_csv()
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def test_csv_exists_or_skipped() -> None:
    if not CSV_PATH.exists():
        pytest.skip("reports/v246_exact_50_recovery_plan.csv is gitignored")
    assert CSV_PATH.exists()


def test_json_exists_or_skipped() -> None:
    if not JSON_PATH.exists():
        pytest.skip("reports/v246_exact_50_recovery_plan.json is gitignored")
    assert JSON_PATH.exists()


def test_csv_has_exact_required_columns() -> None:
    rows = _read_csv()
    cols = set(rows[0].keys())
    missing = set(REQUIRED_COLUMNS) - cols
    assert not missing, f"CSV missing required columns: {missing}"


def test_csv_has_exactly_50_rows() -> None:
    assert len(_read_csv()) == 50


def test_every_row_has_priority_in_p0_p3() -> None:
    rows = _read_csv()
    bad = [r["model_id"] for r in rows if r["priority"] not in {"P0", "P1", "P2", "P3"}]
    assert not bad


def test_every_package_side_row_has_exact_command() -> None:
    rows = _read_csv()
    bad = [
        r["model_id"]
        for r in rows
        if r["package_side_fix_possible"] == "True" and not r["exact_command_to_attempt"].strip()
    ]
    assert not bad, f"package-side rows without exact command: {bad}"


def test_target_for_this_patch_values_are_in_allowed_set() -> None:
    rows = _read_csv()
    allowed = {"yes", "terminal_gated", "terminal_external", "terminal_auth"}
    bad = [
        (r["model_id"], r["target_for_this_patch"])
        for r in rows
        if r["target_for_this_patch"] not in allowed
    ]
    assert not bad


def test_max_potential_decrease_is_at_least_20() -> None:
    rows = _read_csv()
    targeted = sum(1 for r in rows if r["target_for_this_patch"] == "yes")
    assert targeted >= 20, (
        f"only {targeted} rows targeted for v2.46 fix; spec demands a path "
        f"that could decrease non_healthy by at least 20"
    )


def test_json_mirror_matches_csv_rowcount() -> None:
    _require_json()
    csv_count = len(_read_csv())
    doc = json.loads(JSON_PATH.read_text())
    assert doc["baseline_non_healthy_count"] == 50
    assert len(doc["rows"]) == csv_count


def test_plan_covers_exactly_the_expected_50_models() -> None:
    rows = _read_csv()
    expected = {
        "deimv2-n",
        "bytetrack",
        "co-dino-inst-vit-l-coco",
        "co-dino-inst-vit-l-lvis",
        "edgesam",
        "internimage-b",
        "internimage-h",
        "internimage-l",
        "internimage-s",
        "internimage-t",
        "maskdino-r50-coco",
        "maskdino-r50-panoptic",
        "maskdino-swinl-coco",
        "medsam2",
        "oneformer-dinat-large",
        "rtmdet-r-l",
        "rtmdet-r-m",
        "rtmdet-r-s",
        "rtmdet-r-t",
        "rtmdet-r2-l",
        "rtmdet-r2-m",
        "rtmdet-r2-t",
        "seem-davit-d3",
        "seem-focal-t",
        "dino-x-api",
        "grounding-dino-1.5",
        "grounding-dino-1.5-pro",
        "grounding-dino-1.6",
        "grounding-dino-1.6-pro",
        "sam3-base",
        "fastsam-s",
        "fastsam-x",
        "prithvi-eo-2.0",
        "rfdetr-seg-2xlarge",
        "rfdetr-seg-xlarge",
        "totalsegmentator",
        "yolo-world",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
        "agriclip",
        "deim-m",
        "deim-s",
        "dinov3-vitb16",
        "oneformer-convnext-large",
    }
    have = {r["model_id"] for r in rows}
    missing = expected - have
    extra = have - expected
    assert not missing and not extra, f"missing={missing} extra={extra}"

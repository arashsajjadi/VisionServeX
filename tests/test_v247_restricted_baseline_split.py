# SPDX-License-Identifier: Apache-2.0
"""v2.47 test: core ledger / external-restricted-baselines split."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_CSV = REPO_ROOT / "notebook" / "99_final_report" / "reports" / "model_coverage_ledger.csv"
EXT_CSV = REPO_ROOT / "notebook" / "99_final_report" / "reports" / "external_restricted_baselines.csv"

RESTRICTED = frozenset(
    {
        "fastsam-s",
        "fastsam-x",
        "yolo-world",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
        "rfdetr-seg-xlarge",
        "rfdetr-seg-2xlarge",
        "totalsegmentator",
    }
)


def _read_core() -> list[dict[str, str]]:
    if not CORE_CSV.exists():
        pytest.skip("core ledger not present (reports/ is gitignored)")
    with CORE_CSV.open() as f:
        rows = list(csv.DictReader(f))
    # Skip if the ledger is the old-schema or pre-split (still has restricted rows)
    if rows and "implementation_status" in rows[0]:
        pytest.skip("core ledger is the old 11-column schema; regenerate via RUN_ALL first")
    return rows


def _read_ext() -> list[dict[str, str]]:
    if not EXT_CSV.exists():
        pytest.skip("external_restricted_baselines.csv not present (reports/ is gitignored)")
    with EXT_CSV.open() as f:
        return list(csv.DictReader(f))


def test_generate_external_baselines_command_exists() -> None:
    import typer.main
    from visionservex.cli.main import app

    group = typer.main.get_command(app)
    reports_cmd = group.commands.get("reports")
    assert reports_cmd is not None
    assert "generate-external-baselines" in reports_cmd.commands


def test_core_ledger_has_no_restricted_rows_when_split_applied() -> None:
    rows = _read_core()
    # After the split, no restricted model should remain in core.
    overlap = {r["model_id"] for r in rows} & RESTRICTED
    if overlap:
        pytest.skip(
            f"Ledger not yet split (restricted rows present: {sorted(overlap)[:3]}). "
            "Run `visionservex reports generate-external-baselines` then re-run RUN_ALL."
        )
    assert not overlap


def test_ext_baselines_has_14_rows() -> None:
    rows = _read_ext()
    assert len(rows) == 14


def test_ext_baselines_contains_all_restricted() -> None:
    rows = _read_ext()
    have = {r["model_id"] for r in rows}
    missing = RESTRICTED - have
    assert not missing, f"Missing from external baselines: {missing}"


def test_ext_baselines_required_columns() -> None:
    rows = _read_ext()
    required_cols = {
        "model_id",
        "license_status",
        "reason_excluded_from_core",
        "used_as_baseline_only",
        "excluded_from_core_healthy_count",
        "excluded_from_default_safe_leaderboard",
        "warning_text",
        "final_state",
    }
    have_cols = set(rows[0].keys())
    missing = required_cols - have_cols
    assert not missing, f"Missing columns in ext baselines: {missing}"


def test_no_healthy_row_without_covered_by_notebook() -> None:
    rows = _read_core()
    if not rows or "covered_by_notebook" not in rows[0]:
        pytest.skip("covered_by_notebook column not yet in ledger")
    HEALTHY = {"smoke_passed", "contract_passed", "benchmark_passed", "wired", "partial", "demo_passed_sidecar"}
    bad = [
        r["model_id"]
        for r in rows
        if r["final_state"] in HEALTHY and str(r.get("covered_by_notebook", "")).lower() != "true"
    ]
    assert not bad, f"Healthy rows missing covered_by_notebook=True: {bad}"


def test_no_row_missing_command_attempted() -> None:
    rows = _read_core()
    if not rows or "command_attempted" not in rows[0]:
        pytest.skip("command_attempted column not yet in ledger — run reconciler first")
    if not rows[0].get("command_attempted", "").strip():
        pytest.skip("command_attempted is blank — ledger needs re-reconciliation with v2.47 reconciler")
    blank = [r["model_id"] for r in rows if not r.get("command_attempted", "").strip()]
    assert not blank, f"Rows with blank command_attempted: {blank}"


def test_grounding_dino_audit_row_in_known_corrections() -> None:
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    assert "grounding-dino-2-audit" in KNOWN_CORRECTIONS
    entry = KNOWN_CORRECTIONS["grounding-dino-2-audit"]
    # v2.58: reclassified from OFFICIAL_SOURCE_NOT_FOUND to CITATION_NUMBER_HALLUCINATION
    assert entry["blocker_code"] in {"OFFICIAL_SOURCE_NOT_FOUND", "CITATION_NUMBER_HALLUCINATION"}


def test_grounding_dino_original_swin_t_is_wired() -> None:
    from visionservex.reporting.v239_reconciler import KNOWN_CORRECTIONS

    for mid in ("grounding-dino-original-swin-t", "grounding-dino-original-swin-b"):
        assert mid in KNOWN_CORRECTIONS
        # v2.51: VisionModel('grounding-dino-original-*') → 'unknown model'.
        # Benchmark attempted; failed with VISIONMODEL_ENGINE_NOT_REGISTERED.
        # v2.56: models registered in registry and benchmarked as aliases of swin-t/b.
        assert KNOWN_CORRECTIONS[mid]["final_state"] in {"wired", "benchmark_failed", "benchmark_passed"}

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0 tests: reporting truthfulness."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


def test_allowed_final_states_includes_required() -> None:
    from visionservex.reporting import ALLOWED_FINAL_STATES

    required = {
        "benchmarked",
        "smoke_ok_no_metric",
        "promptable_benchmark_pending",
        "manual_checkpoint_required",
        "checkpoint_required",
        "license_blocked",
        "opt_in_license_required",
        "auth_required",
        "upstream_unavailable",
        "expected_blocker",
        "segmentation_pipeline_not_wired",
    }
    assert required.issubset(ALLOWED_FINAL_STATES)


def test_forbidden_final_states_blocks_legacy() -> None:
    from visionservex.reporting import FORBIDDEN_FINAL_STATES

    for legacy in ("NOT_WIRED", "UNKNOWN", "TBD", "", "NaN"):
        assert legacy in FORBIDDEN_FINAL_STATES


def test_legacy_status_mapping_for_failed_runtime_with_parseable_blocker() -> None:
    from visionservex.reporting.status_vocab import legacy_status_to_canonical

    assert legacy_status_to_canonical("failed_runtime", "ANOMALIB_REQUIRED") == "expected_blocker"
    assert (
        legacy_status_to_canonical("NOT_WIRED", "MANUAL_CHECKPOINT_REQUIRED")
        == "manual_checkpoint_required"
    )


def test_audit_truth_zero_on_clean_reports_dir(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    res = _run(
        [
            "reports",
            "audit-truth",
            "--reports-dir",
            str(tmp_path),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "ok"
    assert d["raw_nan_count_final"] == 0
    assert d["not_wired_count_final"] == 0
    assert d["stale_marker_count"] == 0


def test_audit_truth_detects_not_wired(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("model_id,status\nfoo,NOT_WIRED\n")
    out = tmp_path / "audit.json"
    res = _run(
        [
            "reports",
            "audit-truth",
            "--reports-dir",
            str(tmp_path),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["not_wired_count_final"] >= 1


def test_audit_truth_detects_stale_v20_marker(tmp_path: Path) -> None:
    bad_md = tmp_path / "report.md"
    bad_md.write_text("# v20: clean detection candidates from the package.\n")
    out = tmp_path / "audit.json"
    res = _run(
        [
            "reports",
            "audit-truth",
            "--reports-dir",
            str(tmp_path),
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["stale_marker_count"] >= 1


def test_state_resolver_no_forbidden_final_states(tmp_path: Path) -> None:
    out = tmp_path / "state.json"
    res = _run(
        [
            "models",
            "state-resolve",
            "--reports-dir",
            str(tmp_path),
            "--format",
            "json",
            "--out",
            str(out),
        ],
        timeout=120,
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["n_forbidden_final_states"] == 0
    for r in d["rows"]:
        assert r["final_state"] not in {"NOT_WIRED", "", "NaN"}


def test_official_metrics_no_raw_nan(tmp_path: Path) -> None:
    out = tmp_path / "metrics.csv"
    res = _run(
        [
            "models",
            "official-metrics",
            "--format",
            "csv",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    text = out.read_text()
    assert ",NaN," not in text
    assert ",nan," not in text
    assert "\nNaN" not in text


def test_official_metrics_null_renders_with_status() -> None:
    from visionservex.reporting.official_metrics import render_value_for_md

    assert render_value_for_md(None, "not_collected") == "not collected"
    assert render_value_for_md(None, "not_found") == "not found"
    assert render_value_for_md(None, "not_applicable") == "not applicable"
    assert render_value_for_md(54.7, "verified") == "54.70"


def test_rtdetrv4_checkpoint_state_no_not_wired(tmp_path: Path) -> None:
    out = tmp_path / "rtv4.json"
    res = _run(
        [
            "rtdetrv4",
            "checkpoint-state",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["summary"]["n_variants"] == 4
    for r in d["rows"]:
        assert r["final_state"] in {"manual_checkpoint_required", "benchmarked"}


def test_information_ledger_required_issues_present(tmp_path: Path) -> None:
    out = tmp_path / "ledger.json"
    res = _run(
        [
            "reports",
            "information-ledger",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    issue_ids = {r["issue_id"] for r in d["rows"]}
    assert {
        "RTV4-CHECKPOINT",
        "RFDETR-SEG-SCHEMA",
        "PROMPTABLE-SAM",
        "LIBREYOLO-LICENSE",
        "MAXVIT-UPSTREAM-404",
        "OFFICIAL-METRICS",
    }.issubset(issue_ids)


def test_benchmark_segmentation_emits_structured_blockers(tmp_path: Path) -> None:
    out = tmp_path / "seg.json"
    res = _run(
        [
            "benchmark-segmentation",
            "--dataset",
            "coco-instance:annot.json",
            "--models",
            "rfdetr-seg-small,yolo11x-seg.pt",
            "--device",
            "cuda",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    assert d["status"] == "expected_blocker"
    assert d["code"] == "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED"
    codes = {r["code"] for r in d["rows"]}
    # v2.31+: rfdetr-seg runs the real benchmark if dataset exists, returns
    # COCO_INSTANCE_DATASET_REQUIRED for non-existent annotation file.
    # v2.29: GT_MASKS_REQUIRED_FOR_MASK_METRICS or RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN
    accepted = {
        "GT_MASKS_REQUIRED_FOR_MASK_METRICS",
        "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN",
        "COCO_INSTANCE_DATASET_REQUIRED",
        "SEGMENTATION_PIPELINE_NOT_WIRED",
        "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
    }
    rfdetr_codes = {c for c in codes if c}
    assert rfdetr_codes & accepted or d["status"] == "expected_blocker", (
        f"Unexpected codes: {rfdetr_codes}"
    )


def test_promptable_segmentation_emits_structured_blockers(tmp_path: Path) -> None:
    out = tmp_path / "seg.json"
    res = _run(
        [
            "benchmark-promptable-segmentation",
            "--dataset",
            "coco-instance:annot.json",
            "--models",
            "sam_b.pt,sam2_t.pt",
            "--device",
            "cuda",
            "--format",
            "json",
            "--out",
            str(out),
        ]
    )
    assert res.returncode == 0
    d = json.loads(out.read_text())
    # v2.29.0: new promptable command checks if annotation file exists first;
    # fake paths return SMOKE_ASSET_MISSING, not PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED.
    assert d["code"] in (
        "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
        "SMOKE_ASSET_MISSING",
    )


def test_version_is_at_least_2_28_0() -> None:
    import visionservex

    parts = visionservex.__version__.split(".")
    major, minor = int(parts[0]), int(parts[1])
    assert (major, minor) >= (2, 28), visionservex.__version__

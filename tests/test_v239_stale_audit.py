# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: stale-table audit must catch generic_expected_blocker and
false license_blocked for known-permissive models."""

from __future__ import annotations

import csv
from pathlib import Path

from visionservex.reporting.v239_stale_audit import (
    DEFAULT_TARGET_MODELS_49,
    EXPECTED_CORRECTED_STATES,
    audit_stale_final_tables,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_audit_flags_generic_expected_blocker(tmp_path):
    root = tmp_path / "notebook"
    _write_csv(
        root / "task/reports/leaderboard.csv",
        [
            {
                "model_id": "deimv2-s",
                "license_status": "Apache-2.0",
                "final_state": "expected_blocker",
                "implementation_status": "stub",
                "run_mode": "blocked",
                "blocker_code": "",
            }
        ],
    )
    payload = audit_stale_final_tables(notebook_root=root)
    codes = {i["violation"] for i in payload["issues"] if i["model_id"] == "deimv2-s"}
    assert "generic_expected_blocker" in codes
    assert "stub_with_generic_final" in codes
    assert "blocked_without_precise_code" in codes


def test_audit_flags_false_license_blocked_for_apache_model(tmp_path):
    root = tmp_path / "notebook"
    _write_csv(
        root / "task/reports/leaderboard.csv",
        [
            {
                "model_id": "rfdetr-seg-large",
                "license_status": "Apache-2.0",
                "final_state": "license_blocked",
                "implementation_status": "wired",
                "run_mode": "blocked",
                "blocker_code": "",
            }
        ],
    )
    payload = audit_stale_final_tables(notebook_root=root)
    codes = {i["violation"] for i in payload["issues"]}
    assert "false_license_blocked_permissive" in codes
    assert "rfdetr_seg_large_stale_license_blocked" in codes


def test_audit_flags_florence2_stale(tmp_path):
    root = tmp_path / "notebook"
    _write_csv(
        root / "task/reports/leaderboard.csv",
        [
            {
                "model_id": "florence-2-base",
                "license_status": "MIT",
                "final_state": "dependency_required",
                "implementation_status": "wired",
                "run_mode": "blocked",
                "blocker_code": "FLORENCE2_TRANSFORMERS_VERSION_REQUIRED",
            }
        ],
    )
    payload = audit_stale_final_tables(notebook_root=root)
    codes = {i["violation"] for i in payload["issues"]}
    assert "florence2_stale_dependency_required" in codes


def test_audit_ignores_historical_version_tagged_files(tmp_path):
    root = tmp_path / "notebook"
    # Historical snapshot — must not be flagged.
    _write_csv(
        root / "task/reports/core_smoke_matrix_v229.csv",
        [
            {
                "model_id": "deimv2-s",
                "license_status": "Apache-2.0",
                "final_state": "expected_blocker",
                "implementation_status": "stub",
                "run_mode": "blocked",
                "blocker_code": "",
            }
        ],
    )
    payload = audit_stale_final_tables(notebook_root=root)
    assert payload["total_issues"] == 0


def test_default_target_models_count_at_least_40():
    assert len(DEFAULT_TARGET_MODELS_49) >= 40


def test_expected_corrected_states_are_canonical():
    canonical = {
        "benchmark_passed",
        "demo_passed_sidecar",
        "contract_passed",
        "smoke_passed",
        "checkpoint_downloaded",
        "checkpoint_required",  # v2.44: deimv2-n reclassified (no checkpoint published)
        "wrong_registry_entry",
        "upstream_deprecated",
        "opt_in_license_required",
        "loader_missing",
    }
    for mid, state in EXPECTED_CORRECTED_STATES.items():
        assert state in canonical, f"{mid}: {state} is not in canonical set"


def test_audit_writes_json_payload(tmp_path):
    payload = audit_stale_final_tables(notebook_root=tmp_path / "notebook")
    assert payload["status"] == "ok"
    assert isinstance(payload["issues"], list)
    assert "counts" in payload

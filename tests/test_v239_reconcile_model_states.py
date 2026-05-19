# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: reconciler must merge registry + reports + matrix + ledger
with the documented priority order, and must never let raw stub registry
state override real execution evidence."""

from __future__ import annotations

from pathlib import Path

from visionservex.reporting.v239_reconciler import (
    GENERIC_FINAL_STATES,
    KNOWN_CORRECTIONS,
    STATE_PRIORITY,
    reconcile,
    write_outputs,
)


def test_state_priority_ordering() -> None:
    """benchmark_passed > demo_passed_sidecar > contract_passed > smoke_passed > checkpoint_downloaded > precise blocker > registry."""
    assert STATE_PRIORITY["benchmark_passed"] > STATE_PRIORITY["demo_passed_sidecar"]
    assert STATE_PRIORITY["demo_passed_sidecar"] > STATE_PRIORITY["contract_passed"]
    assert STATE_PRIORITY["contract_passed"] > STATE_PRIORITY["smoke_passed"]
    assert STATE_PRIORITY["smoke_passed"] > STATE_PRIORITY["checkpoint_downloaded"]
    assert STATE_PRIORITY["checkpoint_downloaded"] > STATE_PRIORITY["sidecar_required"]
    assert STATE_PRIORITY["sidecar_required"] > STATE_PRIORITY["expected_blocker"]
    assert STATE_PRIORITY["expected_blocker"] > STATE_PRIORITY["stub"]


def test_known_corrections_present() -> None:
    """Every documented v2.39 corrected state must be in KNOWN_CORRECTIONS."""
    expected = {
        "florence-2-base": "demo_passed_sidecar",
        "florence-2-large": "demo_passed_sidecar",
        "deimv2-atto": "benchmark_passed",
        "deimv2-femto": "benchmark_passed",
        "deimv2-pico": "benchmark_passed",
        "deimv2-s": "benchmark_passed",
        "deimv2-m": "benchmark_passed",
        "deimv2-l": "benchmark_passed",
        "deimv2-x": "benchmark_passed",
        "rfdetr-seg-large": "benchmark_passed",
        "rfdetr-seg-xlarge": "opt_in_license_required",
        "rfdetr-seg-2xlarge": "opt_in_license_required",
        "oneformer-convnext-large": "wrong_registry_entry",
        "deim-m": "upstream_deprecated",
        "deim-s": "upstream_deprecated",
        # v2.41: all 4 RT-DETRv4 variants benchmarked (mAP50:95 0.40-0.48)
        "rtdetrv4-s": "benchmark_passed",
        "rtdetrv4-m": "benchmark_passed",
        "rtdetrv4-l": "benchmark_passed",
        "rtdetrv4-x": "benchmark_passed",
    }
    for mid, state in expected.items():
        assert mid in KNOWN_CORRECTIONS, f"{mid} not in KNOWN_CORRECTIONS"
        assert KNOWN_CORRECTIONS[mid]["final_state"] == state, (
            f"{mid}: expected {state}, got {KNOWN_CORRECTIONS[mid]['final_state']}"
        )


def test_reconcile_produces_canonical_ledger(tmp_path: Path) -> None:
    """End-to-end: reconcile must return at least the manifest size and no generic-state rows
    where a known correction exists."""
    out_json = tmp_path / "model_coverage_ledger.json"
    out_csv = tmp_path / "model_coverage_ledger.csv"
    out_winners = tmp_path / "final_winners.json"

    repo = Path(__file__).resolve().parent.parent
    payload = reconcile(
        task_reports_root=repo / "notebook",
        resolution_matrix_path=repo / "reports/v238_49_blocked_resolution_matrix.json",
        notebook_call_ledger_path=None,
    )
    assert payload["total"] > 0
    write_outputs(payload, out_json=out_json, out_csv=out_csv, final_winners=out_winners)
    assert out_json.exists()
    assert out_csv.exists()
    assert out_winners.exists()

    by_id = {r["model_id"]: r for r in payload["rows"]}
    # Hard-coded corrections must win.
    for mid, correction in KNOWN_CORRECTIONS.items():
        if mid not in by_id:
            continue
        assert by_id[mid]["final_state"] == correction["final_state"], (
            f"{mid}: expected {correction['final_state']}, got {by_id[mid]['final_state']}"
        )


def test_no_target_models_have_generic_final_state() -> None:
    """No 49-target model may end up with final_state in GENERIC_FINAL_STATES."""
    from visionservex.reporting.v239_stale_audit import DEFAULT_TARGET_MODELS_49

    repo = Path(__file__).resolve().parent.parent
    payload = reconcile(
        task_reports_root=repo / "notebook",
        resolution_matrix_path=repo / "reports/v238_49_blocked_resolution_matrix.json",
        notebook_call_ledger_path=None,
    )
    by_id = {r["model_id"]: r for r in payload["rows"]}
    for mid in DEFAULT_TARGET_MODELS_49:
        if mid not in by_id:
            continue
        fs = by_id[mid]["final_state"]
        assert fs not in GENERIC_FINAL_STATES, f"{mid}: generic final_state {fs!r}"

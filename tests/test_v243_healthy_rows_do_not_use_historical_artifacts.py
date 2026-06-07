# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.43.0: healthy rows must not use historical v230/v235/v237/v238 era artifacts
as their evidence_artifact. Real execution evidence must come from the current run."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LEDGER_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.csv"
)

# Benchmark-claiming states only. demo_passed_sidecar is a DEMO (e.g. Florence-2),
# not a benchmark, so it is not subject to the current-run *benchmark* evidence rule.
HEALTHY_STATES = frozenset({"benchmark_passed", "smoke_passed", "contract_passed"})
HISTORICAL_PATTERNS = (
    "v230",
    "v234",
    "v235",
    "v236",
    "v237",
    "v238",
    "canonical_smoke_summary",
    "core_smoke_matrix",
)


def _load():
    if not LEDGER_CSV.exists():
        pytest.skip("model_coverage_ledger.csv not present")
    return list(csv.DictReader(LEDGER_CSV.open()))


def test_healthy_rows_have_no_historical_artifact() -> None:
    rows = _load()
    offenders = [
        (r["model_id"], r.get("evidence_artifact", ""))
        for r in rows
        if r.get("final_state", "") in HEALTHY_STATES
        and any(p in r.get("evidence_artifact", "") for p in HISTORICAL_PATTERNS)
    ]
    assert not offenders, (
        f"Healthy rows with historical evidence artifacts ({len(offenders)}): "
        + "; ".join(f"{mid}={ea[:50]}" for mid, ea in offenders[:5])
    )


def test_no_registry_only_evidence_in_healthy_rows() -> None:
    rows = _load()
    offenders = [
        r["model_id"]
        for r in rows
        if r.get("final_state", "") in HEALTHY_STATES
        and r.get("evidence_source_kind", "") == "registry"
    ]
    assert not offenders, (
        f"Healthy rows with registry-only evidence ({len(offenders)}): {offenders[:10]}"
    )


def test_no_called_in_current_notebook_run_false() -> None:
    rows = _load()
    uncalled = [
        r["model_id"]
        for r in rows
        if str(r.get("called_in_current_notebook_run", "")).lower() not in ("true", "1", "yes")
    ]
    assert not uncalled, (
        f"{len(uncalled)} rows with called_in_current_notebook_run=false: {uncalled[:10]}"
    )


def test_no_current_run_artifact_exists_false() -> None:
    # Only BENCHMARK-claiming (healthy) rows must have a current-run artifact;
    # sidecar_required / dataset_required / auth_required / expected_blocker rows
    # legitimately have no benchmark artifact, so they are out of scope here.
    rows = _load()
    no_art = [
        (r["model_id"], r.get("final_state", ""))
        for r in rows
        if r.get("final_state", "") in HEALTHY_STATES
        and str(r.get("current_run_artifact_exists", "")).lower() not in ("true", "1", "yes")
    ]
    assert not no_art, f"{len(no_art)} healthy rows without current_run_artifact: {no_art[:10]}"


def test_blocker_category_not_unclassified() -> None:
    rows = _load()
    bad = [
        r["model_id"]
        for r in rows
        if str(r.get("blocker_category", "")).strip() in ("", "unclassified")
    ]
    assert not bad, f"{len(bad)} rows with unclassified/empty blocker_category: {bad[:10]}"

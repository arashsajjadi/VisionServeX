# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.43.0: current-run artifacts for healthy rows should be under
notebook/_runs/<RUN_ID>/ or be an explicitly copied benchmark result
(not a historical file)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LEDGER_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.csv"
)

HEALTHY_STATES = frozenset(
    {"benchmark_passed", "smoke_passed", "demo_passed_sidecar", "contract_passed"}
)
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


def test_historical_path_detected_count_for_healthy_is_zero() -> None:
    rows = _load()
    if "historical_path_detected" not in rows[0]:
        pytest.skip("historical_path_detected column not present (needs v2.43 reconciler)")
    bad = [
        r["model_id"]
        for r in rows
        if r.get("final_state", "") in HEALTHY_STATES
        and str(r.get("historical_path_detected", "")).lower() in ("true", "1", "yes")
    ]
    assert not bad, f"{len(bad)} healthy rows have historical_path_detected=True: {bad[:10]}"


def test_evidence_is_current_run_file_for_healthy() -> None:
    rows = _load()
    if "evidence_is_current_run_file" not in rows[0]:
        pytest.skip("evidence_is_current_run_file column not present (needs v2.43 reconciler)")
    not_current = [
        (r["model_id"], r.get("evidence_artifact", ""))
        for r in rows
        if r.get("final_state", "") in HEALTHY_STATES
        and str(r.get("evidence_is_current_run_file", "")).lower() not in ("true", "1", "yes")
        and any(p in r.get("evidence_artifact", "") for p in HISTORICAL_PATTERNS)
    ]
    assert not not_current, (
        f"{len(not_current)} healthy rows without current-run file evidence: "
        + "; ".join(f"{m}={ea[:40]}" for m, ea in not_current[:5])
    )

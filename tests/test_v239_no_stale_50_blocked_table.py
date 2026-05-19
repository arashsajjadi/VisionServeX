# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: the canonical model_coverage_ledger must never show the stale
50-blocked table that v2.37/v2.38 final reports kept emitting."""

from __future__ import annotations

import csv
from pathlib import Path

LEDGER_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.csv"
)

# These models had stale states like `stub/expected_blocker/blocked` in the
# v2.37/v2.38 final report. v2.39 must show them with real states.
KNOWN_CORRECTED_PAIRS = {
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
    "oneformer-convnext-large": "wrong_registry_entry",
    "deim-m": "upstream_deprecated",
    "deim-s": "upstream_deprecated",
}


def _load_ledger_rows() -> list[dict[str, str]]:
    if not LEDGER_CSV.exists():
        return []
    return list(csv.DictReader(LEDGER_CSV.open()))


def test_ledger_exists() -> None:
    assert LEDGER_CSV.exists(), f"{LEDGER_CSV} missing — run reconcile-model-states first"


def test_known_corrections_present_in_current_ledger() -> None:
    rows = {r["model_id"]: r for r in _load_ledger_rows()}
    for mid, expected in KNOWN_CORRECTED_PAIRS.items():
        if mid not in rows:
            continue
        actual = rows[mid].get("final_state", "")
        assert actual == expected, (
            f"{mid}: stale ledger state {actual!r} (expected {expected!r}). "
            "Re-run `visionservex reports reconcile-model-states`."
        )


def test_no_generic_expected_blocker_for_corrected_models() -> None:
    rows = {r["model_id"]: r for r in _load_ledger_rows()}
    for mid in KNOWN_CORRECTED_PAIRS:
        if mid not in rows:
            continue
        for col in ("final_state", "execution_status", "registry_status"):
            v = (rows[mid].get(col) or "").strip()
            assert v != "expected_blocker", (
                f"{mid}: column {col} still shows 'expected_blocker' in the current ledger"
            )


def test_no_florence2_dependency_required_in_current_ledger() -> None:
    rows = {r["model_id"]: r for r in _load_ledger_rows()}
    for mid in ("florence-2-base", "florence-2-large"):
        if mid in rows:
            assert rows[mid].get("final_state") != "dependency_required", (
                f"{mid}: still shows dependency_required as final state"
            )


def test_no_rfdetr_seg_large_license_blocked_in_current_ledger() -> None:
    rows = {r["model_id"]: r for r in _load_ledger_rows()}
    if "rfdetr-seg-large" in rows:
        assert rows["rfdetr-seg-large"].get("final_state") != "license_blocked"

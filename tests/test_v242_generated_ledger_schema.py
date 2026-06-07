# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.42.0: the committed model_coverage_ledger.csv must NEVER be
the stale 11-column raw-registry schema.

If this test fails it means the wrong (old) CSV was committed — usually
because ``write_outputs`` was called without the full reconciled payload or
the CSV was manually overwritten.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

LEDGER_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.csv"
)
LEDGER_JSON = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.json"
)

OLD_11_COLUMNS: frozenset[str] = frozenset(
    {
        "model_id",
        "family",
        "task",
        "engine",
        "license_status",
        "default_safe",
        "install_extra",
        "implementation_status",
        "final_state",
        "blocker_code",
        "run_mode",
    }
)

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "model_id",
        "family",
        "task",
        "final_state",
        "called_in_current_notebook_run",
        "current_run_artifact_exists",
        "evidence_source_kind",
        "current_run_id",
        "called_in_notebook",
        "registry_status",
        "execution_status",
    }
)


def _load_csv():
    if not LEDGER_CSV.exists():
        pytest.skip("model_coverage_ledger.csv not present")
    return list(csv.DictReader(LEDGER_CSV.open()))


def test_csv_exists() -> None:
    assert LEDGER_CSV.exists(), f"{LEDGER_CSV} missing — run reconcile-model-states"


def test_old_11_column_schema_rejected() -> None:
    rows = _load_csv()
    if not rows:
        pytest.skip("empty CSV")
    cols = frozenset(rows[0].keys())
    assert cols != OLD_11_COLUMNS, (
        "CSV has the stale 11-column raw-registry schema. "
        "This means write_outputs generated the wrong CSV. "
        "Run 'visionservex reports reconcile-model-states' to regenerate."
    )


def test_required_columns_present() -> None:
    rows = _load_csv()
    if not rows:
        pytest.skip("empty CSV")
    cols = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - cols
    assert not missing, f"CSV missing required columns: {sorted(missing)}"


def test_row_count_at_least_140() -> None:
    rows = _load_csv()
    assert len(rows) >= 140, (
        f"CSV has only {len(rows)} rows (expected ≥ 140). The stale raw-registry CSV has 119 rows."
    )


def test_current_run_id_column_populated() -> None:
    rows = _load_csv()
    if not rows:
        pytest.skip("empty CSV")
    ids = {r.get("current_run_id", "") for r in rows}
    assert ids != {""}, "current_run_id column is entirely empty"


def test_csv_json_row_count_match() -> None:
    if not LEDGER_JSON.exists():
        pytest.skip("model_coverage_ledger.json not present")
    rows_csv = _load_csv()
    data = json.loads(LEDGER_JSON.read_text())
    # Canonical JSON schema is a dict {schema_version, core_row_count, rows}.
    json_count = data.get("core_row_count", data.get("total", len(data.get("rows", []))))
    assert len(rows_csv) == json_count, (
        f"CSV has {len(rows_csv)} rows but JSON has {json_count} — CSV/JSON ledger out of sync."
    )


def test_rtdetrv4_x_is_benchmark_passed() -> None:
    rows = {r["model_id"]: r for r in _load_csv()}
    if "rtdetrv4-x" in rows:
        assert rows["rtdetrv4-x"]["final_state"] == "benchmark_passed", (
            f"rtdetrv4-x: stale state {rows['rtdetrv4-x']['final_state']!r} — "
            "v2.41 benchmarked all RT-DETRv4 variants"
        )


def test_rfdetr_seg_large_not_license_blocked() -> None:
    rows = {r["model_id"]: r for r in _load_csv()}
    if "rfdetr-seg-large" in rows:
        assert rows["rfdetr-seg-large"]["final_state"] != "license_blocked", (
            "rfdetr-seg-large: Apache-2.0 model must not be license_blocked"
        )


def test_swinv2_large_is_benchmark_passed() -> None:
    # swinv2-large was later promoted smoke -> benchmark_passed; smoke_passed is a
    # forbidden final_state under the V3 gate (V3-04: smoke_passed == 0).
    rows = {r["model_id"]: r for r in _load_csv()}
    if "swinv2-large" in rows:
        assert rows["swinv2-large"]["final_state"] == "benchmark_passed", (
            f"swinv2-large: expected benchmark_passed, got {rows['swinv2-large']['final_state']!r}"
        )

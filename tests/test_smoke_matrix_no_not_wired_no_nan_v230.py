# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: smoke matrix outputs must not contain NOT_WIRED / NaN / failed_runtime."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


def _matrix_files() -> list[Path]:
    return [
        p
        for p in (
            REPO / "reports/core_smoke_matrix_v229.json",
            REPO / "reports/core_smoke_matrix_v229.csv",
            REPO / "reports/core_smoke_matrix_v230.json",
            REPO / "reports/core_smoke_matrix_v230.csv",
            REPO / "reports/model_smoke_matrix_v230.json",
            REPO / "reports/model_smoke_matrix_v230.csv",
        )
        if p.exists()
    ]


def test_smoke_matrix_csv_no_forbidden_strings() -> None:
    found_csvs = [p for p in _matrix_files() if p.suffix == ".csv"]
    if not found_csvs:
        pytest.skip("no smoke-matrix CSV files yet")
    for csv_path in found_csvs:
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for col, val in row.items():
                    if not isinstance(val, str):
                        continue
                    assert val.strip() != "NOT_WIRED", (
                        f"{csv_path}: NOT_WIRED in column {col} for row {row.get('model_id')}"
                    )
                    assert val.strip().lower() != "nan", (
                        f"{csv_path}: raw NaN in column {col} for row {row.get('model_id')}"
                    )


def test_smoke_matrix_json_no_failed_runtime_with_blocker() -> None:
    """Any row with a non-empty blocker_code must NOT be failed_runtime."""
    found_jsons = [p for p in _matrix_files() if p.suffix == ".json"]
    if not found_jsons:
        pytest.skip("no smoke-matrix JSON files yet")
    for json_path in found_jsons:
        data = json.loads(json_path.read_text())
        rows = data.get("rows", [])
        for r in rows:
            blocker = r.get("blocker_code") or ""
            final_state = r.get("final_state") or ""
            if blocker and final_state == "failed_runtime":
                pytest.fail(
                    f"{json_path}: model {r.get('model_id')!r} has "
                    f"blocker_code={blocker!r} but final_state={final_state!r}"
                )


def test_smoke_matrix_summary_zero_unclassified() -> None:
    found_jsons = [p for p in _matrix_files() if p.suffix == ".json"]
    if not found_jsons:
        pytest.skip("no smoke-matrix JSON files yet")
    for json_path in found_jsons:
        data = json.loads(json_path.read_text())
        s = data.get("summary", {})
        if "unclassified" in s:
            assert s["unclassified"] == 0, f"{json_path}: {s['unclassified']} unclassified rows"

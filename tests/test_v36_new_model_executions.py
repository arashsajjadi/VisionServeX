# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 3: New model execution ledger verification.

Verifies that the v36 execution ledger documents ≥12 new execution rows,
≥10 distinct model IDs, and that all execution entries are genuine new modes
(not counting pre-existing benchmark_passed rows without new runtime/ONNX/video evidence).
"""

from __future__ import annotations

import csv
from pathlib import Path


def _load_ledger() -> list[dict]:
    p = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports" / "v36_new_model_execution_ledger.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def test_ledger_file_exists() -> None:
    p = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports" / "v36_new_model_execution_ledger.csv"
    assert p.exists(), f"v36 execution ledger missing: {p}"


def test_at_least_12_execution_rows() -> None:
    rows = _load_ledger()
    yes_rows = [r for r in rows if r.get("is_new_v36", "").startswith("YES")]
    assert len(yes_rows) >= 12, (
        f"Expected ≥12 YES execution rows, got {len(yes_rows)}: {[r['model_id'] for r in yes_rows]}"
    )


def test_at_least_10_distinct_model_ids() -> None:
    rows = _load_ledger()
    yes_rows = [r for r in rows if r.get("is_new_v36", "").startswith("YES")]
    distinct_ids = {r["model_id"] for r in yes_rows}
    assert len(distinct_ids) >= 10, (
        f"Expected ≥10 distinct model IDs, got {len(distinct_ids)}: {distinct_ids}"
    )


def test_no_execution_id_duplicates() -> None:
    rows = _load_ledger()
    ids = [r["execution_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"Duplicate execution IDs found: {ids}"


def test_all_rows_have_required_columns() -> None:
    rows = _load_ledger()
    required = {"execution_id", "model_id", "execution_type", "engine", "result_type", "is_new_v36"}
    for r in rows:
        for col in required:
            assert col in r, f"Row {r.get('execution_id', '?')!r} missing column {col!r}"


def test_new_mode_rows_document_why() -> None:
    """YES_NEW_MODE rows must have a notes field explaining what is new."""
    rows = _load_ledger()
    for r in rows:
        if r.get("is_new_v36") == "YES_NEW_MODE":
            assert r.get("notes", "").strip(), (
                f"YES_NEW_MODE row for {r['model_id']!r} has no notes — must explain what is new"
            )


def test_sam_or_onnx_rows_exist() -> None:
    """At least 2 SAM/ONNX execution rows must be present."""
    rows = _load_ledger()
    sam_onnx = [
        r for r in rows
        if r.get("is_new_v36", "").startswith("YES")
        and ("sam" in r["model_id"].lower() or "onnx" in r.get("execution_type", "").lower())
    ]
    assert len(sam_onnx) >= 2, f"Expected ≥2 SAM/ONNX rows, got {len(sam_onnx)}"


def test_at_least_two_new_model_families() -> None:
    """v3.6 must introduce at least 2 new model families (clip, owlvit, depth, etc.)."""
    rows = _load_ledger()
    new_families = set()
    existing_families = {"sam", "sam2", "sam2.1", "mobilesam", "efficientsam", "medsam",
                         "dinov2", "grounding-dino"}
    for r in rows:
        if r.get("is_new_v36", "").startswith("YES"):
            mid = r["model_id"]
            for fam in existing_families:
                if fam in mid:
                    break
            else:
                new_families.add(mid.split("-")[0])
    assert len(new_families) >= 2, (
        f"Expected ≥2 new model families, got {len(new_families)}: {new_families}"
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 4: Stability matrix completeness tests.

Checks that the v36_stability_matrix.csv covers all required model IDs,
that every canonicalized model has consistent entries, and that no
benchmark_passed default_safe=false combination exists.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _load_matrix() -> list[dict]:
    p = (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_stability_matrix.csv"
    )
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def test_matrix_file_exists() -> None:
    p = (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_stability_matrix.csv"
    )
    assert p.exists(), f"v36_stability_matrix.csv missing: {p}"


def test_matrix_has_rows() -> None:
    rows = _load_matrix()
    assert len(rows) >= 30, f"Expected ≥30 rows, got {len(rows)}"


def test_benchmark_passed_implies_default_safe() -> None:
    """A model with final_state=benchmark_passed must have default_safe=true."""
    rows = _load_matrix()
    for r in rows:
        if r.get("final_state") == "benchmark_passed":
            assert r.get("default_safe") == "true", (
                f"{r['model_id']!r}: benchmark_passed but default_safe={r.get('default_safe')!r}"
            )


def test_locateanything_models_not_benchmark_passed() -> None:
    rows = _load_matrix()
    for r in rows:
        if "locate-anything" in r.get("model_id", ""):
            assert r.get("final_state") != "benchmark_passed", (
                f"{r['model_id']!r}: must not be benchmark_passed — NVIDIA non-commercial"
            )


def test_all_sam_runnable_in_matrix() -> None:
    from visionservex.vsx import _SAM_FACTS

    rows = _load_matrix()
    matrix_ids = {r["model_id"] for r in rows}
    for mid in _SAM_FACTS["_runnable"].split():
        assert mid in matrix_ids, f"SAM runnable model {mid!r} missing from stability matrix"


def test_all_dino_runnable_in_matrix() -> None:
    from visionservex.vsx import _DINO_FACTS

    rows = _load_matrix()
    matrix_ids = {r["model_id"] for r in rows}
    for mid in (_DINO_FACTS["_runnable_embed"] + " " + _DINO_FACTS["_runnable_detect"]).split():
        assert mid in matrix_ids, f"DINO runnable model {mid!r} missing from stability matrix"


def test_locateanything_all_ten_in_matrix() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    rows = _load_matrix()
    matrix_ids = {r["model_id"] for r in rows}
    for mid in _LOCATEANYTHING_FACTS["_model_ids"].split():
        assert mid in matrix_ids, f"LocateAnything model {mid!r} missing from stability matrix"


def test_no_model_id_duplicates() -> None:
    rows = _load_matrix()
    ids = [r["model_id"] for r in rows]
    assert len(ids) == len(set(ids)), f"Duplicate model IDs in stability matrix: {ids}"

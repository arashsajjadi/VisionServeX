# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 3: Sidecar/checkpoint execution evidence tests.

Verifies that sidecar-required models (medsam2, maskdino, rtdetrv4) and the
LocateAnything sidecar are correctly documented with the expected state,
not incorrectly counted as benchmark_passed.
"""

from __future__ import annotations

import json
from pathlib import Path


def _artifact(name: str) -> Path:
    return Path(__file__).parent.parent / "notebook" / "99_final_report" / "artifacts" / name


def test_medsam2_state_is_sidecar_required() -> None:
    from visionservex.vsx import _SAM_FACTS, VSX

    assert "medsam2" in _SAM_FACTS["_sidecar"]
    h = VSX.sam("medsam2")
    assert h.status() == "sidecar_required"


def test_medsam_state_is_benchmark_passed() -> None:
    """MedSAM (not MedSAM2) via HF is benchmark_passed — executed in v3.5."""
    from visionservex.vsx import _SAM_FACTS, VSX

    assert "medsam" in _SAM_FACTS["_runnable"]
    h = VSX.sam("medsam")
    assert h.status() == "benchmark_passed"


def test_maskdino_sidecar_attempt_artifact_exists() -> None:
    art = _artifact("v35/maskdino_sidecar_attempt.json")
    assert art.exists(), f"MaskDINO sidecar attempt artifact missing: {art}"
    data = json.loads(art.read_text())
    assert (
        data.get("status") in {"expected_blocker", "sidecar_required", "attempt"}
        or "sidecar" in str(data).lower()
    )


def test_rtdetrv4_attempt_artifact_exists() -> None:
    art = _artifact("v35/rtdetrv4_attempt.json")
    assert art.exists(), f"RT-DETRv4 attempt artifact missing: {art}"


def test_locateanything_sidecar_matrix_exists() -> None:
    p = (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_sidecar_checkpoint_matrix.csv"
    )
    assert p.exists(), f"Sidecar checkpoint matrix missing: {p}"


def test_locateanything_sidecar_in_matrix() -> None:
    import csv

    p = (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_sidecar_checkpoint_matrix.csv"
    )
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open()))
    locate_rows = [r for r in rows if "locate-anything" in r.get("model_id", "")]
    assert locate_rows, "locate-anything-3b must appear in sidecar_checkpoint_matrix.csv"
    for r in locate_rows:
        assert r.get("sidecar_type") == "sidecar_required"


def test_three_or_more_sidecar_checkpoint_entries() -> None:
    import csv

    p = (
        Path(__file__).parent.parent
        / "notebook"
        / "99_final_report"
        / "reports"
        / "v36_sidecar_checkpoint_matrix.csv"
    )
    if not p.exists():
        return
    rows = list(csv.DictReader(p.open()))
    assert len(rows) >= 3, f"Expected ≥3 sidecar/checkpoint entries, got {len(rows)}"

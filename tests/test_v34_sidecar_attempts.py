# SPDX-License-Identifier: Apache-2.0
"""v3.4 sidecar-attempt tests.

Verifies that sidecar-required models (MaskDINO, Co-DINO) carry the correct
status in the manifest and model matrix, and that rtdetrv4-s is correctly
flagged as checkpoint_required rather than accidentally promoted.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_MATRIX_CSV = (
    ROOT / "notebook" / "99_final_report" / "reports" / "v33_model_pass_fail_matrix.csv"
)

_SIDECAR_ACTIONS = {"sidecar_required", "expert_sidecar"}


def _matrix_rows() -> list[dict]:
    if not _MATRIX_CSV.exists():
        pytest.skip(f"model matrix CSV not found: {_MATRIX_CSV}")
    return list(csv.DictReader(_MATRIX_CSV.open()))


# ---------------------------------------------------------------------------
# a. MaskDINO sidecar state
# ---------------------------------------------------------------------------


def test_maskdino_sidecar_state() -> None:
    """maskdino-r50-coco must be flagged as sidecar-required in the manifest."""
    from visionservex.model_zoo.manifest import get_model_source

    src = get_model_source("maskdino-r50-coco")
    assert src is not None, "maskdino-r50-coco not found in SOURCE_MANIFEST"

    # recommended_action must be in the sidecar set, OR notes/blockers mention sidecar
    action_ok = src.recommended_action in _SIDECAR_ACTIONS
    notes_text = (src.notes or "").lower() + " ".join(src.known_blockers).lower()
    sidecar_in_text = "sidecar" in notes_text

    assert action_ok or sidecar_in_text, (
        f"maskdino-r50-coco: recommended_action={src.recommended_action!r}, "
        f"notes/blockers do not mention sidecar either. "
        f"Expected recommended_action in {_SIDECAR_ACTIONS} or 'sidecar' in notes/blockers."
    )


# ---------------------------------------------------------------------------
# b. Co-DINO sidecar state
# ---------------------------------------------------------------------------


def test_codino_sidecar_state() -> None:
    """co-dino-inst-vit-l-coco must be flagged as sidecar-required in the manifest."""
    from visionservex.model_zoo.manifest import get_model_source

    src = get_model_source("co-dino-inst-vit-l-coco")
    assert src is not None, "co-dino-inst-vit-l-coco not found in SOURCE_MANIFEST"

    action_ok = src.recommended_action in _SIDECAR_ACTIONS
    notes_text = (src.notes or "").lower() + " ".join(src.known_blockers).lower()
    sidecar_in_text = "sidecar" in notes_text

    assert action_ok or sidecar_in_text, (
        f"co-dino-inst-vit-l-coco: recommended_action={src.recommended_action!r}, "
        f"notes/blockers do not mention sidecar either. "
        f"Expected recommended_action in {_SIDECAR_ACTIONS} or 'sidecar' in notes/blockers."
    )


# ---------------------------------------------------------------------------
# c. RT-DETRv4-s checkpoint_required
# ---------------------------------------------------------------------------


def test_rtdetrv4_checkpoint_required() -> None:
    """rtdetrv4-s must be reported as checkpoint_required in the model matrix."""
    rows = {r["model_id"]: r for r in _matrix_rows()}
    assert "rtdetrv4-s" in rows, (
        "rtdetrv4-s not found in model matrix CSV; "
        "re-run the notebook to regenerate the report"
    )
    final_state = rows["rtdetrv4-s"]["final_state"]
    assert final_state == "checkpoint_required", (
        f"rtdetrv4-s final_state={final_state!r}; expected 'checkpoint_required'"
    )


# ---------------------------------------------------------------------------
# d. MaskDINO / Co-DINO rows must NOT have final_state=benchmark_passed
# ---------------------------------------------------------------------------


def test_sidecar_models_not_accidentally_benchmark_passed() -> None:
    """Neither maskdino nor co-dino rows may claim benchmark_passed in the model matrix."""
    rows = _matrix_rows()
    violations = [
        r
        for r in rows
        if r.get("family", "") in ("maskdino", "co-dino")
        and r.get("final_state", "") == "benchmark_passed"
    ]
    assert not violations, (
        "The following sidecar models are incorrectly marked benchmark_passed: "
        + ", ".join(v["model_id"] for v in violations)
    )

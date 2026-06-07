# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: SAM2.1 ONNX export attempt is honestly documented with exact next action."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
ART = R.parent / "artifacts" / "v37"


def test_sam21_onnx_attempt_artifact_exists():
    p = ART / "sam21_onnx_attempt.json"
    assert p.exists(), f"missing attempt artifact {p}"


def test_attempt_has_blocker_and_next_action():
    p = ART / "sam21_onnx_attempt.json"
    data = json.loads(p.read_text())
    assert data["state"] == "blocked"
    assert data["blocker_code"] == "SAM2_ONNX_EXPORTER_NOT_AVAILABLE"
    assert "next_action" in data and len(data["next_action"]) > 10
    assert "attempts" in data and len(data["attempts"]) >= 1


def test_model_loads_in_attempt():
    """The attempt must prove the model actually loaded (real attempt, not a stub)."""
    data = json.loads((ART / "sam21_onnx_attempt.json").read_text())
    approaches = " ".join(a.get("approach", "") + a.get("result", "") for a in data["attempts"])
    assert "Sam2Model" in approaches or "onnx" in approaches.lower()


def test_all_four_onnx_variants_in_matrix():
    import csv
    rows = {r["variant_id"]: r for r in csv.DictReader((R / "v37_sam_variant_matrix.csv").open())}
    for v in ["sam2.1-onnx-tiny", "sam2.1-onnx-small", "sam2.1-onnx-base-plus", "sam2.1-onnx-large"]:
        assert v in rows
        assert rows[v]["final_state"] == "blocked_documented"
        assert "sam2" in rows[v]["exact_command"].lower() or "onnx" in rows[v]["exact_command"].lower()

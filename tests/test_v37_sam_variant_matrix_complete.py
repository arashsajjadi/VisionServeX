# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: every SAM variant must be decided (one final_state + exact command)."""

from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
M = R / "v37_sam_variant_matrix.csv"

VALID = {
    "benchmark_passed",
    "auth_required",
    "checkpoint_required",
    "sidecar_required",
    "legal_review_required",
    "excluded_restricted",
    "not_released",
    "blocked_documented",
    "external_api_only",
}

REQUIRED_VARIANTS = [
    "sam-vit-b",
    "sam-vit-l",
    "sam-vit-h",
    "sam-vit-b-onnx",
    "sam-vit-l-onnx",
    "sam-vit-h-onnx",
    "mobilesam",
    "mobilesam-onnx",
    "efficientsam-l0",
    "efficientsam-l1",
    "efficientsam-l2",
    "efficientsam-onnx",
    "sam2-hiera-tiny",
    "sam2-hiera-small",
    "sam2-hiera-base-plus",
    "sam2-hiera-large",
    "sam2.1-hiera-tiny",
    "sam2.1-hiera-small",
    "sam2.1-hiera-base-plus",
    "sam2.1-hiera-large",
    "sam2.1-video-tiny",
    "sam2.1-video-small",
    "sam2.1-video-base-plus",
    "sam2.1-video-large",
    "sam2.1-onnx-tiny",
    "sam2.1-onnx-small",
    "sam2.1-onnx-base-plus",
    "sam2.1-onnx-large",
    "medsam",
    "medsam2",
    "hq-sam",
    "hq-sam2",
    "light-hq-sam",
    "tinysam",
    "q-tinysam",
    "edgesam",
    "sam3-base",
    "sam3-image",
    "sam3-video",
    "sam3-text-prompt",
    "sam3-visual-prompt",
    "sam3-exemplar-prompt",
    "sam3-open-vocabulary",
    "sam3-tracking",
    "sam3.1-base",
    "sam3.1-image",
    "sam3.1-video",
    "sam3.1-real-time-tracking",
    "sam3.1-text-prompt",
    "sam3.1-visual-prompt",
    "sam3.1-open-vocabulary",
]


def _rows():
    return {r["variant_id"]: r for r in csv.DictReader(M.open())}


def test_matrix_exists():
    assert M.exists()


def test_every_required_variant_present():
    rows = _rows()
    missing = [v for v in REQUIRED_VARIANTS if v not in rows]
    assert not missing, f"SAM variants missing a decision: {missing}"


def test_every_variant_has_valid_state():
    for vid, r in _rows().items():
        assert r["final_state"] in VALID, f"{vid}: invalid state {r['final_state']!r}"


def test_every_variant_has_exact_command():
    for vid, r in _rows().items():
        assert r["exact_command"].strip(), f"{vid} missing exact_command"


def test_benchmark_passed_have_real_artifacts():
    for vid, r in _rows().items():
        if r["final_state"] == "benchmark_passed" and r["evidence_artifact"]:
            assert r["artifact_exists"] == "True", (
                f"{vid}: artifact missing {r['evidence_artifact']}"
            )


def test_at_least_18_benchmark_passed():
    n = sum(1 for r in _rows().values() if r["final_state"] == "benchmark_passed")
    assert n >= 18, f"expected >=18 executed SAM variants, got {n}"


def test_edgesam_excluded():
    assert _rows()["edgesam"]["final_state"] == "excluded_restricted"


def test_sam3_family_auth_required():
    for vid, r in _rows().items():
        if vid.startswith("sam3"):
            assert r["final_state"] in ("auth_required", "not_released"), (
                f"{vid}: SAM3 must be auth_required/not_released, got {r['final_state']}"
            )
            assert r["commercial_safe"] == "False"

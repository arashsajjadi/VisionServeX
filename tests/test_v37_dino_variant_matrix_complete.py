# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: every DINO variant must be decided."""

from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
M = R / "v37_dino_variant_matrix.csv"

VALID = {
    "benchmark_passed",
    "auth_required",
    "external_api_only",
    "checkpoint_required",
    "legal_review_required",
    "not_released",
}

REQUIRED = [
    "dinov2-vits14",
    "dinov2-vitb14",
    "dinov2-vitl14",
    "dinov2-vitg14",
    "dinov2-small",
    "dinov2-base",
    "dinov2-large",
    "dinov2-giant",
    "dino-vits8",
    "dinov3-vits16",
    "dinov3-vitb16",
    "dinov3-vitl16",
    "dinov3-vit7b16",
    "dinov3-convnext-tiny",
    "dinov3-convnext-small",
    "dinov3-convnext-base",
    "dinov3-convnext-large",
    "grounding-dino-tiny",
    "grounding-dino-swin-t",
    "grounding-dino-swin-b",
    "grounding-dino-original-swin-t",
    "grounding-dino-original-swin-b",
    "grounding-dino-1.5",
    "grounding-dino-1.6",
    "grounding-dino-1.5-pro",
    "grounding-dino-1.6-pro",
    "dino-x-api",
    "dino-x-detection",
    "dino-x-segmentation",
    "dino-x-phrase-grounding",
    "dino-x-counting",
    "dino-x-region-captioning",
]


def _rows():
    return {r["variant_id"]: r for r in csv.DictReader(M.open())}


def test_matrix_exists():
    assert M.exists()


def test_every_required_variant_present():
    rows = _rows()
    missing = [v for v in REQUIRED if v not in rows]
    assert not missing, f"DINO variants missing a decision: {missing}"


def test_every_variant_valid_state():
    for vid, r in _rows().items():
        assert r["final_state"] in VALID, f"{vid}: {r['final_state']!r}"


def test_dinov2_executed():
    rows = _rows()
    for v in ["dinov2-small", "dinov2-base", "dinov2-large", "dinov2-giant"]:
        assert rows[v]["final_state"] == "benchmark_passed", f"{v} should be benchmark_passed"
        assert rows[v]["artifact_exists"] == "True"


def test_dinov3_never_apache_and_gated():
    for vid, r in _rows().items():
        if vid.startswith("dinov3"):
            assert r["final_state"] == "auth_required", f"{vid} must be auth_required (gated)"
            assert "Apache" not in r["license"], f"{vid} must NOT be labeled Apache"
            assert r["commercial_safe"] == "False"


def test_dino_x_external_api():
    for vid, r in _rows().items():
        if vid.startswith("dino-x") or vid.endswith("-pro"):
            assert r["final_state"] == "external_api_only", f"{vid} should be external_api_only"


def test_grounding_dino_executed_variants():
    rows = _rows()
    assert rows["grounding-dino-tiny"]["final_state"] == "benchmark_passed"
    assert rows["grounding-dino-swin-b"]["final_state"] == "benchmark_passed"

# SPDX-License-Identifier: Apache-2.0
"""v3.8 — release readiness: version, matrix artifact, warning texts, no shipped weights."""

from __future__ import annotations

import re
from pathlib import Path

import visionservex
from visionservex.licensing import policy as P

EXPECTED_VERSION = "3.8.0"


def test_package_version_bumped():
    assert visionservex.__version__ == EXPECTED_VERSION


def test_pyproject_version_matches():
    text = Path("pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m and m.group(1) == EXPECTED_VERSION


def test_license_matrix_artifact_exists():
    csv = Path("notebook/99_final_report/reports/v38_license_policy_matrix.csv")
    md = Path("notebook/99_final_report/reports/v38_license_policy_report.md")
    assert csv.exists() and md.exists()
    head = csv.read_text().splitlines()[0]
    assert "final_policy" in head and "can_ship_weights" in head


def test_five_mandatory_warning_texts_present():
    w = P.WARNING_TEXTS
    assert w["byot"].startswith("This model is gated or uses a custom upstream license")
    assert w["noncommercial"].startswith("WARNING: This model is non-commercial/restricted")
    assert w["enterprise"].startswith("WARNING: This model requires an enterprise/commercial license")
    assert w["api"].startswith("External API model. Your data may leave the local environment")
    assert w["legal_review"].startswith("License/provenance is unclear")


def test_all_nine_buckets_defined():
    assert len(P.FINAL_POLICIES) == 9
    counts = {fp: 0 for fp in P.FINAL_POLICIES}
    for r in P.iter_policies():
        counts[r.final_policy] += 1
    # the policy must populate at least the production-relevant buckets
    for fp in ("commercial_safe_core", "byot_license_required",
               "external_api_only_terms_required", "noncommercial_restricted",
               "enterprise_license_required", "legal_review_required"):
        assert counts[fp] > 0


def test_no_weight_binaries_in_package_tree():
    bad = []
    for ext in (".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".engine", ".trt"):
        bad += list(Path("src/visionservex").rglob(f"*{ext}"))
    assert not bad, f"weight binaries inside the package: {bad}"

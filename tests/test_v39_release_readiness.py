# SPDX-License-Identifier: Apache-2.0
"""v3.9 — release readiness: version, access matrix, execution ledger, no weights."""

from __future__ import annotations

import re
from pathlib import Path

import visionservex
from visionservex.licensing import policy as P

MIN_VERSION = (3, 9)


def _ver_tuple(v: str) -> tuple[int, int]:
    return tuple(int(x) for x in v.split(".")[:2])


def test_package_version_bumped():
    assert _ver_tuple(visionservex.__version__) >= MIN_VERSION


def test_pyproject_version_matches():
    text = Path("pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m and _ver_tuple(m.group(1)) >= MIN_VERSION


def test_v39_hf_access_matrix_exists():
    csv_path = Path("notebook/99_final_report/reports/v39_hf_access_matrix.csv")
    assert csv_path.exists(), "Missing v39 HF access matrix CSV"
    head = csv_path.read_text().splitlines()[0]
    assert "repo" in head and "state" in head


def test_v39_dinov3_execution_ledger_has_passed_entries():
    ledger = Path("notebook/99_final_report/reports/v39_dinov3_execution_ledger.csv")
    assert ledger.exists(), "Missing v39 DINOv3 execution ledger"
    text = ledger.read_text()
    assert "benchmark_passed_byot" in text, "No benchmark_passed_byot entries in DINOv3 ledger"


def test_v39_sam3_execution_ledger_has_passed_entries():
    ledger = Path("notebook/99_final_report/reports/v39_sam3_execution_ledger.csv")
    assert ledger.exists(), "Missing v39 SAM3 execution ledger"
    text = ledger.read_text()
    assert "benchmark_passed_byot" in text, "No benchmark_passed_byot entries in SAM3 ledger"


def test_five_mandatory_warning_texts_intact():
    w = P.WARNING_TEXTS
    assert w["byot"].startswith("This model is gated")
    assert w["noncommercial"].startswith("WARNING: This model is non-commercial/restricted")
    assert w["enterprise"].startswith("WARNING: This model requires an enterprise/commercial license")
    assert w["api"].startswith("External API model.")
    assert w["legal_review"].startswith("License/provenance is unclear")


def test_no_weight_binaries_in_src():
    bad = []
    for ext in (".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin"):
        bad += list(Path("src/visionservex").rglob(f"*{ext}"))
    assert not bad, f"Binary weights in package tree: {bad}"


def test_all_nine_buckets_still_populated():
    assert len(P.FINAL_POLICIES) == 9
    counts = dict.fromkeys(P.FINAL_POLICIES, 0)
    for r in P.iter_policies():
        counts[r.final_policy] += 1
    for fp in ("commercial_safe_core", "byot_license_required",
               "external_api_only_terms_required", "noncommercial_restricted",
               "enterprise_license_required"):
        assert counts[fp] > 0, f"Policy bucket '{fp}' has zero entries"

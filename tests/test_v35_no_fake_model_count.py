# SPDX-License-Identifier: Apache-2.0
"""v3.5 anti-fake-count guard tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_sam2_hiera_tiny_actually_ran():
    artifact = _ARTIFACTS / "sam2_hiera_segmentation.json"
    if not artifact.exists():
        pytest.skip(f"SAM2 hiera artifact not in CI env: {artifact}")
    data = json.loads(artifact.read_text())
    assert "sam2-hiera-tiny" in data, "sam2-hiera-tiny missing from artifact"
    assert data["sam2-hiera-tiny"].get("status") == "ok", (
        f"sam2-hiera-tiny not ok: {data['sam2-hiera-tiny']}"
    )


def test_dinov2_large_giant_actually_ran():
    artifact = _ARTIFACTS / "dinov2_lg_embed_results.json"
    if not artifact.exists():
        pytest.skip(f"DINOv2 L/G artifact not in CI env: {artifact}")
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 DINOv2 L/G must succeed, got {ok_count}"


def test_pipeline_v35_actually_ran():
    artifact = _ARTIFACTS / "v35_pipeline_results.json"
    if not artifact.exists():
        pytest.skip(f"v35 pipeline artifact not in CI env: {artifact}")
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 new pipeline must succeed, got {ok_count}"


def test_new_execution_ledger_has_15_rows():
    import csv

    ledger = (
        Path(__file__).parent.parent
        / "notebook/99_final_report/reports/v35_new_model_execution_ledger.csv"
    )
    if not ledger.exists():
        pytest.skip("Ledger CSV not created yet")
    with open(ledger) as f:
        rows = list(csv.DictReader(f))
    new_rows = [r for r in rows if r.get("is_new_v35", "").startswith("YES")]
    assert len(new_rows) >= 15, (
        f"Need >=15 new executions, got {len(new_rows)}: {[r['model_id'] for r in new_rows]}"
    )


def test_version_is_35():
    # v3.5 introduced these executions; later releases supersede the version string.
    # Assert v3.5 *or later* (forward-compatible) rather than pinning an old version.
    import visionservex

    parts = tuple(int(x) for x in visionservex.__version__.split(".")[:2])
    assert parts >= (3, 5), f"Expected >= 3.5, got {visionservex.__version__}"

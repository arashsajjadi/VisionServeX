# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: SAM3 real mask benchmark upgrade — mask_area > 0 state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path("notebook/99_final_report/artifacts/v310/sam3_benchmark")


def test_sam3_benchmark_summary_exists():
    summary = ART / "sam3_benchmark_summary.json"
    if not summary.exists():
        pytest.skip("SAM3 benchmark summary not on disk")
    data = json.loads(summary.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1


def test_sam3_mask_area_gt0():
    summary = ART / "sam3_benchmark_summary.json"
    if not summary.exists():
        pytest.skip("SAM3 benchmark summary not on disk")
    data = json.loads(summary.read_text())
    sam3 = next((r for r in data if r["model_id"] == "sam3"), None)
    if sam3 is None:
        pytest.skip("sam3 entry not in summary")
    assert sam3.get("mask_area_gt0") is True, f"SAM3 mask_area not > 0: {sam3}"


def test_sam31_mask_area_gt0():
    summary = ART / "sam3_benchmark_summary.json"
    if not summary.exists():
        pytest.skip("SAM3 benchmark summary not on disk")
    data = json.loads(summary.read_text())
    sam31 = next((r for r in data if "sam3.1" in r["model_id"]), None)
    if sam31 is None:
        pytest.skip("sam3.1 entry not in summary")
    assert sam31.get("mask_area_gt0") is True, f"SAM3.1 mask_area not > 0: {sam31}"


def test_sam3_benchmark_state_is_mask():
    summary = ART / "sam3_benchmark_summary.json"
    if not summary.exists():
        pytest.skip("SAM3 benchmark summary not on disk")
    data = json.loads(summary.read_text())
    for entry in data:
        assert "byot_mask" in entry.get("benchmark_state", ""), (
            f"{entry['model_id']} still smoke-only: {entry.get('benchmark_state')}"
        )


def test_sam3_artifacts_present():
    for model_dir in ("sam3", "sam3_1_base_plus"):
        d = ART / model_dir
        if not d.exists():
            pytest.skip(f"SAM3 artifact dir {d} not on disk")
        for fname in ("mask.png", "overlay.png", "metadata.json", "latency.json"):
            assert (d / fname).exists(), f"Missing {fname} in {d}"


def test_byot_runtime_sam3_segment_importable():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import sam3_segment

    assert callable(sam3_segment)


def test_byot_runtime_exports_sam3_segment():
    from visionservex import byot_runtime

    assert "sam3_segment" in byot_runtime.__all__

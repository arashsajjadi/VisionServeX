# SPDX-License-Identifier: Apache-2.0
"""v3.4 SAM ONNX export and CPU runtime tests.

Each test is tagged pytest.mark.sam_onnx. Tests that require a local
checkpoint use pytest.skip() when the checkpoint is absent so CI stays
green without heavy model files.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from visionservex.onnx_export import (
    export_sam_decoder_onnx,
    list_onnx_eligible_models,
    run_sam_onnx_cpu,
)

# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
_CACHE = Path.home() / ".cache" / "visionservex"
_SAM_VIT_B_CKPT = _CACHE / "sam" / "sam_vit_b_01ec64.pth"
_SAM_VIT_L_CKPT = _CACHE / "sam" / "sam_vit_l_0b3195.pth"
_MOBILESAM_CKPT = _CACHE / "mobilesam" / "mobile_sam.pt"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.sam_onnx
def test_list_onnx_eligible_models_returns_expected_ids():
    """list_onnx_eligible_models() must include sam-vit-b and mobilesam."""
    result = list_onnx_eligible_models()
    ids = [r["model_id"] for r in result]
    assert "sam-vit-b" in ids, f"sam-vit-b missing from eligible list: {ids}"
    assert "mobilesam" in ids, f"mobilesam missing from eligible list: {ids}"


@pytest.mark.sam_onnx
def test_export_sam_vit_b_when_checkpoint_exists(tmp_path):
    """Export sam-vit-b decoder to ONNX; skip when checkpoint is absent."""
    if not _SAM_VIT_B_CKPT.exists():
        pytest.skip(
            "sam-vit-b checkpoint not cached; run: visionservex pull sam-vit-b"
        )
    out = tmp_path / "sam_vit_b.onnx"
    result = export_sam_decoder_onnx("sam-vit-b", str(out))
    assert out.exists(), "ONNX file was not created"
    assert out.stat().st_size > 1_000_000, (
        f"ONNX file too small: {out.stat().st_size} bytes"
    )
    assert result["size_mb"] > 1.0


@pytest.mark.sam_onnx
def test_run_sam_vit_b_onnx_cpu(tmp_path):
    """Load sam-vit-b ONNX on CPU and assert decoder_latency_ms > 0."""
    if not _SAM_VIT_B_CKPT.exists():
        pytest.skip("sam-vit-b checkpoint not cached")
    out = tmp_path / "sam_vit_b_cpu.onnx"
    export_sam_decoder_onnx("sam-vit-b", str(out))
    rt = run_sam_onnx_cpu(str(out))
    assert rt["status"] == "ok"
    assert rt["decoder_latency_ms"] > 0, "decoder_latency_ms must be positive"
    assert rt["mask_shape"][0] == 1


@pytest.mark.sam_onnx
def test_export_sam_vit_l_checkpoint_required(tmp_path):
    """export_sam_decoder_onnx raises FileNotFoundError when sam-vit-l ckpt is missing."""
    if _SAM_VIT_L_CKPT.exists():
        pytest.skip("sam-vit-l checkpoint is present; skipping absent-check test")
    with pytest.raises(FileNotFoundError):
        export_sam_decoder_onnx("sam-vit-l", str(tmp_path / "l.onnx"))


@pytest.mark.sam_onnx
def test_export_non_eligible_raises_value_error(tmp_path):
    """export_sam_decoder_onnx raises ValueError for a non-eligible model id."""
    with pytest.raises(ValueError, match="not ONNX-eligible"):
        export_sam_decoder_onnx("edgesam", str(tmp_path / "edgesam.onnx"))


@pytest.mark.sam_onnx
def test_mobilesam_export_and_runtime(tmp_path):
    """Export mobilesam to ONNX and run CPU inference; skip if checkpoint absent."""
    if not _MOBILESAM_CKPT.exists():
        pytest.skip(
            "mobilesam checkpoint not cached; run: visionservex pull mobilesam"
        )
    out = tmp_path / "mobilesam.onnx"
    result = export_sam_decoder_onnx("mobilesam", str(out))
    assert out.exists(), "ONNX file was not created"
    assert result["size_mb"] > 1.0
    rt = run_sam_onnx_cpu(str(out))
    assert rt["status"] == "ok"
    assert rt["decoder_latency_ms"] > 0

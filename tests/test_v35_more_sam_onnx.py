# SPDX-License-Identifier: Apache-2.0
"""v3.5 EfficientSAM ONNX and SAM ViT-L/H ONNX tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"
_CKPT_EFF = Path.home() / ".cache/visionservex/efficientsam/efficientvit_sam_l0.pt"


def test_efficientsam_checkpoint_exists():
    if not _CKPT_EFF.exists():
        pytest.skip(f"EfficientSAM checkpoint not in CI env: {_CKPT_EFF}")


def test_efficientsam_onnx_module_importable():
    pytest.importorskip("efficientsam", reason="efficientsam package not installed")
    from efficientsam.segment_anything.utils.onnx import SamOnnxModel

    assert SamOnnxModel is not None


def test_efficientsam_onnx_artifact_created():
    onnx_path = _ARTIFACTS / "efficientsam_l0_decoder.onnx"
    if not onnx_path.exists():
        pytest.skip("EfficientSAM ONNX not yet exported")
    assert onnx_path.stat().st_size > 1_000_000, "ONNX file too small"


def test_efficientsam_onnx_cpu_runtime():
    onnx_path = _ARTIFACTS / "efficientsam_l0_decoder.onnx"
    if not onnx_path.exists():
        pytest.skip("EfficientSAM ONNX artifact missing")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    assert "image_embeddings" in names


def test_efficientsam_onnx_latency_under_1s():
    _ARTIFACTS / "efficientsam_l0_decoder.onnx"
    result_json = _ARTIFACTS / "efficientsam_onnx_result.json"
    if not result_json.exists():
        pytest.skip("EfficientSAM result JSON missing")
    import json

    data = json.loads(result_json.read_text())
    assert data["status"] == "ok", f"EfficientSAM ONNX failed: {data}"
    assert data["decoder_latency_ms"] < 1000, f"decoder too slow: {data['decoder_latency_ms']}ms"


def test_sam_vit_l_onnx_status_checkpoint_required():
    from visionservex.onnx_export import list_onnx_eligible_models

    result = {r["model_id"]: r for r in list_onnx_eligible_models()}
    assert "sam-vit-l" in result
    r = result["sam-vit-l"]
    if not r["checkpoint_exists"]:
        assert r["status"] == "checkpoint_required"


def test_onnx_eligible_list_not_empty():
    from visionservex.onnx_export import list_onnx_eligible_models

    models = list_onnx_eligible_models()
    assert len(models) >= 2
    ids = [r["model_id"] for r in models]
    assert "sam-vit-b" in ids
    assert "mobilesam" in ids

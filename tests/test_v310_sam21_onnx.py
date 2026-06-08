# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: SAM2.1 ONNX export — evidence and runtime tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path("notebook/99_final_report/artifacts/v310")


def test_sam21_onnx_attempt_json_exists():
    f = ART / "sam21_onnx_attempt.json"
    if not f.exists():
        pytest.skip("sam21_onnx_attempt.json not on disk")
    data = json.loads(f.read_text())
    assert "onnx_export_success" in data


def test_sam21_onnx_export_succeeded():
    f = ART / "sam21_onnx_attempt.json"
    if not f.exists():
        pytest.skip("sam21_onnx_attempt.json not on disk")
    data = json.loads(f.read_text())
    assert data["onnx_export_success"] is True, f"ONNX export failed: {data.get('blocker_detail')}"


def test_sam21_onnx_state_is_passed():
    f = ART / "sam21_onnx_attempt.json"
    if not f.exists():
        pytest.skip("sam21_onnx_attempt.json not on disk")
    data = json.loads(f.read_text())
    assert data.get("benchmark_state") == "benchmark_passed_byot_onnx"


def test_sam21_onnx_file_exists_and_nonempty():
    onnx_path = ART / "sam21_hiera_base_plus_encoder.onnx"
    if not onnx_path.exists():
        pytest.skip("sam21 ONNX file not on disk")
    assert onnx_path.stat().st_size > 1000, "ONNX file suspiciously small"


def test_sam21_onnx_runnable_with_ort():
    pytest.importorskip("onnxruntime")
    onnx_path = ART / "sam21_hiera_base_plus_encoder.onnx"
    if not onnx_path.exists():
        pytest.skip("sam21 ONNX file not on disk")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    pixel = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    result = sess.run(None, {inp.name: pixel})
    # Should return image_embeddings + multi-scale features
    assert len(result) >= 1
    assert result[0].shape[0] == 1  # batch dimension preserved


def test_sam21_onnx_ort_infer_ms_reasonable():
    f = ART / "sam21_onnx_attempt.json"
    if not f.exists():
        pytest.skip("sam21_onnx_attempt.json not on disk")
    data = json.loads(f.read_text())
    ort_ms = data.get("ort_infer_ms")
    if ort_ms is None:
        pytest.skip("ort_infer_ms not recorded")
    assert ort_ms > 0, "ORT infer_ms must be positive"
    assert ort_ms < 60000, f"ORT infer_ms suspiciously high: {ort_ms}"

# SPDX-License-Identifier: Apache-2.0
"""v3.5 sidecar attempt documentation tests."""
from __future__ import annotations
from pathlib import Path
import pytest

_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_maskdino_sidecar_attempt_documented():
    artifact = _ARTIFACTS / "maskdino_sidecar_attempt.json"
    assert artifact.exists(), "MaskDINO sidecar attempt not documented"
    import json
    data = json.loads(artifact.read_text())
    assert data.get("model") == "maskdino-r50-coco"
    assert data.get("sidecar_type") == "detectron2"
    assert "blocker" in data or data.get("detectron2_available") is False


def test_rtdetrv4_checkpoint_documented():
    artifact = _ARTIFACTS / "rtdetrv4_attempt.json"
    assert artifact.exists(), "RT-DETRv4 attempt not documented"
    import json
    data = json.loads(artifact.read_text())
    assert data.get("ckpt_exists") is True, "RT-DETRv4-s checkpoint should exist"
    assert data.get("final_status") in ("checkpoint_only", "ok")


def test_medsam_execution_documented():
    artifact = _ARTIFACTS / "medsam2_result.json"
    assert artifact.exists(), "MedSAM execution not documented"
    import json
    data = json.loads(artifact.read_text())
    # medsam via VisionModel should have succeeded
    medsam_key = "medsam_visionservex"
    assert medsam_key in data, f"medsam_visionservex key missing from {list(data.keys())}"
    assert data[medsam_key].get("status") == "ok", f"MedSAM failed: {data[medsam_key]}"


def test_sam_vit_hf_execution_documented():
    artifact = _ARTIFACTS / "sam_vit_hf_results.json"
    assert artifact.exists(), "SAM ViT HF results not documented"
    import json
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 SAM ViT HF model should work, got {ok_count}"

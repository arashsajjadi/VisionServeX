# SPDX-License-Identifier: Apache-2.0
"""v3.5 new pipeline execution tests."""
from __future__ import annotations
from pathlib import Path
import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"
_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_v35_pipeline_artifact_exists():
    artifact = _ARTIFACTS / "v35_pipeline_results.json"
    assert artifact.exists(), "v35 pipeline artifact missing"
    import json
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 new pipeline must succeed, got {ok_count}"


def test_gd_swin_t_sam2_hiera_pipeline():
    artifact = _ARTIFACTS / "v35_pipeline_results.json"
    if not artifact.exists():
        pytest.skip("pipeline artifact missing")
    import json
    data = json.loads(artifact.read_text())
    key = "grounding-dino-swin-t+sam2-hiera-tiny"
    if key not in data:
        pytest.skip(f"{key} not in artifact")
    r = data[key]
    assert r.get("status") == "ok", f"Pipeline failed: {r}"
    assert r.get("n_segments", 0) >= 1


def test_gd_swin_b_sam21_pipeline():
    artifact = _ARTIFACTS / "v35_pipeline_results.json"
    if not artifact.exists():
        pytest.skip("pipeline artifact missing")
    import json
    data = json.loads(artifact.read_text())
    key = "grounding-dino-swin-b+sam2.1-hiera-small"
    if key not in data:
        pytest.skip(f"{key} not in artifact")
    r = data[key]
    assert r.get("status") == "ok", f"Pipeline failed: {r}"
    assert r.get("n_segments", 0) >= 1


def test_pipeline_gd_sam2_runs_live():
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image
    from visionservex.core.model import VisionModel
    img = Image.open(str(_IMG)).convert("RGB")
    det = VisionModel("grounding-dino-swin-t")
    det_result = det.predict(img, text="person . car")
    assert det_result is not None
    seg = VisionModel("sam2-hiera-tiny")
    seg_result = seg.predict(img, boxes=[[50, 50, 300, 300]])
    assert seg_result is not None

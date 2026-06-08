# SPDX-License-Identifier: Apache-2.0
"""v3.4 SAM+DINO pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"

_11_PIPELINES = [
    "grounding-dino-swin-t+sam-vit-base",
    "grounding-dino-swin-b+sam-vit-base",
    "grounding-dino-swin-t+sam2.1-hiera-small",
    "grounding-dino-swin-b+sam2.1-hiera-large",
    "grounding-dino-original-swin-t+sam2-hiera-small",
    "grounding-dino-original-swin-b+sam2-hiera-large",
    "grounding-dino-1.5+sam3-base",
    "grounding-dino-1.6+sam3-base",
    "dino-x-api+sam3-base",
    "dinov3-vitb16+sam2.1-hiera-small",
    "dinov3-vitb16+sam3-base",
]


def test_pipeline_demo_ready_status():
    """PipelineHandle for a fully-runnable pair reports pipeline_demo_ready state."""
    from visionservex.vsx import VSX

    info = VSX.pipeline("grounding-dino-swin-t+sam-vit-base").explain()
    assert isinstance(info, dict), "explain() must return a dict"
    assert "state" in info, "explain() dict must contain 'state' key"
    assert info["state"] == "pipeline_demo_ready"


def test_pipeline_auth_required_gated():
    """Pipeline with both components blocked reflects auth_required state."""
    from visionservex.vsx import VSX

    info = VSX.pipeline("grounding-dino-1.5+sam3-base").explain()
    assert isinstance(info, dict), "explain() must return a dict"
    assert info["state"] == "auth_required", f"expected auth_required, got {info['state']!r}"


def test_pipeline_handle_has_explain():
    """PipelineHandle returned by VSX.pipeline() has an explain() method."""
    from visionservex.vsx import VSX

    pipe = VSX.pipeline("grounding-dino-swin-t+sam-vit-base")
    assert hasattr(pipe, "explain"), "PipelineHandle must have explain()"
    assert hasattr(pipe, "run"), "PipelineHandle must have run()"


def test_grounding_dino_sam_demo_runs():
    """Run grounding-dino-swin-t to detect, then sam-vit-b to segment — no exceptions."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image

    from visionservex import VisionModel

    img = Image.open(str(_IMG)).convert("RGB")

    det = VisionModel("grounding-dino-swin-t")
    det_result = det.predict(img, text="person . car")
    assert det_result is not None

    seg = VisionModel("sam-vit-base")
    seg_result = seg.predict(img, boxes=[[50, 50, 300, 400]])
    assert seg_result is not None


def test_pipeline_parse_id():
    """PipelineHandle('a+b') correctly parses detector_id and segmenter_id."""
    from visionservex._v34_handles import PipelineHandle

    handle = PipelineHandle("a+b")
    assert handle.detector_id == "a"
    assert handle.segmenter_id == "b"


def test_all_11_pipelines_recognized():
    """For each of the 11 pipeline IDs, VSX.pipeline(pid).explain() returns a state."""
    from visionservex.vsx import VSX

    for pid in _11_PIPELINES:
        info = VSX.pipeline(pid).explain()
        assert isinstance(info, dict), (
            f"pipeline {pid!r}: explain() must return a dict, got {type(info)}"
        )
        assert "state" in info, f"pipeline {pid!r}: missing 'state' key"

"""Predictable result schema tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from visionservex import VisionModel
from visionservex.core.results import (
    Box,
    Detection,
    DetectionResult,
    Segment,
    SegmentationResult,
)


def _img() -> Image.Image:
    return Image.new("RGB", (64, 48), color="white")


@pytest.mark.parametrize(
    "model_id,result_kind",
    [
        ("mock-detect", "detection"),
        ("mock-segment", "segmentation"),
        ("mock-pose", "pose"),
        ("mock-classify", "classification"),
        ("mock-obb", "obb"),
        ("mock-open-vocab", "open_vocab"),
    ],
)
def test_mock_engine_returns_expected_kind(model_id, result_kind):
    m = VisionModel(model_id)
    r = m.predict(_img(), prompts=["thing"])
    assert r.kind == result_kind
    j = json.loads(r.to_json())
    assert j["kind"] == result_kind
    assert j["model_id"] == model_id
    assert "latency_ms" in j


def test_detection_to_coco():
    r = DetectionResult(
        model_id="x",
        task="detect",
        image_size=(10, 10),
        detections=[Detection(box=Box(0, 0, 5, 5), score=0.9, label="cat", class_id=0)],
    )
    coco = r.to_coco()
    assert coco["annotations"][0]["bbox"] == [0, 0, 5, 5]
    assert coco["annotations"][0]["label"] == "cat"


def test_segmentation_to_dict_strips_mask():
    mask = np.ones((4, 4), dtype=np.uint8)
    r = SegmentationResult(
        model_id="x",
        task="segment",
        image_size=(4, 4),
        segments=[Segment(box=Box(0, 0, 4, 4), score=0.9, label="z", mask=mask)],
    )
    d = r.to_dict()
    assert "mask" not in d["segments"][0]
    assert d["segments"][0]["mask_shape"] == [4, 4]
    assert d["segments"][0]["mask_pixels_on"] == 16


def test_result_plot_does_not_crash():
    m = VisionModel("mock-detect")
    img = _img()
    r = m.predict(img)
    canvas = r.plot()
    assert canvas.size == img.size


def test_result_save_json(tmp_path):
    m = VisionModel("mock-detect")
    out = tmp_path / "r.json"
    r = m.predict(_img())
    p = r.save(out)
    assert p.exists()
    assert json.loads(p.read_text())["kind"] == "detection"

# SPDX-License-Identifier: Apache-2.0
"""v3.18 detection output-schema contract (weight-free, via mock engines)."""

from __future__ import annotations

from PIL import Image

from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities
from visionservex.core.results import DetectionResult, OpenVocabularyResult

_IMG = Image.new("RGB", (64, 64), (120, 130, 140))


def test_mock_detect_returns_well_formed_detection_result():
    r = VisionModel("mock-detect").predict(_IMG)
    assert isinstance(r, DetectionResult)
    assert hasattr(r, "detections") and isinstance(r.detections, list)
    for d in r.detections:
        assert hasattr(d, "box") and len(d.box.to_xyxy()) == 4
        assert hasattr(d, "label")
        assert hasattr(d, "score") and 0.0 <= float(d.score) <= 1.0
        assert hasattr(d, "class_id")
    # result carries provenance metadata
    assert r.model_id == "mock-detect"
    assert r.task == "detect"


def test_mock_open_vocab_returns_open_vocabulary_result():
    r = VisionModel("mock-open-vocab").detect(_IMG, prompts=["cat", "dog"])
    assert isinstance(r, OpenVocabularyResult)
    for d in r.detections:
        assert len(d.box.to_xyxy()) == 4
        assert 0.0 <= float(d.score) <= 1.0


def test_detection_models_declare_detect_syntax():
    for mid in list_models():
        cap = model_capabilities(mid)
        if cap["task"] in ("detect", "obb", "open_vocab_detect"):
            assert "detect" in cap["validated_syntax"], mid

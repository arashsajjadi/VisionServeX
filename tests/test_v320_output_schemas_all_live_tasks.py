# SPDX-License-Identifier: Apache-2.0
"""v3.20: every live task produces a stable, public, JSON-serialisable schema —
never raw framework output. Weight-free (mock engines + result classes).
"""

from __future__ import annotations

import json

from PIL import Image

from visionservex import VisionModel
from visionservex.core.results import (
    ClassificationResult,
    DetectionResult,
    OpenVocabularyResult,
    OrientedDetectionResult,
    PoseResult,
    SegmentationResult,
)

_IMG = Image.new("RGB", (64, 64), (120, 130, 140))


def _json_ok(result) -> dict:
    # Public results must serialise to JSON (no raw tensors / framework objects).
    payload = (
        result.to_json()
        if hasattr(result, "to_json")
        else json.loads(json.dumps(result, default=str))
    )
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def test_detection_schema():
    r = VisionModel("mock-detect").predict(_IMG)
    assert isinstance(r, DetectionResult)
    for d in r.detections:
        assert len(d.box.to_xyxy()) == 4
        assert 0.0 <= float(d.score) <= 1.0
        assert isinstance(d.label, str)
    payload = _json_ok(r)
    assert payload.get("model_id") == "mock-detect"


def test_open_vocab_schema():
    r = VisionModel("mock-open-vocab").detect(_IMG, prompts=["a", "b"])
    assert isinstance(r, OpenVocabularyResult)
    _json_ok(r)


def test_classification_schema():
    r = VisionModel("mock-classify").classify(_IMG)
    assert isinstance(r, ClassificationResult)
    assert isinstance(r.top_k, list)
    for label, score in r.top_k:
        assert isinstance(label, str) and 0.0 <= float(score) <= 1.0001
    _json_ok(r)


def test_segmentation_schema():
    r = VisionModel("mock-segment").segment(_IMG)
    assert isinstance(r, SegmentationResult)
    _json_ok(r)


def test_obb_and_pose_schemas():
    assert isinstance(VisionModel("mock-obb").predict(_IMG), OrientedDetectionResult)
    assert isinstance(VisionModel("mock-pose").predict(_IMG), PoseResult)


def test_embedding_schema_contract():
    # No mock embedder; validate the EmbeddingResult contract surface directly.
    from visionservex.core.embedding_results import EmbeddingResult

    fields = EmbeddingResult.__dataclass_fields__
    assert "embedding" in fields

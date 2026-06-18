# SPDX-License-Identifier: Apache-2.0
"""v3.18 segmentation contract (weight-free; mock engine for schema)."""

from __future__ import annotations

from PIL import Image

from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities
from visionservex.core.results import SegmentationResult
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
SEG = {
    m: c
    for m, c in CAPS.items()
    if c["task"] in ("segment", "foundation_segment", "grounded_segment")
}
_IMG = Image.new("RGB", (64, 64), (200, 50, 50))


def test_there_are_segmentation_models():
    assert SEG


def test_mock_segment_returns_segmentation_result():
    r = VisionModel("mock-segment").segment(_IMG)
    assert isinstance(r, SegmentationResult)
    assert r.task in ("segment", "foundation_segment", "grounded_segment")


def test_mock_foundation_segment_accepts_point_prompt():
    r = VisionModel("mock-foundation-segment").segment(
        _IMG, points=[[32.0, 32.0]], point_labels=[1]
    )
    assert isinstance(r, SegmentationResult)


def test_segmentation_models_expose_segment_syntax():
    for mid, cap in SEG.items():
        assert "segment" in cap["validated_syntax"], mid


def test_live_segmentation_models_use_segmentation_state():
    # A pure (non-trainable) segmentation model whose inference is live-verified
    # earns SEGMENTATION_READY_LIVE. Trainable segmentors (e.g. RF-DETR-seg)
    # headline their TRAIN state instead, but are still shown for inference.
    for mid, cap in SEG.items():
        if (
            cap["live_verified_inference"]
            and not cap["legal_review_required"]
            and not cap["train_supported"]
        ):
            assert cap["readiness_state"] == taxonomy.SEGMENTATION_READY_LIVE, mid
            assert cap["anastig_visibility"] == "show_segmentation", mid


def test_live_trainable_segmentors_are_shown_for_inference():
    for mid, cap in SEG.items():
        if (
            cap["live_verified_inference"]
            and cap["train_supported"]
            and not cap["legal_review_required"]
        ):
            assert cap["anastig_visibility"] in (
                "show_segmentation",
                "show_inference",
                "show_train",
            ), (mid, cap["anastig_visibility"])

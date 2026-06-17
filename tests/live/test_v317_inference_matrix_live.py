# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0 LIVE: real inference smoke across tasks (gated, commercial-safe only).

VSX_LIVE_INFERENCE_MATRIX=1 pytest tests/live/test_v317_inference_matrix_live.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_INFERENCE_MATRIX") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_INFERENCE_MATRIX=1")
DEVICE = os.environ.get("VSX_LIVE_DEVICE", "cpu")


def _img():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), (120, 140, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 120, 220, 400], fill=(200, 80, 60))
    return img


def test_live_detect():
    from visionservex.core.model import VisionModel

    m = VisionModel("libreyolo-yolox-s", device=DEVICE)
    res = m.predict(_img(), threshold=0.1)
    assert res.kind == "detection"
    assert res.metadata["post_nms_count"] <= res.metadata["raw_count"]
    m.unload()


def test_live_classify():
    from visionservex.core.model import VisionModel

    m = VisionModel("torchvision-resnet50", device=DEVICE)
    res = m.classify(_img(), top_k=5)
    assert res.kind == "classification" and len(res.top_k) == 5
    m.unload()


@pytest.mark.skipif(
    os.environ.get("VSX_LIVE_HEAVY") != "1", reason="set VSX_LIVE_HEAVY=1 (downloads ~330MB)"
)
def test_live_embed_and_similarity():
    from visionservex.core.model import VisionModel

    m = VisionModel("dinov2-base", device=DEVICE)
    a = m.embed(_img())
    b = m.embed(_img())
    assert a.kind == "embedding" and a.embedding.size > 0
    assert abs(m.similarity(a, b) - 1.0) < 1e-3  # identical image -> ~1.0
    m.unload()

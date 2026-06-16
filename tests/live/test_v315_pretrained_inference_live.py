# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0 LIVE: pretrained inference across newly-covered families (gated).

VSX_LIVE_PRETRAINED=1 pytest tests/live/test_v315_pretrained_inference_live.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_PRETRAINED") == "1"
pytestmark = pytest.mark.skipif(
    not LIVE, reason="set VSX_LIVE_PRETRAINED=1 to run live pretrained inference"
)


@pytest.mark.parametrize(
    "model_id",
    [
        "torchvision-alexnet",
        "torchvision-resnet50",
        "torchvision-densenet121",
        "torchvision-mobilenet-v3-large",
        "torchvision-efficientnet-b0",
        "torchvision-convnext-tiny",
    ],
)
def test_live_pretrained_classify(model_id):
    from PIL import Image

    from visionservex.core.model import VisionModel

    m = VisionModel(model_id)
    res = m.predict(Image.new("RGB", (256, 256), (120, 90, 60)), top_k=5)
    assert res.kind == "classification"
    assert len(res.top_k) == 5
    assert all(isinstance(label, str) and 0.0 <= score <= 1.0 for label, score in res.top_k)
    m.unload()

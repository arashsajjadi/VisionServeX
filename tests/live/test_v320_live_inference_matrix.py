# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 LIVE: representative inference re-verification (env-gated).

VSX_LIVE_INFERENCE_MATRIX=1 pytest tests/live/test_v320_live_inference_matrix.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_INFERENCE_MATRIX") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_INFERENCE_MATRIX=1"),
    pytest.mark.real_model,
]


@pytest.mark.parametrize(
    "mid,method",
    [
        ("libreyolo-yolox-s", "detect"),
        ("dinov2-small", "embed"),
        ("maxvit-tiny-tf-224", "classify"),
        ("sam-vit-base", "segment"),
    ],
)
def test_live_inference(mid, method):
    from PIL import Image

    from visionservex import VisionModel
    from visionservex.core.results import BaseResult

    m = VisionModel(mid, device="cpu")
    img = Image.new("RGB", (256, 256), (140, 150, 160))
    if method == "segment":
        r = m.segment(img, points=[[128.0, 128.0]], point_labels=[1])
    else:
        r = getattr(m, method)(img)
    assert isinstance(r, BaseResult)

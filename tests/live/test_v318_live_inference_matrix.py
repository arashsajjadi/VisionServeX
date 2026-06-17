# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.18 LIVE: re-verify representative live-inference PASS rows.

    VSX_LIVE_INFERENCE_MATRIX=1 pytest tests/live/test_v318_live_inference_matrix.py -q

The full 105-model matrix lives in tools/qa/v318_live_inference_matrix.py; this
gated test re-runs a small representative subset and cross-checks the committed
matrix against the baked live-evidence frozenset.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_INFERENCE_MATRIX") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_INFERENCE_MATRIX=1"),
    pytest.mark.real_model,
]

MATRIX = Path("docs/qa/v318_full_model_truth/live_inference_matrix.json")


def _img():
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 320), (160, 180, 200))
    ImageDraw.Draw(img).rectangle([40, 40, 180, 220], fill=(200, 70, 60))
    return img


@pytest.mark.parametrize(
    "mid,method",
    [
        ("libreyolo-yolox-s", "detect"),
        ("dinov2-small", "embed"),
        ("torchvision-resnet50", "classify"),
    ],
)
def test_live_inference_subset(mid, method):
    from visionservex import VisionModel
    from visionservex.core.results import BaseResult

    m = VisionModel(mid, device="cpu")
    r = getattr(m, method)(_img())
    assert isinstance(r, BaseResult)


def test_committed_matrix_matches_baked_evidence():
    from visionservex.readiness import live_evidence

    assert MATRIX.exists(), "run tools/qa/v318_live_inference_matrix.py first"
    rows = json.loads(MATRIX.read_text())["results"]
    passed = {r["model_id"] for r in rows if r["status"] == "PASS"}
    assert passed == set(live_evidence.LIVE_INFERENCE_VERIFIED)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.19 LIVE: re-verify v3.19-operationalized inference (maxvit). Env-gated.

VSX_LIVE_INFERENCE_MATRIX=1 pytest tests/live/test_v319_live_inference_matrix.py -q
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

MATRIX = Path("docs/qa/v319_operationalize_all_models/v319_inference_matrix.json")


def test_maxvit_live_classify():
    from PIL import Image

    from visionservex import VisionModel
    from visionservex.core.results import ClassificationResult

    r = VisionModel("maxvit-tiny-tf-224", device="cpu").classify(
        Image.new("RGB", (224, 224)), top_k=5
    )
    assert isinstance(r, ClassificationResult)


def test_v319_matrix_pass_models_are_inference_live():
    from visionservex.core.model import model_capabilities

    if not MATRIX.exists():
        pytest.skip("v319 inference matrix not generated")
    for r in json.loads(MATRIX.read_text())["results"]:
        if r["status"] == "PASS":
            assert model_capabilities(r["model_id"])["live_verified_inference"]

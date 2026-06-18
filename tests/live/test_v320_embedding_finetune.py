# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 LIVE: embedding head fine-tune lifecycle (env-gated).

VSX_LIVE_EMBED_FINETUNE=1 pytest tests/live/test_v320_embedding_finetune.py -q
"""

from __future__ import annotations

import os

import pytest

LIVE = os.environ.get("VSX_LIVE_EMBED_FINETUNE") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_EMBED_FINETUNE=1"),
    pytest.mark.real_model,
]


def test_dinov2_small_head_finetune_lifecycle(tmp_path):
    from PIL import Image, ImageDraw

    from visionservex.core.results import ClassificationResult
    from visionservex.training import EmbeddingHeadModel, finetune_embedding_head

    for cname, col in (("warm", (200, 70, 50)), ("cool", (50, 90, 200))):
        (tmp_path / cname).mkdir()
        for i in range(6):
            img = Image.new("RGB", (96, 96), col)
            ImageDraw.Draw(img).ellipse([10, 10, 86, 86], fill=col)
            img.save(tmp_path / cname / f"{i}.jpg")

    res = finetune_embedding_head(
        "dinov2-small", tmp_path, epochs=40, device="cpu", output_dir=str(tmp_path / "run")
    )
    assert res["status"] == "OK"
    reloaded = EmbeddingHeadModel.from_checkpoint(res["checkpoint"], device="cpu")
    val = next((tmp_path).rglob("*.jpg"))
    cls = reloaded.classify(str(val), top_k=2)
    assert isinstance(cls, ClassificationResult) and cls.top_k
    sim = reloaded.similarity(reloaded.embed(str(val)), reloaded.embed(str(val)))
    assert 0.99 <= float(sim) <= 1.0001

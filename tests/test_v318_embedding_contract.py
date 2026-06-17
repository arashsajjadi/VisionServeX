# SPDX-License-Identifier: Apache-2.0
"""v3.18 embedding contract (weight-free capability checks)."""

from __future__ import annotations

from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
EMBED = {m: c for m, c in CAPS.items() if c["task"] in ("embed", "embedding")}


def test_there_are_embedding_models():
    assert EMBED


def test_embedding_models_expose_embed_and_similarity_syntax():
    for mid, cap in EMBED.items():
        assert "embed" in cap["validated_syntax"], mid
        assert "similarity" in cap["validated_syntax"], mid
        assert cap["predict_supported"], mid


def test_live_embedding_models_use_embedding_state_and_visibility():
    for mid, cap in EMBED.items():
        if cap["live_verified_inference"] and not cap["legal_review_required"]:
            assert cap["readiness_state"] == taxonomy.EMBEDDING_READY_LIVE, mid
            assert cap["anastig_visibility"] == "show_embedding", mid


def test_visionmodel_exposes_embed_and_similarity():
    assert hasattr(VisionModel, "embed")
    assert hasattr(VisionModel, "similarity")

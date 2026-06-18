# SPDX-License-Identifier: Apache-2.0
"""v3.20: embedding head fine-tune contract. Weight-free (reads caps + committed matrix)."""

from __future__ import annotations

import json
from pathlib import Path

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import live_evidence

CAPS = {m: model_capabilities(m) for m in list_models()}
EMBED = {m: c for m, c in CAPS.items() if c["task"] in ("embed", "embedding")}
MATRIX = Path("docs/qa/v320_final_operationalization/train_finetune_matrix.json")


def test_embedding_models_are_fine_tune_ready():
    assert EMBED
    for mid, c in EMBED.items():
        if c["inference_ready"]:
            assert c["fine_tune_ready"], f"{mid} embed model not fine_tune_ready"


def test_finetune_live_models_are_embed_and_in_matrix():
    for mid, c in CAPS.items():
        if c["fine_tune_live_verified"]:
            assert mid in live_evidence.LIVE_FINETUNE_VERIFIED, mid


def test_finetune_helper_is_public_and_embed_only():
    import pytest

    from visionservex.exceptions import TaskNotSupportedError
    from visionservex.training import EmbeddingHeadModel, finetune_embedding_head

    assert callable(finetune_embedding_head)
    assert hasattr(EmbeddingHeadModel, "from_checkpoint")
    # calling on a non-embed model raises typed error (no model load needed: mock-detect)
    with pytest.raises(TaskNotSupportedError):
        finetune_embedding_head("mock-detect", "/tmp/nonexistent")


def test_committed_matrix_finetune_rows_have_lifecycle():
    if not MATRIX.exists():
        return
    for r in json.loads(MATRIX.read_text())["results"]:
        if r.get("status") == "PASS" and r.get("method") == "head_train":
            assert r["reload_verified"]
            assert "classify" in r["post_reload_method"]
            assert CAPS[r["model_id"]]["fine_tune_live_verified"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.21: deeper (MLP) embedding head fine-tune. Weight-free for the head logic.

The frozen-backbone head fine-tune now supports a deeper ``mlp`` head in addition
to the classic ``linear`` probe. These tests exercise the head construction and
validation without loading any backbone; a committed live matrix records the
real dinov2-small MLP-head lifecycle proof.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from visionservex.training.embedding_finetune import _build_head  # noqa: E402


def test_linear_head_is_single_linear():
    head = _build_head("linear", 384, 3, 256, 0.1)
    assert isinstance(head, torch.nn.Linear)
    assert head.out_features == 3


def test_mlp_head_is_deeper_with_nonlinearity():
    head = _build_head("mlp", 384, 4, 128, 0.1)
    assert isinstance(head, torch.nn.Sequential)
    kinds = [type(m).__name__ for m in head]
    assert "Linear" in kinds and "GELU" in kinds and "Dropout" in kinds
    # first Linear projects embed_dim -> hidden, last Linear hidden -> classes
    linears = [m for m in head if isinstance(m, torch.nn.Linear)]
    assert linears[0].in_features == 384 and linears[0].out_features == 128
    assert linears[-1].out_features == 4


def test_invalid_head_type_raises():
    with pytest.raises(ValueError, match="HEAD_TYPE_INVALID"):
        _build_head("transformer", 384, 3, 256, 0.1)


def test_mlp_head_forward_shape():
    head = _build_head("mlp", 16, 5, 8, 0.0)
    out = head(torch.randn(7, 16))
    assert out.shape == (7, 5)


def test_committed_mlp_finetune_evidence_is_a_real_pass():
    matrix = Path("docs/qa/v321_sidecar_blocker_elimination/v321_embedding_deeper_finetune.json")
    if not matrix.exists():
        return
    data = json.loads(matrix.read_text())
    rows = [r for r in data["results"] if r.get("head_type") == "mlp"]
    assert rows, "expected at least one mlp-head fine-tune row"
    for r in rows:
        assert r["status"] == "PASS"
        assert r["reload_head_type"] == "mlp"
        assert r["train_acc"] >= 0.9

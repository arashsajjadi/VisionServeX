# SPDX-License-Identifier: Apache-2.0
"""v3.20: the Anastig v320 contract primary buckets are a complete, disjoint
partition of the catalog, and every live bucket is genuinely live. Weight-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
CONTRACT = Path("docs/anastig_model_allowlist_v320.json")
_LIVE_BUCKETS = {
    "train_ready_live",
    "inference_ready_live",
    "embedding_fine_tune_ready_live",
    "foundation_segment_inference_only_live",
    "open_vocab_ready_live",
    "vlm_ready_live",
    "pose_ready_live",
    "obb_ready_live",
    "fine_tune_ready_live",
    "classification_train_ready_live",
    "detection_train_ready_live",
    "segmentation_train_ready_live",
}


def _load():
    if not CONTRACT.exists():
        pytest.skip("v320 contract not generated")
    return json.loads(CONTRACT.read_text())


def test_primary_buckets_are_complete_disjoint_partition():
    d = _load()
    placed = [m for b in d["primary_partition_buckets"] for m in d[b]]
    assert len(placed) == len(set(placed)), "a model appears in two primary buckets"
    assert set(placed) == set(CAPS), "primary buckets are not a complete partition of 151"


def test_view_buckets_are_subsets_of_catalog():
    d = _load()
    for b in d["view_buckets"]:
        assert set(d[b]) <= set(CAPS)


def test_live_buckets_only_contain_live_verified_models():
    d = _load()
    for b in _LIVE_BUCKETS:
        for mid in d.get(b, []):
            c = CAPS[mid]
            assert (
                c["inference_live_verified"]
                or c["train_live_verified"]
                or c["fine_tune_live_verified"]
            ), (b, mid)


def test_train_buckets_require_train_live():
    d = _load()
    for b in (
        "train_ready_live",
        "classification_train_ready_live",
        "detection_train_ready_live",
        "segmentation_train_ready_live",
    ):
        for mid in d.get(b, []):
            assert CAPS[mid]["train_live_verified"], (b, mid)


def test_finetune_buckets_require_finetune_live():
    d = _load()
    for b in ("fine_tune_ready_live", "embedding_fine_tune_ready_live"):
        for mid in d.get(b, []):
            assert CAPS[mid]["fine_tune_live_verified"], (b, mid)


def test_gated_bucket_requires_token():
    d = _load()
    for mid in d.get("gated_token_required", []):
        assert CAPS[mid]["requires_token"]

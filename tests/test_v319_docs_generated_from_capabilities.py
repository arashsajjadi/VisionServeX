# SPDX-License-Identifier: Apache-2.0
"""v3.19: the Anastig v319 contract is a faithful projection of capabilities.

Weight-free — guards against hand-edited drift between the JSON contract and
``model_capabilities()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
ALLOWLIST = Path("docs/anastig_model_allowlist_v319.json")
FINAL = Path("docs/qa/v319_operationalize_all_models/final_model_matrix.json")

_STATE_TO_BUCKET = {
    "TRAIN_READY_LIVE": "train_ready_live",
    "TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION": "train_ready_derived_admin_only",
    "INFERENCE_READY_LIVE": "inference_ready_live",
    "VLM_READY_LIVE": "inference_ready_live",
    "SEGMENTATION_READY_LIVE": "segmentation_ready_live",
    "OPEN_VOCAB_READY_LIVE": "open_vocab_ready_live",
    "EMBEDDING_READY_LIVE": "embedding_ready_live",
    "GATED_TOKEN_REQUIRED": "gated_token_required",
    "CATALOG_ONLY_ENGINE_NOT_WIRED": "hidden_catalog_only",
    "CUSTOM_LOADER_REQUIRED": "hidden_custom_loader_required",
    "DEPENDENCY_MISSING": "blocked_dependency",
    "WEIGHTS_MISSING": "blocked_weights",
    "PARTIAL_IMPLEMENTATION_BLOCKED": "blocked_partial",
}


def _load(p):
    if not p.exists():
        pytest.skip(f"{p} not generated")
    return json.loads(p.read_text())


def test_allowlist_is_complete_partition():
    data = _load(ALLOWLIST)
    placed = [m for v in data.values() if isinstance(v, list) for m in v]
    assert len(placed) == len(set(placed)), "a model appears in two buckets"
    assert set(placed) == set(CAPS), "allowlist is not a complete partition of the catalog"


def test_each_model_in_the_bucket_its_state_implies():
    data = _load(ALLOWLIST)
    for mid, c in CAPS.items():
        want = _STATE_TO_BUCKET.get(c["readiness_state"])
        if want:
            assert mid in data[want], f"{mid} ({c['readiness_state']}) not in bucket {want}"


def test_ready_buckets_are_all_live_verified():
    data = _load(ALLOWLIST)
    for bucket in (
        "train_ready_live",
        "inference_ready_live",
        "segmentation_ready_live",
        "open_vocab_ready_live",
        "embedding_ready_live",
    ):
        for mid in data[bucket]:
            assert CAPS[mid]["live_verified_inference"] or CAPS[mid]["live_verified_train"], (
                bucket,
                mid,
            )


def test_final_matrix_matches_catalog():
    data = _load(FINAL)
    assert {r["model_id"] for r in data["models"]} == set(CAPS)

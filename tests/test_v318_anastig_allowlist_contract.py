# SPDX-License-Identifier: Apache-2.0
"""v3.18 Anastig allowlist contract — the JSON must be a faithful projection of
``model_capabilities`` and must only ever list live-verified models as ready.

Weight-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
ALLOWLIST = Path("docs/anastig_model_allowlist_v318.json")

_READY_BUCKETS = {
    "train_ready_live",
    "inference_ready_live",
    "embedding_ready_live",
    "segmentation_ready_live",
    "open_vocab_ready_live",
}


def _load():
    if not ALLOWLIST.exists():
        pytest.skip("anastig allowlist not generated")
    return json.loads(ALLOWLIST.read_text())


def test_allowlist_buckets_present():
    data = _load()
    for k in _READY_BUCKETS | {
        "gated_token_required",
        "hidden_catalog_only",
        "blocked",
        "license_blocked",
    }:
        assert k in data, f"missing bucket {k}"


def test_allowlist_is_partition_of_catalog():
    data = _load()
    buckets = [v for k, v in data.items() if isinstance(v, list)]
    allmodels = [m for b in buckets for m in b]
    # no duplicates across buckets, and every catalog model is placed exactly once
    assert len(allmodels) == len(set(allmodels)), "a model appears in two buckets"
    assert set(allmodels) == set(CAPS), "allowlist is not a complete partition of the catalog"


def test_ready_buckets_only_contain_live_verified_models():
    data = _load()
    for bucket in _READY_BUCKETS:
        for mid in data[bucket]:
            cap = CAPS[mid]
            assert cap["readiness_state"] in taxonomy.LIVE_READY_STATES, (bucket, mid)
            assert cap["live_verified_inference"] or cap["live_verified_train"], (bucket, mid)


def test_train_ready_live_bucket_requires_live_train():
    data = _load()
    for mid in data["train_ready_live"]:
        assert CAPS[mid]["live_verified_train"], mid
        assert CAPS[mid]["readiness_state"] == taxonomy.TRAIN_READY_LIVE, mid


def test_gated_bucket_requires_token():
    data = _load()
    for mid in data["gated_token_required"]:
        assert CAPS[mid]["requires_token"], mid


def test_license_blocked_bucket_is_never_commercial_safe():
    data = _load()
    for mid in data["license_blocked"]:
        assert not CAPS[mid]["commercial_safe"], mid

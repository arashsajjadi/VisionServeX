# SPDX-License-Identifier: Apache-2.0
"""v3.18 capability-truth contract — the schema Anastig drives off.

Weight-free: every assertion reads only ``model_capabilities`` metadata and the
committed live matrices. No model is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import visionservex as vsx
from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import live_evidence, taxonomy

ALL_IDS = list_models()
CAPS = {m: model_capabilities(m) for m in ALL_IDS}

REQUIRED_KEYS = {
    "model_id",
    "family",
    "task",
    "engine",
    "readiness",
    "readiness_state",
    "anastig_visibility",
    "blocker",
    "legal_status",
    "license",
    "license_class",
    "commercial_safe",
    "gated",
    "requires_token",
    "legal_review_required",
    "predict_supported",
    "live_verified_inference",
    "live_verified_train",
    "train_supported",
    "checkpoint_load_supported",
    "export_supported",
    "validated_syntax",
    "tasks",
}


def test_top_level_exports_present():
    assert callable(vsx.list_models)
    assert callable(vsx.model_capabilities)
    assert vsx.list_models() == list_models()


@pytest.mark.parametrize("mid", ALL_IDS)
def test_every_model_has_required_keys(mid):
    cap = CAPS[mid]
    missing = REQUIRED_KEYS - set(cap)
    assert not missing, f"{mid} missing capability keys: {missing}"


def test_readiness_state_is_in_vocabulary():
    for mid, cap in CAPS.items():
        assert cap["readiness_state"] in taxonomy.READINESS_STATES, mid


def test_coarse_readiness_is_compatible_with_precise_state():
    # The legacy 4-bucket must be a faithful (if coarser) view of the precise
    # state — never contradict it.
    for mid, cap in CAPS.items():
        coarse = cap["readiness"]
        state = cap["readiness_state"]
        assert coarse in ("train-ready", "inference-ready", "catalog-only", "blocked")
        if state in taxonomy.LIVE_READY_STATES:
            assert coarse in ("train-ready", "inference-ready"), (mid, coarse, state)
        if coarse == "train-ready":
            # a train-ready coarse bucket is either live or derived train
            assert "TRAIN_READY" in state, (mid, state)


def test_requires_token_iff_gated():
    for mid, cap in CAPS.items():
        assert cap["requires_token"] == cap["gated"], mid


def test_commercial_safe_never_for_copyleft_or_noncommercial():
    for mid, cap in CAPS.items():
        if cap["license_class"] in ("copyleft", "noncommercial"):
            assert not cap["commercial_safe"], mid


def test_live_verified_flags_match_baked_evidence():
    for mid, cap in CAPS.items():
        assert cap["live_verified_inference"] == live_evidence.live_inference_verified(mid)
        assert cap["live_verified_train"] == live_evidence.live_train_verified(mid)


def test_baked_evidence_matches_committed_matrices():
    """The frozensets shipped in the package must equal the committed matrices."""
    inf_matrix = live_evidence.inference_verified_from_matrix()
    if inf_matrix:  # only enforce when the matrix has been generated
        assert set(live_evidence.LIVE_INFERENCE_VERIFIED) == inf_matrix, (
            "LIVE_INFERENCE_VERIFIED drifted from live_inference_matrix.json — "
            "run tools/qa/v318_sync_live_evidence.py"
        )
    trn_matrix = live_evidence.train_verified_from_matrix()
    if trn_matrix:
        assert set(live_evidence.LIVE_TRAIN_VERIFIED) == trn_matrix, (
            "LIVE_TRAIN_VERIFIED drifted from live_train_lifecycle_matrix.json — "
            "run tools/qa/v318_sync_live_evidence.py"
        )


def test_live_inference_failed_models_are_blocked_not_derived():
    """A model that was live-tested and FAILED must show its real blocker, never
    an optimistic ``*_DERIVED``."""
    for mid in live_evidence.LIVE_INFERENCE_FAILED:
        if mid not in CAPS:
            continue
        state = CAPS[mid]["readiness_state"]
        assert "DERIVED" not in state, (mid, state)
        assert state not in taxonomy.LIVE_READY_STATES, (mid, state)


def test_inventory_artifact_matches_capabilities_when_present():
    p = Path("docs/qa/v318_full_model_truth/discovered_models.json")
    if not p.exists():
        pytest.skip("inventory artifact not generated")
    data = json.loads(p.read_text())
    ids = {m["model_id"] for m in data["models"]}
    assert ids == set(ALL_IDS), "inventory drifted from list_models()"

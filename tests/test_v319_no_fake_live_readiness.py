# SPDX-License-Identifier: Apache-2.0
"""v3.19: no model is *_LIVE without committed live evidence. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import live_evidence, taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_every_live_state_has_a_live_flag():
    for mid, c in CAPS.items():
        if c["readiness_state"] in taxonomy.LIVE_READY_STATES:
            assert c["live_verified_inference"] or c["live_verified_train"], mid


def test_train_ready_live_is_in_committed_train_evidence():
    matrix = live_evidence.train_verified_from_matrix()
    for mid, c in CAPS.items():
        if c["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert mid in matrix, f"{mid} TRAIN_READY_LIVE but not in any committed train matrix"


def test_inference_live_is_in_committed_inference_evidence():
    matrix = live_evidence.inference_verified_from_matrix()
    inf_live = {
        taxonomy.INFERENCE_READY_LIVE,
        taxonomy.EMBEDDING_READY_LIVE,
        taxonomy.SEGMENTATION_READY_LIVE,
        taxonomy.OPEN_VOCAB_READY_LIVE,
        taxonomy.VLM_READY_LIVE,
    }
    for mid, c in CAPS.items():
        # train-ready-live models prove inference via the train lifecycle, not the
        # inference matrix — exclude them here.
        if c["readiness_state"] in inf_live and not c["live_verified_train"]:
            assert mid in matrix, f"{mid} inference-live but not in any committed inference matrix"


def test_baked_evidence_matches_committed_matrices():
    assert (
        set(live_evidence.LIVE_INFERENCE_VERIFIED) == live_evidence.inference_verified_from_matrix()
    )
    assert set(live_evidence.LIVE_TRAIN_VERIFIED) == live_evidence.train_verified_from_matrix()


def test_no_train_ready_derived_remains():
    # v3.19 promoted all RF-DETR; the derived-train bucket must be empty.
    derived = [
        m
        for m, c in CAPS.items()
        if c["readiness_state"] == taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION
    ]
    assert derived == [], f"unexpected derived-train models: {derived}"

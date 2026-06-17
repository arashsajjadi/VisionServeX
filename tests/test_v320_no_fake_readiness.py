# SPDX-License-Identifier: Apache-2.0
"""v3.20: no fake readiness — every *_LIVE state and every live flag has evidence.

Weight-free.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import live_evidence, taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_live_states_have_a_live_flag():
    for mid, c in CAPS.items():
        if c["readiness_state"] in taxonomy.LIVE_READY_STATES:
            assert (
                c["inference_live_verified"]
                or c["train_live_verified"]
                or c["fine_tune_live_verified"]
            ), mid


def test_every_live_flag_is_backed_by_baked_evidence():
    for mid, c in CAPS.items():
        if c["inference_live_verified"]:
            assert mid in live_evidence.LIVE_INFERENCE_VERIFIED, mid
        if c["train_live_verified"]:
            assert mid in live_evidence.LIVE_TRAIN_VERIFIED, mid
        if c["fine_tune_live_verified"]:
            assert mid in live_evidence.LIVE_FINETUNE_VERIFIED, mid
        if c["reload_live_verified"]:
            assert mid in live_evidence.LIVE_RELOAD_VERIFIED, mid
        if c["export_live_verified"]:
            assert mid in live_evidence.LIVE_EXPORT_VERIFIED, mid


def test_baked_evidence_matches_committed_matrices():
    assert (
        set(live_evidence.LIVE_INFERENCE_VERIFIED) == live_evidence.inference_verified_from_matrix()
    )
    assert set(live_evidence.LIVE_TRAIN_VERIFIED) == live_evidence.train_verified_from_matrix()
    assert (
        set(live_evidence.LIVE_FINETUNE_VERIFIED) == live_evidence.finetune_verified_from_matrix()
    )
    assert set(live_evidence.LIVE_RELOAD_VERIFIED) == live_evidence.reload_verified_from_matrix()
    assert set(live_evidence.LIVE_EXPORT_VERIFIED) == live_evidence.export_verified_from_matrix()


def test_no_derived_train_remains():
    derived = [
        m
        for m, c in CAPS.items()
        if c["readiness_state"] == taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION
    ]
    assert derived == [], derived

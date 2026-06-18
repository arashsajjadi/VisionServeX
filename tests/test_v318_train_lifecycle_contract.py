# SPDX-License-Identifier: Apache-2.0
"""v3.18 train-lifecycle capability contract (weight-free).

The live lifecycle itself runs in ``tests/live/test_v318_live_train_lifecycle_matrix.py``;
this file enforces the *capability* invariants that must hold without loading a model.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import live_evidence, taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_train_ready_live_models_passed_the_live_matrix():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert cap["live_verified_train"], mid
            assert mid in live_evidence.LIVE_TRAIN_VERIFIED, mid


def test_train_ready_live_implies_full_capability_chain():
    # A live-trained model must support the whole reload+predict chain — no
    # train-ready without a way to use the trained weights.
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert cap["train_supported"], mid
            assert cap["checkpoint_save_supported"], mid
            assert cap["checkpoint_load_supported"], mid
            assert cap["trained_checkpoint_predict_supported"], mid


def test_train_derived_models_are_not_live_train_verified():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION:
            assert not cap["live_verified_train"], mid


def test_train_supported_requires_reload_predict_support():
    # No overclaim: a trainable model must be able to reload+predict its checkpoint.
    for mid, cap in CAPS.items():
        if cap["train_supported"] and cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert cap["trained_checkpoint_predict_supported"], mid


def test_at_least_the_known_train_ready_live_models():
    live = {m for m, c in CAPS.items() if c["readiness_state"] == taxonomy.TRAIN_READY_LIVE}
    # libreyolo + torchvision lifecycle was live-validated this sprint.
    for m in ("libreyolo-yolox-s", "torchvision-resnet50"):
        assert m in live, f"{m} should be TRAIN_READY_LIVE"

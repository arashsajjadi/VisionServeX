# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0: matrix-wide capability-truth invariants (no fake-ready, evidence-backed)."""

from __future__ import annotations

from visionservex.core.model import model_capabilities
from visionservex.registry import default_registry

_ALL = [e.id for e in default_registry().list()]


def test_train_ready_requires_validated_lifecycle_and_reload():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["readiness"] == "train-ready":
            assert cap["train_supported"] is True, mid
            assert cap["trained_checkpoint_predict_supported"] is True, mid
            assert cap["validated_lifecycle"] is True, mid
            assert cap["post_nms_predict_supported"] is True, mid


def test_inference_ready_requires_registered_engine():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["pretrained_inference_supported"]:
            assert cap["engine_registered"], mid
            assert cap["implementation_status"] == "wired", mid


def test_catalog_only_and_blocked_have_exact_blocker():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["readiness"] in ("catalog-only", "blocked"):
            assert cap["exact_blocker"], f"{mid} {cap['readiness']} without exact_blocker"
            assert not cap["pretrained_inference_supported"], mid


def test_train_supported_implies_reload_predict():
    offenders = [
        mid
        for mid in _ALL
        if model_capabilities(mid)["train_supported"]
        and not model_capabilities(mid)["trained_checkpoint_predict_supported"]
    ]
    assert not offenders, offenders


def test_commercial_safe_requires_policy():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["commercial_safe"]:
            assert cap["has_policy_row"], mid
            assert cap["legal_status"] == "commercial_safe_core", mid


def test_no_model_is_both_train_ready_and_blocked():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["readiness"] == "train-ready":
            assert cap["exact_blocker"] is None, mid


def test_embed_models_are_inference_not_detector():
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["task"] == "embed":
            assert cap["readiness"] in ("inference-ready", "catalog-only", "blocked")
            assert cap["train_supported"] is False  # embedding != detector training

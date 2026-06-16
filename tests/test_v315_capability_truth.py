# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0: capability-truth contract + no-fake-ready invariants.

Every registry model must have a complete, honest capability object. No model is
"ready" without the real backing (engine for inference; reload+predict for
training). Derived entirely from source — no memory.
"""

from __future__ import annotations

import pytest

from visionservex.core.model import model_capabilities
from visionservex.registry import default_registry

_ALL_IDS = [e.id for e in default_registry().list()]
_REQUIRED_KEYS = {
    "model_id",
    "family",
    "task",
    "engine",
    "readiness",
    "legal_status",
    "commercial_safe",
    "license_code",
    "has_policy_row",
    "implementation_status",
    "engine_registered",
    "pretrained_inference_supported",
    "train_supported",
    "finetune_supported",
    "checkpoint_save_supported",
    "checkpoint_load_supported",
    "trained_checkpoint_predict_supported",
    "export_supported",
    "supported_dataset_formats",
    "exact_blocker",
}
_VALID_READINESS = {"train-ready", "inference-ready", "catalog-only", "blocked"}


def test_every_model_has_capability_object():
    for mid in _ALL_IDS:
        cap = model_capabilities(mid)
        missing = _REQUIRED_KEYS - set(cap)
        assert not missing, f"{mid} capability missing keys: {missing}"
        assert cap["readiness"] in _VALID_READINESS, f"{mid} bad readiness {cap['readiness']!r}"


def test_train_ready_implies_reload_predict():
    """Invariant: train_supported=True ⇒ trained_checkpoint_predict_supported=True."""
    offenders = [
        mid
        for mid in _ALL_IDS
        if model_capabilities(mid)["train_supported"]
        and not model_capabilities(mid)["trained_checkpoint_predict_supported"]
    ]
    assert not offenders, f"train_supported without reload+predict (overclaim): {offenders}"


def test_train_ready_readiness_requires_reload_predict():
    """A model whose readiness=='train-ready' must support reload+predict and inference."""
    for mid in _ALL_IDS:
        cap = model_capabilities(mid)
        if cap["readiness"] == "train-ready":
            assert cap["trained_checkpoint_predict_supported"], (
                f"{mid} train-ready w/o reload+predict"
            )
            assert cap["pretrained_inference_supported"], f"{mid} train-ready w/o inference"
            assert cap["checkpoint_load_supported"], f"{mid} train-ready w/o checkpoint load"


def test_inference_ready_has_registered_engine():
    """No inference-ready model without a real registered engine path."""
    for mid in _ALL_IDS:
        cap = model_capabilities(mid)
        if cap["pretrained_inference_supported"]:
            assert cap["engine_registered"], f"{mid} inference-ready but engine not registered"
            assert cap["implementation_status"] == "wired", f"{mid} inference-ready but not wired"


def test_no_fake_ready_catalog_only_has_blocker():
    """Catalog-only / blocked models must carry an exact blocker (no silent fake-ready)."""
    for mid in _ALL_IDS:
        cap = model_capabilities(mid)
        if cap["readiness"] in ("catalog-only", "blocked"):
            assert cap["exact_blocker"], f"{mid} is {cap['readiness']} but has no exact_blocker"
            assert not cap["pretrained_inference_supported"], (
                f"{mid} {cap['readiness']} yet inference-ready"
            )


def test_commercial_safe_requires_policy_row():
    """A model is only 'commercial_safe' if a curated policy row grants default-safe."""
    for mid in _ALL_IDS:
        cap = model_capabilities(mid)
        if cap["commercial_safe"]:
            assert cap["has_policy_row"], f"{mid} commercial_safe without a policy row"
            assert cap["legal_status"] == "commercial_safe_core", (
                f"{mid} commercial_safe but {cap['legal_status']}"
            )


@pytest.mark.parametrize(
    "mid", ["libreyolo-yolox-s", "rfdetr-nano", "torchvision-resnet50", "sam-vit-base"]
)
def test_capabilities_method_matches_function(mid):
    from visionservex.core.model import VisionModel

    assert VisionModel(mid).capabilities() == model_capabilities(mid)

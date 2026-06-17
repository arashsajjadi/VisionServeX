# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: model_capabilities truth — detectors carry post_nms / validated fields,
and no detector is train-ready without a validated, NMS'd predict path."""

from __future__ import annotations

from visionservex.core.model import model_capabilities
from visionservex.registry import default_registry

_DETECT_TASKS = {"detect", "obb"}
_ALL = [e.id for e in default_registry().list()]


def test_capability_has_v316_fields():
    cap = model_capabilities("libreyolo-yolox-s")
    for k in ("post_nms_predict_supported", "validated_lifecycle", "exact_blocker"):
        assert k in cap


def test_train_ready_requires_validated_lifecycle_and_nms():
    """No model is train-ready unless its lifecycle is validated AND its predict
    output is post-NMS (no raw-proposal flood claimed as final)."""
    for mid in _ALL:
        cap = model_capabilities(mid)
        if cap["readiness"] == "train-ready":
            assert cap["trained_checkpoint_predict_supported"] is True, mid
            assert cap["validated_lifecycle"] is True, mid
            assert cap["post_nms_predict_supported"] is True, mid


def test_train_supported_implies_reload_predict():
    offenders = [
        mid
        for mid in _ALL
        if model_capabilities(mid)["train_supported"]
        and not model_capabilities(mid)["trained_checkpoint_predict_supported"]
    ]
    assert not offenders, f"train_supported without reload+predict: {offenders}"


def test_inference_only_detectors_have_exact_blocker():
    """A wired detector that is NOT train-ready must explain why (exact_blocker)."""
    for e in default_registry().list():
        if e.family != "libreyolo":
            continue
        cap = model_capabilities(e.id)
        if cap["readiness"] == "inference-ready" and not cap["train_supported"]:
            assert cap["exact_blocker"], f"{e.id} inference-only without exact_blocker"

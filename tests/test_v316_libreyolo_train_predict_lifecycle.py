# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: per-variant LibreYOLO train→predict lifecycle truth.

A LibreYOLO variant is train-ready ONLY if its full lifecycle (train → checkpoint
→ reload → predict confident boxes → NMS → export) is live-validated. Larger /
unvalidated variants are inference-ready; D-FINE training is blocked upstream.
"""

from __future__ import annotations

import pytest

from visionservex.core.model import _training_capabilities, model_capabilities
from visionservex.registry import default_registry

VALIDATED = ["libreyolo-yolox-s", "libreyolo-yolov9-s", "libreyolo-rtdetr-r50"]
INFERENCE_ONLY = [
    "libreyolo-yolox-m",
    "libreyolo-yolox-l",
    "libreyolo-yolox-x",
    "libreyolo-yolov9-m",
    "libreyolo-yolov9-c",
    "libreyolo-rtdetr-r101",
]
DFINE_BLOCKED = [
    "libreyolo-dfine-n",
    "libreyolo-dfine-s",
    "libreyolo-dfine-m",
    "libreyolo-dfine-l",
    "libreyolo-dfine-x",
]


@pytest.mark.parametrize("mid", VALIDATED)
def test_validated_variants_are_train_ready(mid):
    cap = _training_capabilities(mid)
    assert cap["train_supported"] is True
    assert cap["trained_checkpoint_predict_supported"] is True
    assert cap["post_nms_predict_supported"] is True
    assert cap["validated_lifecycle"] is True
    assert cap["exact_blocker"] is None
    assert model_capabilities(mid)["readiness"] == "train-ready"


@pytest.mark.parametrize("mid", INFERENCE_ONLY)
def test_larger_variants_inference_ready_not_overclaimed(mid):
    cap = _training_capabilities(mid)
    assert cap["train_supported"] is False  # not individually lifecycle-validated
    assert cap["trained_checkpoint_predict_supported"] is False
    assert cap["validated_lifecycle"] is False
    assert cap["exact_blocker"] == "VARIANT_NOT_LIFECYCLE_VALIDATED"
    assert cap["post_nms_predict_supported"] is True  # predict still applies NMS
    assert model_capabilities(mid)["readiness"] == "inference-ready"


@pytest.mark.parametrize("mid", DFINE_BLOCKED)
def test_dfine_training_blocked_inference_ready(mid):
    cap = _training_capabilities(mid)
    assert cap["train_supported"] is False
    assert cap["exact_blocker"] == "UPSTREAM_DFINE_FDR_TOPK_CRASH"
    assert cap["post_nms_predict_supported"] is True
    # still runnable for inference
    assert model_capabilities(mid)["readiness"] == "inference-ready"


def test_every_libreyolo_train_ready_is_validated():
    """Invariant: no LibreYOLO variant is train-ready without validated_lifecycle."""
    for e in default_registry().list():
        if e.family != "libreyolo":
            continue
        cap = _training_capabilities(e.id)
        if cap["train_supported"]:
            assert cap["validated_lifecycle"] is True, f"{e.id} train-ready but not validated"
            assert cap["trained_checkpoint_predict_supported"] is True


def test_yolonas_never_trainable():
    cap = _training_capabilities("libreyolo-yolonas-s")
    assert cap["train_supported"] is False
    assert cap["exact_blocker"] == "LIBREYOLO_NONCOMMERCIAL_FAMILY"

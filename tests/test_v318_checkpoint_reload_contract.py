# SPDX-License-Identifier: Apache-2.0
"""v3.18 checkpoint-reload capability contract (weight-free)."""

from __future__ import annotations

from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_from_checkpoint_and_load_checkpoint_exist():
    assert hasattr(VisionModel, "from_checkpoint")
    assert hasattr(VisionModel, "load_checkpoint")


def test_train_supported_never_overclaims_reload():
    # If a model says it can train AND it is train-ready-live, it must support
    # trained-checkpoint reload+predict (the v3.14 invariant, kept).
    from visionservex.readiness import taxonomy

    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert cap["checkpoint_load_supported"], mid
            assert cap["trained_checkpoint_predict_supported"], mid


def test_checkpoint_load_supported_is_boolean_everywhere():
    for mid, cap in CAPS.items():
        assert isinstance(cap["checkpoint_load_supported"], bool), mid
        assert isinstance(cap["trained_checkpoint_predict_supported"], bool), mid


def test_unsupported_reload_raises_structured_error():
    # A non-trainable family (mock detector) must refuse trained-checkpoint reload
    # with a structured NotImplementedError, never a silent base-weight fallback.
    import pytest

    m = VisionModel("mock-detect")
    with pytest.raises(NotImplementedError) as ei:
        m.load_checkpoint("/nonexistent/path.pt")
    assert "CHECKPOINT_LOAD_UNSUPPORTED" in str(ei.value)

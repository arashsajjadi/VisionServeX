# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0: unified checkpoint-reload contract across all reload-capable families."""

from __future__ import annotations

import pytest

from visionservex.core.model import VisionModel, model_capabilities
from visionservex.engines import build_engine
from visionservex.engines.base import MissingDependencyError
from visionservex.registry import default_registry


def test_from_checkpoint_requires_model_id():
    with pytest.raises(TypeError):
        VisionModel.from_checkpoint("x.pt")  # model_id is keyword-required


def test_from_checkpoint_unsupported_family_raises():
    with pytest.raises(NotImplementedError) as exc:
        VisionModel.from_checkpoint("x.pt", model_id="mock-detect")
    assert "CHECKPOINT_LOAD_UNSUPPORTED" in str(exc.value)


def test_load_checkpoint_unsupported_family_raises():
    with pytest.raises(NotImplementedError) as exc:
        VisionModel("mock-detect").load_checkpoint("x.pt")
    assert "CHECKPOINT_LOAD_UNSUPPORTED" in str(exc.value)


@pytest.mark.parametrize("mid", ["libreyolo-yolox-s", "rfdetr-nano", "torchvision-resnet18"])
def test_engine_has_load_checkpoint(mid):
    eng = build_engine(default_registry().get(mid))
    assert callable(getattr(eng, "load_checkpoint", None))


def test_torchvision_missing_checkpoint_clean_error(tmp_path):
    eng = build_engine(default_registry().get("torchvision-resnet18"))
    with pytest.raises(MissingDependencyError):
        eng.load_checkpoint(tmp_path / "nope.pt", device="cpu")


def test_every_reload_claim_has_load_checkpoint_engine():
    """Invariant: every model claiming trained_checkpoint_predict_supported has an
    engine that actually implements load_checkpoint (no fake reload)."""
    offenders = []
    for e in default_registry().list():
        if model_capabilities(e.id)["trained_checkpoint_predict_supported"]:
            eng = build_engine(e)
            if not callable(getattr(eng, "load_checkpoint", None)):
                offenders.append(e.id)
    assert not offenders, f"claim reload+predict but engine has no load_checkpoint: {offenders}"

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.15.0: torchvision classic classifiers — wiring + capability + arch truth.

Weight-free: proves the registry/engine/capability wiring without downloading
ImageNet weights. The real pretrained-inference and fine-tune lifecycle are
proven by ``tests/live/test_v315_*`` (gated) and the validation in the release.
"""

from __future__ import annotations

import pytest

from visionservex.core.model import _export_capabilities, _training_capabilities, model_capabilities
from visionservex.engines.torchvision_classification import _TV_ARCHS, TorchvisionClassifyEngine
from visionservex.registry import default_registry

TV_IDS = sorted(_TV_ARCHS)


def test_thirteen_curated_variants():
    assert len(_TV_ARCHS) == 13
    assert "torchvision-resnet50" in _TV_ARCHS
    assert _TV_ARCHS["torchvision-wide-resnet50-2"] == "wide_resnet50_2"
    assert _TV_ARCHS["torchvision-resnext50-32x4d"] == "resnext50_32x4d"
    assert _TV_ARCHS["torchvision-mobilenet-v3-large"] == "mobilenet_v3_large"


@pytest.mark.parametrize("mid", TV_IDS)
def test_registry_and_engine_wiring(mid):
    entry = default_registry().get(mid)
    assert entry.task == "classify"
    assert entry.family == "torchvision-classify"
    assert entry.engine == "torchvision_classify"
    assert entry.license == "BSD-3-Clause"
    assert entry.install_extra == "torchvision"
    assert entry.implementation_status == "wired"
    from visionservex.engines import build_engine

    eng = build_engine(entry)
    assert isinstance(eng, TorchvisionClassifyEngine)
    assert callable(getattr(eng, "train", None))
    assert callable(getattr(eng, "load_checkpoint", None))
    assert callable(getattr(eng, "export", None))


@pytest.mark.parametrize("mid", TV_IDS)
def test_capability_truth_full_lifecycle(mid):
    cap = _training_capabilities(mid)
    assert cap["train_supported"] is True
    assert cap["finetune_supported"] is True
    assert cap["checkpoint_load_supported"] is True
    assert cap["trained_checkpoint_predict_supported"] is True
    assert cap["supported_dataset_formats"] == ["imagefolder"]
    assert cap["required_extra"] == "torchvision"
    assert _export_capabilities(mid)["onnx"]["status"] == "supported"

    full = model_capabilities(mid)
    assert full["readiness"] == "train-ready"
    assert full["commercial_safe"] is True
    assert full["legal_status"] == "commercial_safe_core"


def test_no_untested_variants_exposed():
    """Every registry torchvision-classify id is a curated arch, and vice versa."""
    reg_ids = {e.id for e in default_registry().list() if e.family == "torchvision-classify"}
    assert reg_ids == set(_TV_ARCHS), f"registry/arch mismatch: {reg_ids ^ set(_TV_ARCHS)}"


def test_unknown_arch_rejected():
    import types

    from visionservex.engines import build_engine

    eng = build_engine(default_registry().get("torchvision-resnet18"))
    eng.entry = types.SimpleNamespace(id="torchvision-bogusnet")
    from visionservex.engines.base import MissingDependencyError

    with pytest.raises(MissingDependencyError):
        eng._arch()

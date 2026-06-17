# SPDX-License-Identifier: Apache-2.0
"""v3.18 public-API surface + validated_syntax contract (weight-free)."""

from __future__ import annotations

import visionservex as vsx
from visionservex import VisionModel
from visionservex.core.model import list_models, model_capabilities


def test_top_level_functions_exist():
    assert callable(vsx.list_models)
    assert callable(vsx.model_capabilities)
    assert "list_models" in vsx.__all__
    assert "model_capabilities" in vsx.__all__


def test_visionmodel_has_every_task_method():
    for name in (
        "predict",
        "detect",
        "segment",
        "classify",
        "embed",
        "similarity",
        "correspond",
        "train",
        "val",
        "export",
        "from_checkpoint",
        "load_checkpoint",
        "capabilities",
    ):
        assert hasattr(VisionModel, name), f"VisionModel missing {name}()"


def test_list_models_filters():
    all_ids = list_models()
    assert list_models(task="detect") and set(list_models(task="detect")) <= set(all_ids)
    fams = {model_capabilities(m)["family"] for m in all_ids}
    some_fam = next(iter(fams))
    assert all(model_capabilities(m)["family"] == some_fam for m in list_models(family=some_fam))


def test_validated_syntax_matches_task():
    for mid in list_models():
        cap = model_capabilities(mid)
        syn = cap["validated_syntax"]
        assert "predict" in syn
        if cap["task"] in ("classify", "classification"):
            assert "classify" in syn
        if cap["task"] in ("embed", "embedding"):
            assert "embed" in syn and "similarity" in syn
        if cap["task"] in ("segment", "foundation_segment", "grounded_segment"):
            assert "segment" in syn
        if cap["task"] in ("detect", "obb", "open_vocab_detect"):
            assert "detect" in syn
        # train-ready models must advertise train + from_checkpoint syntax
        if cap["train_supported"]:
            assert "train" in syn and "from_checkpoint" in syn


def test_capabilities_method_matches_function():
    mid = list_models()[0]
    assert VisionModel(mid).capabilities() == model_capabilities(mid)

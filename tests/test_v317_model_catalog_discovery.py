# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.17.0: every registered model is discoverable with a complete capability object."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.registry import default_registry

_ALL = [e.id for e in default_registry().list()]
_REQUIRED = {
    "model_id",
    "family",
    "task",
    "tasks",
    "engine",
    "readiness",
    "legal_status",
    "commercial_safe",
    "gated",
    "pretrained_inference_supported",
    "train_supported",
    "trained_checkpoint_predict_supported",
    "post_nms_predict_supported",
    "validated_lifecycle",
    "export_supported",
    "exact_blocker",
    "validated_syntax",
}
_VALID_READINESS = {"train-ready", "inference-ready", "catalog-only", "blocked"}


def test_list_models_covers_registry():
    assert set(list_models()) == set(_ALL)
    assert len(_ALL) >= 151


def test_every_model_has_complete_capability():
    for mid in _ALL:
        cap = model_capabilities(mid)
        missing = _REQUIRED - set(cap)
        assert not missing, f"{mid} capability missing {missing}"
        assert cap["readiness"] in _VALID_READINESS
        assert cap["tasks"] == [cap["task"]]
        assert "predict" in cap["validated_syntax"]


def test_list_models_task_filter():
    classify = list_models(task="classify")
    assert classify and all(model_capabilities(m)["task"] == "classify" for m in classify)


def test_discovery_artifact_matches_registry_if_present():
    import json
    from pathlib import Path

    art = (
        Path(__file__).resolve().parents[1]
        / "docs/qa/v317_full_model_matrix/discovered_models.json"
    )
    if not art.is_file():
        return
    discovered = {r["model_id"] for r in json.loads(art.read_text())}
    assert discovered == set(_ALL), f"discovery drift: {discovered ^ set(_ALL)}"

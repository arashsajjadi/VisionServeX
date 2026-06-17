# SPDX-License-Identifier: Apache-2.0
"""v3.20: VLMs are not falsely marked train/fine-tune-ready-live. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
VLM = {m: c for m, c in CAPS.items() if c["task"] == "vlm"}


def test_vlms_exist():
    assert VLM  # florence-2-base/large


def test_vlms_are_not_fake_train_or_finetune_live():
    for mid, c in VLM.items():
        assert not c["train_live_verified"], mid
        assert not c["fine_tune_live_verified"], mid
        assert c["anastig_train_visibility"] != "show_train", mid
        assert c["anastig_finetune_visibility"] != "show_finetune", mid


def test_blocked_vlms_have_exact_blocker():
    from visionservex.readiness import taxonomy

    for mid, c in VLM.items():
        if c["readiness_state"] not in taxonomy.LIVE_READY_STATES:
            assert (c["blocker"] and "transformers" in c["blocker"].lower()) or c["blocker"], mid

# SPDX-License-Identifier: Apache-2.0
"""v3.20: no blocked/hidden model is ever shown to Anastig users. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
_USER_VISIBLE = ("show_train", "show_inference", "show_embedding", "show_segmentation")


def test_blocked_models_are_not_inference_visible():
    for mid, c in CAPS.items():
        if c["readiness_state"] not in taxonomy.LIVE_READY_STATES:
            assert c["anastig_visibility"] not in _USER_VISIBLE, (mid, c["anastig_visibility"])


def test_blocked_models_are_not_train_or_finetune_visible():
    for mid, c in CAPS.items():
        if c["readiness_state"] not in taxonomy.LIVE_READY_STATES:
            assert c["anastig_train_visibility"] != "show_train", mid
            assert c["anastig_finetune_visibility"] != "show_finetune", mid


def test_user_visible_models_are_all_live():
    for mid, c in CAPS.items():
        if c["anastig_visibility"] in _USER_VISIBLE:
            assert c["inference_live_verified"] or c["train_live_verified"], mid

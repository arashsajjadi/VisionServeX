# SPDX-License-Identifier: Apache-2.0
"""v3.19: Anastig default-visible == live-ready only. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
_DEFAULT_VISIBLE = ("show_train", "show_inference", "show_embedding", "show_segmentation")


def test_default_visible_iff_live_verified():
    for mid, c in CAPS.items():
        visible = c["anastig_visibility"] in _DEFAULT_VISIBLE
        live = c["live_verified_inference"] or c["live_verified_train"]
        if visible:
            assert live, f"{mid} visible ({c['anastig_visibility']}) without live proof"


def test_show_train_requires_live_train():
    for mid, c in CAPS.items():
        if c["anastig_visibility"] == "show_train":
            assert c["live_verified_train"], mid
            assert c["readiness_state"] == taxonomy.TRAIN_READY_LIVE, mid


def test_blocked_and_gated_never_default_visible():
    for mid, c in CAPS.items():
        st = c["readiness_state"]
        if st not in taxonomy.LIVE_READY_STATES:
            assert c["anastig_visibility"] not in _DEFAULT_VISIBLE, (
                mid,
                st,
                c["anastig_visibility"],
            )


def test_no_derived_train_is_shown_as_trainable():
    for mid, c in CAPS.items():
        if c["readiness_state"] == taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION:
            assert c["anastig_visibility"] != "show_train", mid

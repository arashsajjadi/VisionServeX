# SPDX-License-Identifier: Apache-2.0
"""v3.20: model_capabilities exposes separate inference/train/fine-tune dimensions.

Weight-free.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}
_DIMS = {
    "inference_ready",
    "inference_live_verified",
    "train_ready",
    "train_live_verified",
    "fine_tune_ready",
    "fine_tune_live_verified",
    "reload_supported",
    "reload_live_verified",
    "export_live_verified",
    "token_never_logged",
    "anastig_train_visibility",
    "anastig_finetune_visibility",
}


def test_all_dimensions_present_and_typed():
    for mid, c in CAPS.items():
        for k in _DIMS:
            assert k in c, f"{mid} missing {k}"
        for k in _DIMS - {"anastig_train_visibility", "anastig_finetune_visibility"}:
            assert isinstance(c[k], bool), (mid, k, type(c[k]))


def test_live_implies_ready():
    # You can't be live-verified for a dimension without being ready for it.
    for mid, c in CAPS.items():
        if c["train_live_verified"]:
            assert c["train_ready"], mid
        if c["fine_tune_live_verified"]:
            assert c["fine_tune_ready"], mid
        if c["inference_live_verified"]:
            assert c["inference_ready"], mid


def test_anastig_train_visibility_consistent():
    for mid, c in CAPS.items():
        v = c["anastig_train_visibility"]
        assert v in ("show_train", "admin_only", "hide")
        if v == "show_train":
            assert c["train_live_verified"], mid
        if v == "admin_only":
            assert c["train_ready"] and c["inference_ready"] and not c["train_live_verified"], mid


def test_anastig_finetune_visibility_consistent():
    for mid, c in CAPS.items():
        v = c["anastig_finetune_visibility"]
        assert v in ("show_finetune", "admin_only", "hide")
        if v == "show_finetune":
            assert c["fine_tune_live_verified"], mid


def test_token_never_logged_is_a_standing_guarantee():
    for mid, c in CAPS.items():
        assert c["token_never_logged"] is True, mid


def test_reload_export_live_are_backed_by_a_live_lifecycle():
    # Reload is proven by either a train OR a fine-tune (head-checkpoint) lifecycle;
    # export is proven only by the full-train lifecycle.
    for mid, c in CAPS.items():
        if c["reload_live_verified"]:
            assert c["train_live_verified"] or c["fine_tune_live_verified"], mid
        if c["export_live_verified"]:
            assert c["train_live_verified"], mid

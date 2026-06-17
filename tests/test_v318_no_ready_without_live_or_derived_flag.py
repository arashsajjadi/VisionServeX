# SPDX-License-Identifier: Apache-2.0
"""v3.18 honesty invariant: no model is reported READY without proof.

A readiness state may promise readiness only when it is either
  * live-verified this sprint (``*_READY_LIVE``), or
  * explicitly flagged derived (``*_DERIVED_NEEDS_LIVE_CONFIRMATION``).

Weight-free.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}

# The only states whose name contains "READY".
_LIVE = taxonomy.LIVE_READY_STATES
_DERIVED = {
    taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
    taxonomy.INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
}


def test_ready_named_states_are_only_live_or_derived():
    for mid, cap in CAPS.items():
        state = cap["readiness_state"]
        if "READY" in state:
            assert state in (_LIVE | _DERIVED), (
                f"{mid}: state {state!r} contains READY but is neither live nor explicitly derived"
            )


def test_live_ready_requires_a_live_flag():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] in _LIVE:
            assert cap["live_verified_inference"] or cap["live_verified_train"], (
                f"{mid}: claims {cap['readiness_state']} but no live_verified flag is set"
            )


def test_train_ready_live_requires_live_train():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_LIVE:
            assert cap["live_verified_train"], mid


def test_derived_states_carry_a_needs_confirmation_blocker():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] in _DERIVED:
            assert cap["blocker"] and "derived" in cap["blocker"].lower(), (mid, cap["blocker"])
            # derived is never live
            assert not (cap["live_verified_inference"] and "INFERENCE" in cap["readiness_state"])


def test_only_live_ready_is_default_visible():
    """Anastig default-visible (show_*, excluding token) requires a live state."""
    for mid, cap in CAPS.items():
        vis = cap["anastig_visibility"]
        if vis in ("show_train", "show_inference", "show_embedding", "show_segmentation"):
            assert cap["live_verified_inference"] or cap["live_verified_train"], (
                f"{mid}: visible as {vis} without live verification"
            )


def test_blocked_and_inference_derived_states_are_not_default_visible():
    # INFERENCE_READY_DERIVED is, by definition, not live-verified -> always hidden.
    # (TRAIN_READY_DERIVED is handled separately: it MAY be show_inference when its
    # inference is live-verified even though its train lifecycle is only derived.)
    always_hidden = {
        taxonomy.INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION,
        taxonomy.CATALOG_ONLY_ENGINE_NOT_WIRED,
        taxonomy.CUSTOM_LOADER_REQUIRED,
        taxonomy.PARTIAL_IMPLEMENTATION_BLOCKED,
        taxonomy.DEPENDENCY_MISSING,
        taxonomy.WEIGHTS_MISSING,
        taxonomy.UPSTREAM_CRASH,
        taxonomy.OOM_BLOCKED,
        taxonomy.TASK_NOT_SUPPORTED,
        taxonomy.LICENSE_BLOCKED,
        taxonomy.NON_COMMERCIAL_BLOCKED,
        taxonomy.UNKNOWN_REVIEW_REQUIRED,
    }
    for mid, cap in CAPS.items():
        if cap["readiness_state"] in always_hidden:
            assert cap["anastig_visibility"] in (
                "hide",
                "blocked_admin_only",
                "show_token_required",
            ), (mid, cap["readiness_state"], cap["anastig_visibility"])


def test_train_ready_derived_is_never_shown_as_trainable():
    # Train is only derived (not live) -> never show_train; at most show_inference
    # when inference itself is live-verified.
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION:
            assert cap["anastig_visibility"] != "show_train", mid
            if (
                cap["anastig_visibility"].startswith("show_")
                and cap["anastig_visibility"] != "show_token_required"
            ):
                assert cap["live_verified_inference"], (mid, "shown without live inference")

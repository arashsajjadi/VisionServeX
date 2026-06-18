# SPDX-License-Identifier: Apache-2.0
"""v3.18 legal gate: gated models require BYOT and never default-commercial-safe."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
GATED = {m: c for m, c in CAPS.items() if c["gated"]}


def test_there_is_at_least_one_gated_model():
    # sam3-base is BYOT; if this ever hits zero the gate is silently vacuous.
    assert GATED, "expected at least one gated model in the catalog"


def test_gated_models_require_token():
    for mid, cap in GATED.items():
        assert cap["requires_token"] is True, mid


def test_gated_models_are_not_default_commercial_safe():
    for mid, cap in GATED.items():
        assert not cap["commercial_safe"], mid


def test_gated_byot_models_use_token_visibility():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] == taxonomy.GATED_TOKEN_REQUIRED:
            assert cap["anastig_visibility"] == "show_token_required", mid
            assert cap["requires_token"] is True, mid


def test_gated_models_never_default_visible_inference():
    # A gated model must never appear as a plain inference/train-ready model.
    for mid, cap in GATED.items():
        assert cap["anastig_visibility"] in ("show_token_required", "hide", "blocked_admin_only"), (
            mid,
            cap["anastig_visibility"],
        )

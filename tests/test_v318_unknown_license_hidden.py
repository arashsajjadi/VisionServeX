# SPDX-License-Identifier: Apache-2.0
"""v3.18 legal gate: unknown / custom license without a policy row stays hidden."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_unknown_or_custom_license_without_policy_is_not_default_visible():
    for mid, cap in CAPS.items():
        if cap["license_class"] in ("custom_unknown", "unknown") and not cap["has_policy_row"]:
            assert cap["anastig_visibility"] in (
                "hide",
                "blocked_admin_only",
                "show_token_required",
            ), (mid, cap["anastig_visibility"])
            assert not cap["commercial_safe"], mid


def test_unknown_custom_license_without_policy_is_flagged_for_review():
    for mid, cap in CAPS.items():
        if cap["license_class"] in ("custom_unknown", "unknown") and not cap["has_policy_row"]:
            assert cap["legal_review_required"] is True, mid


def test_commercial_safe_requires_a_policy_row():
    # commercial-safe is granted ONLY by a curated policy row, never inferred
    # from a bare registry license.
    for mid, cap in CAPS.items():
        if cap["commercial_safe"]:
            assert cap["has_policy_row"], mid

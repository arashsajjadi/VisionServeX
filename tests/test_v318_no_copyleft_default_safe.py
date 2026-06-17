# SPDX-License-Identifier: Apache-2.0
"""v3.18 legal gate: no copyleft / non-commercial model is ever default-safe.

AGPL/GPL/SSPL (copyleft) and non-commercial/research-only licenses may never be
``commercial_safe``, may never be default-visible, and must surface a blocked
readiness state. This binds forever, even though the permissive-only catalog
satisfies it on an empty set today.
"""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}


def test_classify_license_detects_copyleft_and_noncommercial():
    # The classifier itself must work, so the gate is real and not vacuous.
    assert taxonomy.classify_license("AGPL-3.0") == "copyleft"
    assert taxonomy.classify_license("GPL-3.0") == "copyleft"
    assert taxonomy.classify_license("SSPL-1.0") == "copyleft"
    assert taxonomy.classify_license("CC-BY-NC-4.0") == "noncommercial"
    assert taxonomy.classify_license("Deci non-commercial") == "noncommercial"
    assert taxonomy.classify_license("Apache-2.0") == "permissive"
    assert taxonomy.classify_license("MIT") == "permissive"


def test_no_copyleft_model_is_commercial_safe():
    for mid, cap in CAPS.items():
        if cap["license_class"] == "copyleft":
            assert not cap["commercial_safe"], mid


def test_no_noncommercial_model_is_commercial_safe():
    for mid, cap in CAPS.items():
        if cap["license_class"] == "noncommercial":
            assert not cap["commercial_safe"], mid


def test_copyleft_and_noncommercial_are_blocked_states():
    for mid, cap in CAPS.items():
        if cap["license_class"] == "copyleft":
            assert cap["readiness_state"] == taxonomy.LICENSE_BLOCKED, (mid, cap["readiness_state"])
        if cap["license_class"] == "noncommercial":
            assert cap["readiness_state"] == taxonomy.NON_COMMERCIAL_BLOCKED, (
                mid,
                cap["readiness_state"],
            )


def test_blocked_legal_states_are_never_default_visible():
    for mid, cap in CAPS.items():
        if cap["readiness_state"] in (taxonomy.LICENSE_BLOCKED, taxonomy.NON_COMMERCIAL_BLOCKED):
            assert cap["anastig_visibility"] in ("hide", "blocked_admin_only"), mid
            assert not cap["commercial_safe"], mid

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything addendum: _LOCATEANYTHING_FACTS integrity tests.

Tests that the embedded honesty table contains all 10 model IDs, the exact
NVIDIA non-commercial warning text, and that default_safe / commercial_safe
are both False for every locate-anything-* model ID.
"""

from __future__ import annotations

import pytest

_REQUIRED_MODEL_IDS = [
    "locate-anything-3b",
    "locate-anything-3b-v2",
    "locate-anything-3b-grounded",
    "locate-anything-3b-coco",
    "locate-anything-3b-lvis",
    "locate-anything-3b-objects365",
    "locate-anything-3b-open-vocab",
    "locate-anything-3b-caption",
    "locate-anything-3b-video",
    "locate-anything-3b-ft",
]

_REQUIRED_WARNING_FRAGMENTS = [
    "WARNING:",
    "NVIDIA License",
    "non-commercial use only",
    "commercial products",
    "paid SaaS",
    "VisionServeX does not ship or mirror the weights",
    "BYOT/user-local-cache only",
]


def test_locateanything_facts_exist() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    assert "_model_ids" in _LOCATEANYTHING_FACTS
    assert "_warning" in _LOCATEANYTHING_FACTS


def test_all_ten_model_ids_present() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    model_ids_str = _LOCATEANYTHING_FACTS["_model_ids"]
    for mid in _REQUIRED_MODEL_IDS:
        assert mid in model_ids_str, f"Missing model ID: {mid!r}"


def test_exactly_ten_model_ids() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    ids = _LOCATEANYTHING_FACTS["_model_ids"].split()
    assert len(ids) == 10, f"Expected 10 model IDs, got {len(ids)}: {ids}"


def test_warning_text_contains_required_fragments() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    warning = _LOCATEANYTHING_FACTS["_warning"]
    for fragment in _REQUIRED_WARNING_FRAGMENTS:
        assert fragment in warning, f"Warning missing required fragment: {fragment!r}"


def test_license_is_nvidia_noncommercial() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    assert "NVIDIA" in _LOCATEANYTHING_FACTS["_license"]
    assert "non-commercial" in _LOCATEANYTHING_FACTS["_license"]


def test_default_safe_false() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    assert _LOCATEANYTHING_FACTS["_default_safe"] == "false"


def test_commercial_safe_false() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    assert _LOCATEANYTHING_FACTS["_commercial_safe"] == "false"


def test_sidecar_install_references_eagle() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    sidecar = _LOCATEANYTHING_FACTS["_sidecar_install"]
    assert "Eagle" in sidecar or "eagle" in sidecar
    assert "NVlabs" in sidecar


@pytest.mark.parametrize("model_id", _REQUIRED_MODEL_IDS)
def test_vsx_locateanything_handle_explain(model_id: str) -> None:
    from visionservex.vsx import VSX

    h = VSX.locateanything(model_id)
    info = h.explain()
    assert info["family"] == "locate_anything"
    assert info["default_safe"] is False
    assert info["commercial_safe"] is False
    assert info["byot"] is True
    assert "NVIDIA" in info["warning"]
    assert info["state"] == "excluded_restricted"
    assert "--accept-noncommercial" in info["next_command"]


@pytest.mark.parametrize("model_id", _REQUIRED_MODEL_IDS)
def test_vsx_locateanything_status_is_legal_review(model_id: str) -> None:
    from visionservex.vsx import VSX

    assert VSX.locateanything(model_id).status() == "excluded_restricted"

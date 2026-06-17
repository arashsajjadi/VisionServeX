# SPDX-License-Identifier: Apache-2.0
"""v3.19: custom-loader (DEIM/DEIMv2/RT-DETRv4) models stay hidden. Weight-free."""

from __future__ import annotations

from visionservex.core.model import list_models, model_capabilities
from visionservex.readiness import taxonomy

CAPS = {m: model_capabilities(m) for m in list_models()}
CUSTOM = {m: c for m, c in CAPS.items() if c["readiness_state"] == taxonomy.CUSTOM_LOADER_REQUIRED}


def test_custom_loader_models_exist():
    assert CUSTOM  # deim/deimv2/rtdetrv4


def test_custom_loader_models_are_hidden():
    for mid, c in CUSTOM.items():
        assert c["anastig_visibility"] == "hide", (mid, c["anastig_visibility"])
        assert not c["live_verified_inference"], mid


def test_custom_loader_models_carry_a_blocker():
    for mid, c in CUSTOM.items():
        assert c["blocker"], mid


def test_deim_and_rtdetrv4_families_are_custom_loader_or_catalog():
    for mid, c in CAPS.items():
        if c["family"] in ("deim", "rtdetrv4"):
            assert c["readiness_state"] in (
                taxonomy.CUSTOM_LOADER_REQUIRED,
                taxonomy.CATALOG_ONLY_ENGINE_NOT_WIRED,
            ), (mid, c["readiness_state"])
            assert c["anastig_visibility"] == "hide"

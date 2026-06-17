# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16.0: larger LibreYOLO variants are registered honestly (inference-ready)."""

from __future__ import annotations

import pytest

from visionservex.engines import build_engine
from visionservex.engines.libreyolo import LibreYOLOEngine
from visionservex.licensing.policy import get_policy
from visionservex.registry import default_registry

NEW_VARIANTS = {
    "libreyolo-yolox-m": "Apache-2.0",
    "libreyolo-yolox-l": "Apache-2.0",
    "libreyolo-yolox-x": "Apache-2.0",
    "libreyolo-yolov9-m": "MIT",
    "libreyolo-yolov9-c": "MIT",
    "libreyolo-rtdetr-r101": "Apache-2.0",
    "libreyolo-dfine-s": "Apache-2.0",
    "libreyolo-dfine-m": "Apache-2.0",
    "libreyolo-dfine-l": "Apache-2.0",
    "libreyolo-dfine-x": "Apache-2.0",
}


@pytest.mark.parametrize("mid,lic", list(NEW_VARIANTS.items()))
def test_variant_registered_and_wired(mid, lic):
    entry = default_registry().get(mid)
    assert entry.family == "libreyolo"
    assert entry.engine == "libreyolo"
    assert entry.license == lic
    assert entry.download_type == "huggingface"
    assert entry.hf_repo_id and entry.checkpoint_filename
    assert entry.implementation_status == "wired"
    assert isinstance(build_engine(entry), LibreYOLOEngine)


@pytest.mark.parametrize("mid,lic", list(NEW_VARIANTS.items()))
def test_variant_has_commercial_safe_policy(mid, lic):
    pol = get_policy(mid)
    assert pol is not None, f"{mid} missing policy row"
    assert pol.final_policy == "commercial_safe_core"
    assert pol.default_safe is True
    assert pol.can_ship_weights is False
    assert "GPL" not in (pol.weights_license or "").upper()


def test_no_yolonas_variant_registered():
    ids = {e.id for e in default_registry().list() if e.family == "libreyolo"}
    assert not any("yolonas" in i for i in ids)


def test_engine_parses_new_variant_ids():
    from visionservex.engines.libreyolo import _FAMILY_TO_CLASS, _parse_model_id

    assert _parse_model_id("libreyolo-yolox-m") == ("yolox", "m")
    assert _parse_model_id("libreyolo-rtdetr-r101") == ("rtdetr", "r101")
    assert _parse_model_id("libreyolo-dfine-x") == ("dfine", "x")
    # all parsed sub-families map to a runnable libreyolo class
    for mid in NEW_VARIANTS:
        fam, _ = _parse_model_id(mid)
        assert fam in _FAMILY_TO_CLASS

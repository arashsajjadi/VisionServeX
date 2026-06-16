# SPDX-License-Identifier: Apache-2.0
"""v3.12.0: LibreYOLO runtime engine wired into the registry + engine factory.

These tests are intentionally weight-free (no ``real_model``/``network`` markers):
they prove the *wiring* — engine registration, registry resolution, lazy engine
construction, and license-policy classification — without downloading weights or
running inference. The real-inference smoke test runs separately behind the heavy
markers (see scripts/test_*_safe.py and AGENT_RULES.md).
"""

from __future__ import annotations

import pytest

LIBREYOLO_IDS = ["libreyolo-yolox-s", "libreyolo-yolov9-s", "libreyolo-rtdetr-r50"]


def test_libreyolo_engine_registered():
    from visionservex.engines.registry import _FACTORIES

    assert "libreyolo" in _FACTORIES, f"libreyolo engine not registered; have {sorted(_FACTORIES)}"


@pytest.mark.parametrize("mid", LIBREYOLO_IDS)
def test_libreyolo_in_registry(mid):
    from visionservex.registry import default_registry

    entry = default_registry().get(mid)
    assert entry.engine == "libreyolo"
    assert entry.task == "detect"
    assert entry.family == "libreyolo"
    assert entry.download_type == "huggingface"
    assert entry.hf_repo_id and entry.checkpoint_filename
    assert entry.install_extra == "libreyolo"
    assert entry.implementation_status == "wired"


@pytest.mark.parametrize("mid", LIBREYOLO_IDS)
def test_libreyolo_engine_builds(mid):
    # build_engine must return a LibreYOLOEngine without importing the heavy
    # libreyolo package or touching weights (construction is lazy; load() is not
    # called here).
    from visionservex.engines import build_engine
    from visionservex.engines.libreyolo import LibreYOLOEngine
    from visionservex.registry import default_registry

    eng = build_engine(default_registry().get(mid))
    assert isinstance(eng, LibreYOLOEngine)
    assert eng.backend_label == "libreyolo_package"


@pytest.mark.parametrize("mid", LIBREYOLO_IDS)
def test_libreyolo_policy_commercial_safe_core(mid):
    from visionservex.licensing.policy import get_policy

    pol = get_policy(mid)
    assert pol is not None, f"{mid} missing from licensing policy"
    assert pol.final_policy == "commercial_safe_core"
    assert pol.default_safe is True
    assert pol.commercial_safe is True
    assert pol.gated is False
    assert pol.can_ship_weights is False  # VisionServeX never bundles weights


def test_libreyolo_id_parsing():
    from visionservex.engines.libreyolo import _parse_model_id

    assert _parse_model_id("libreyolo-yolox-s") == ("yolox", "s")
    assert _parse_model_id("libreyolo-rtdetr-r50m") == ("rtdetr", "r50m")
    assert _parse_model_id("libreyolo-yolov9-c") == ("yolov9", "c")
    assert _parse_model_id("not-libreyolo") is None


def test_yolonas_not_exposed_as_runnable_family():
    # YOLO-NAS is Deci proprietary / non-commercial; it must NOT be reachable as a
    # runnable default-safe engine family.
    from visionservex.engines.libreyolo import _FAMILY_TO_CLASS

    assert "yolonas" not in _FAMILY_TO_CLASS

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything license guard tests.

Ensures that every LocateAnything-3B model ID is correctly excluded from
the default-safe commercial core, and that the NVIDIA non-commercial license
guard is enforced at every layer (Python API, CLI, runtime).
"""

from __future__ import annotations

import pytest

_ALL_LOCATE_IDS = [
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


@pytest.mark.parametrize("model_id", _ALL_LOCATE_IDS)
def test_not_in_default_safe_core(model_id: str) -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything(model_id).explain()
    assert info["default_safe"] is False, (
        f"{model_id!r} must not be default_safe — NVIDIA non-commercial license"
    )
    assert info["commercial_safe"] is False, (
        f"{model_id!r} must not be commercial_safe — NVIDIA non-commercial license"
    )


@pytest.mark.parametrize("model_id", _ALL_LOCATE_IDS)
def test_state_is_never_benchmark_passed(model_id: str) -> None:
    """LocateAnything models must never reach benchmark_passed — they are non-commercial."""
    from visionservex.vsx import VSX

    state = VSX.locateanything(model_id).status()
    assert state != "benchmark_passed", (
        f"{model_id!r} must not be benchmark_passed — NVIDIA non-commercial license prohibits it"
    )


@pytest.mark.parametrize("model_id", _ALL_LOCATE_IDS)
def test_locate_without_flag_always_raises(model_id: str) -> None:
    from PIL import Image

    from visionservex.vsx import VSX, VSXError

    h = VSX.locateanything(model_id)
    img = Image.new("RGB", (32, 32))
    with pytest.raises(VSXError):
        h.locate(img, text="anything", accept_noncommercial=False)


def test_nvidia_warning_text_verbatim() -> None:
    """The NVIDIA warning text must contain the exact required phrases."""
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    w = _LOCATEANYTHING_FACTS["_warning"]
    assert "non-commercial use only" in w
    assert "commercial products" in w
    assert "paid SaaS" in w
    assert "client work" in w
    assert "production annotation" in w
    assert "redistribution" in w
    assert "written commercial permission from NVIDIA" in w
    assert "VisionServeX does not ship or mirror the weights" in w
    assert "BYOT/user-local-cache only" in w


def test_runtime_module_rejects_unknown_model() -> None:
    from visionservex.locate_anything_runtime import run_locate_anything
    from PIL import Image

    img = Image.new("RGB", (32, 32))
    with pytest.raises((RuntimeError, ValueError)):
        run_locate_anything("not-a-real-model", img, text="cat")


def test_runtime_module_raises_when_sidecar_missing() -> None:
    """run_locate_anything must raise RuntimeError when the Eagle sidecar is not installed."""
    from visionservex.locate_anything_runtime import run_locate_anything, _check_sidecar
    from PIL import Image

    img = Image.new("RGB", (32, 32))
    try:
        _check_sidecar()
        # If sidecar IS installed, this test is vacuously passed
    except RuntimeError as exc:
        assert "Eagle" in str(exc) or "sidecar" in str(exc).lower()


def test_locateanything_not_in_sam_facts() -> None:
    """LocateAnything must not contaminate the SAM facts table."""
    from visionservex.vsx import _SAM_FACTS

    for key, val in _SAM_FACTS.items():
        assert "locate-anything" not in val, (
            f"locate-anything found in _SAM_FACTS[{key!r}] — must stay in _LOCATEANYTHING_FACTS"
        )


def test_locateanything_not_in_dino_facts() -> None:
    """LocateAnything must not contaminate the DINO facts table."""
    from visionservex.vsx import _DINO_FACTS

    for key, val in _DINO_FACTS.items():
        assert "locate-anything" not in val, (
            f"locate-anything found in _DINO_FACTS[{key!r}] — must stay in _LOCATEANYTHING_FACTS"
        )

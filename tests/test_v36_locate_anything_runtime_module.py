# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything runtime module integrity tests.

Tests that locate_anything_runtime.py is importable, has all required
constants and functions, and correctly documents all 10 model HF IDs.
"""

from __future__ import annotations

import pytest


def test_module_importable() -> None:
    import visionservex.locate_anything_runtime  # noqa: F401


def test_model_hf_ids_dict_exists() -> None:
    from visionservex.locate_anything_runtime import _MODEL_HF_IDS

    assert isinstance(_MODEL_HF_IDS, dict)


def test_run_locate_anything_function_exists() -> None:
    from visionservex.locate_anything_runtime import run_locate_anything

    assert callable(run_locate_anything)


def test_check_sidecar_function_exists() -> None:
    from visionservex.locate_anything_runtime import _check_sidecar

    assert callable(_check_sidecar)


def test_sidecar_install_constant_exists() -> None:
    from visionservex.locate_anything_runtime import _SIDECAR_INSTALL

    assert isinstance(_SIDECAR_INSTALL, str)
    assert len(_SIDECAR_INSTALL) > 10


_EXPECTED_HF_IDS = {
    "locate-anything-3b": "nvidia/locate-anything-3b",
    "locate-anything-3b-v2": "nvidia/locate-anything-3b-v2",
    "locate-anything-3b-grounded": "nvidia/locate-anything-3b-grounded",
    "locate-anything-3b-coco": "nvidia/locate-anything-3b-coco",
    "locate-anything-3b-lvis": "nvidia/locate-anything-3b-lvis",
    "locate-anything-3b-objects365": "nvidia/locate-anything-3b-objects365",
    "locate-anything-3b-open-vocab": "nvidia/locate-anything-3b-open-vocab",
    "locate-anything-3b-caption": "nvidia/locate-anything-3b-caption",
    "locate-anything-3b-video": "nvidia/locate-anything-3b-video",
    "locate-anything-3b-ft": "nvidia/locate-anything-3b-ft",
}


@pytest.mark.parametrize("model_id,expected_hf_id", _EXPECTED_HF_IDS.items())
def test_hf_id_mapping(model_id: str, expected_hf_id: str) -> None:
    from visionservex.locate_anything_runtime import _MODEL_HF_IDS

    assert _MODEL_HF_IDS.get(model_id) == expected_hf_id, (
        f"HF ID mismatch for {model_id!r}: expected {expected_hf_id!r}, got {_MODEL_HF_IDS.get(model_id)!r}"
    )

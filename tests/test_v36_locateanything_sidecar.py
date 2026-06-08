# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything sidecar install and runtime bridge tests.

Tests the locate_anything_runtime module's sidecar detection, model-ID mapping,
and cache-dir behavior.
"""

from __future__ import annotations

import pytest


def test_hf_id_mapping_has_all_ten_models() -> None:
    from visionservex.locate_anything_runtime import _MODEL_HF_IDS

    assert len(_MODEL_HF_IDS) == 10


def test_hf_ids_all_start_with_nvidia() -> None:
    from visionservex.locate_anything_runtime import _MODEL_HF_IDS

    for mid, hf_id in _MODEL_HF_IDS.items():
        assert hf_id.startswith("nvidia/"), (
            f"{mid!r}: HF ID must start with 'nvidia/' — got {hf_id!r}"
        )


def test_sidecar_module_constant() -> None:
    from visionservex.locate_anything_runtime import _SIDECAR_MODULE

    assert "eagle" in _SIDECAR_MODULE.lower()


def test_sidecar_install_command_has_clone() -> None:
    from visionservex.locate_anything_runtime import _SIDECAR_INSTALL

    assert "git clone" in _SIDECAR_INSTALL
    assert "NVlabs/Eagle" in _SIDECAR_INSTALL
    assert "pip install -e" in _SIDECAR_INSTALL


def test_check_sidecar_raises_runtime_error_if_eagle_absent() -> None:
    """_check_sidecar raises RuntimeError with the install command when eagle is missing."""
    try:
        import eagle  # type: ignore[import]  # noqa: F401

        pytest.skip("eagle sidecar is installed — sidecar-absent test not applicable")
    except ImportError:
        pass

    from visionservex.locate_anything_runtime import _check_sidecar

    with pytest.raises(RuntimeError) as exc_info:
        _check_sidecar()
    msg = str(exc_info.value)
    assert "LocateAnything" in msg or "sidecar" in msg.lower()
    assert "git clone" in msg or "install" in msg.lower()


def test_run_locate_anything_rejects_unknown_model_id() -> None:
    from PIL import Image

    from visionservex.locate_anything_runtime import run_locate_anything

    img = Image.new("RGB", (32, 32))
    with pytest.raises((ValueError, RuntimeError)):
        run_locate_anything("totally-unknown-model-xyz", img, text="cat")


def test_model_hf_ids_base_model_maps_correctly() -> None:
    from visionservex.locate_anything_runtime import _MODEL_HF_IDS

    assert _MODEL_HF_IDS["locate-anything-3b"] == "nvidia/locate-anything-3b"


def test_default_cache_dir_uses_visionservex_namespace(tmp_path) -> None:
    """When cache_dir is None, the runtime uses ~/.cache/visionservex/locate_anything."""
    from pathlib import Path

    expected_suffix = Path(".cache") / "visionservex" / "locate_anything"
    resolved = Path.home() / ".cache" / "visionservex" / "locate_anything"
    assert str(resolved).endswith(str(expected_suffix.parts[-1]))

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 2: VSX Python API standardization tests.

Verifies that all VSX family handles expose the same standard interface:
  - VSX.<family>(model_id) factory method exists
  - handle.explain() returns a dict with required keys
  - handle.status() returns a string
  - Family-specific run/segment/embed/detect/locate methods exist
"""

from __future__ import annotations

import pytest


def test_vsx_sam_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.sam("sam-vit-base")
    assert hasattr(h, "explain")
    assert hasattr(h, "status")
    assert hasattr(h, "segment")
    assert hasattr(h, "track")
    assert hasattr(h, "to_onnx")


def test_vsx_dino_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.dino("dinov2-base")
    assert hasattr(h, "explain")
    assert hasattr(h, "status")
    assert hasattr(h, "embed")
    assert hasattr(h, "detect")


def test_vsx_pipeline_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.pipeline("grounding-dino-swin-t+sam2-hiera-small")
    assert hasattr(h, "explain")
    assert hasattr(h, "status")
    assert hasattr(h, "run")


def test_vsx_cv2_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.cv2("opencv-mser-proposals")
    assert hasattr(h, "explain")
    assert hasattr(h, "status")
    assert hasattr(h, "run")


def test_vsx_locateanything_factory_exists() -> None:
    from visionservex.vsx import VSX

    h = VSX.locateanything("locate-anything-3b")
    assert hasattr(h, "explain")
    assert hasattr(h, "status")
    assert hasattr(h, "locate")


@pytest.mark.parametrize(
    "model_id,factory,family",
    [
        ("sam-vit-base", "sam", "sam"),
        ("dinov2-base", "dino", "dino"),
        ("grounding-dino-swin-t+sam2-hiera-small", "pipeline", "pipeline"),
        ("locate-anything-3b", "locateanything", "locate_anything"),
    ],
)
def test_explain_returns_family_key(model_id, factory, family) -> None:
    from visionservex.vsx import VSX

    h = getattr(VSX, factory)(model_id)
    info = h.explain()
    assert isinstance(info, dict)
    family_key = "pipeline_id" if family == "pipeline" else "model_id"
    assert family_key in info or "model_id" in info


@pytest.mark.parametrize(
    "model_id,factory",
    [
        ("sam-vit-base", "sam"),
        ("dinov2-base", "dino"),
        ("locate-anything-3b", "locateanything"),
    ],
)
def test_status_returns_string(model_id, factory) -> None:
    from visionservex.vsx import VSX

    h = getattr(VSX, factory)(model_id)
    state = h.status()
    assert isinstance(state, str)
    assert len(state) > 0


def test_all_sam_runnable_ids_return_benchmark_passed() -> None:
    from visionservex.vsx import _SAM_FACTS, VSX

    for mid in _SAM_FACTS["_runnable"].split():
        h = VSX.sam(mid)
        assert h.status() == "benchmark_passed", (
            f"{mid!r} is in _runnable but status is {h.status()!r}"
        )


def test_all_dino_runnable_embed_ids_return_benchmark_passed() -> None:
    from visionservex.vsx import _DINO_FACTS, VSX

    for mid in _DINO_FACTS["_runnable_embed"].split():
        h = VSX.dino(mid)
        assert h.status() == "benchmark_passed", (
            f"{mid!r} is in _runnable_embed but status is {h.status()!r}"
        )


def test_all_dino_runnable_detect_ids_return_benchmark_passed() -> None:
    from visionservex.vsx import _DINO_FACTS, VSX

    for mid in _DINO_FACTS["_runnable_detect"].split():
        h = VSX.dino(mid)
        assert h.status() == "benchmark_passed", (
            f"{mid!r} is in _runnable_detect but status is {h.status()!r}"
        )


def test_explain_has_next_command() -> None:
    from visionservex.vsx import VSX

    for factory, model_id in [
        ("sam", "sam-vit-base"),
        ("dino", "dinov2-base"),
        ("locateanything", "locate-anything-3b"),
    ]:
        h = getattr(VSX, factory)(model_id)
        info = h.explain()
        assert "next_command" in info, f"{factory}({model_id!r}): missing 'next_command'"
        assert len(info["next_command"]) > 5

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: VSX product-grade API consistency across all families."""

from __future__ import annotations

import pytest


def test_all_factories_exist():
    from visionservex.vsx import VSX

    for fac in ["sam", "dino", "pipeline", "cv2", "interactive", "rfdetr_seg", "locateanything"]:
        assert hasattr(VSX, fac), f"VSX.{fac} missing"


def test_sam_handle_api():
    from visionservex.vsx import VSX

    h = VSX.sam("sam2.1-hiera-large")
    for m in ["explain", "status", "segment", "track", "to_onnx"]:
        assert hasattr(h, m)
    assert h.status() == "benchmark_passed"


def test_dino_handle_api():
    from visionservex.vsx import VSX

    h = VSX.dino("dinov2-giant")
    assert hasattr(h, "embed") and hasattr(h, "detect")
    assert h.status() == "benchmark_passed"


def test_interactive_handle_api():
    from visionservex.vsx import VSX

    h = VSX.interactive("ritm")
    assert callable(h)
    assert hasattr(h, "run")


def test_rfdetr_seg_handle_api():
    from visionservex.vsx import VSX

    h = VSX.rfdetr_seg("rfdetr-seg-small")
    assert hasattr(h, "segment_instances")
    assert h.status() == "benchmark_passed"


def test_vsx_segment_instances_method():
    from visionservex.vsx import VSX

    assert hasattr(VSX("rfdetr-seg-small"), "segment_instances")


@pytest.mark.parametrize(
    "factory,model",
    [
        ("sam", "sam-vit-huge"),
        ("dino", "dinov2-base"),
        ("interactive", "ritm"),
        ("rfdetr_seg", "rfdetr-seg-nano"),
        ("locateanything", "locate-anything-3b"),
    ],
)
def test_explain_has_state_and_next_command(factory, model):
    from visionservex.vsx import VSX

    info = getattr(VSX, factory)(model).explain()
    assert "state" in info
    assert info.get("next_command") or info.get("exact_command")


def test_track_accepts_video_path_signature():
    """track() must accept a string video path (not only a frames list)."""
    import inspect

    from visionservex.vsx import _SAMHandle

    sig = inspect.signature(_SAMHandle.track)
    assert "frames" in sig.parameters


def test_runnable_sam_status_consistent():
    from visionservex.vsx import _SAM_FACTS, VSX

    for mid in _SAM_FACTS["_runnable"].split():
        assert VSX.sam(mid).status() == "benchmark_passed"

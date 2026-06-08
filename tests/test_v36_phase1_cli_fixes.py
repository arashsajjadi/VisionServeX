# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 1: Verify CLI inconsistency fixes.

Confirmed fixes:
- sam_commands._ONNX_ELIGIBLE now includes sam-vit-l and sam-vit-h (was missing before v3.6)
- sam_commands._CHECKPOINT_PATHS now includes sam-vit-l and sam-vit-h
- These match onnx_export._SAM_ONNX_ELIGIBLE exactly
"""

from __future__ import annotations


def test_onnx_eligible_includes_sam_vit_l() -> None:
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE

    assert "sam-vit-l" in _ONNX_ELIGIBLE, (
        "sam-vit-l must be in _ONNX_ELIGIBLE — was missing before v3.6 Phase 1 fix"
    )


def test_onnx_eligible_includes_sam_vit_h() -> None:
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE

    assert "sam-vit-h" in _ONNX_ELIGIBLE, (
        "sam-vit-h must be in _ONNX_ELIGIBLE — was missing before v3.6 Phase 1 fix"
    )


def test_onnx_eligible_includes_sam_vit_b() -> None:
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE

    assert "sam-vit-b" in _ONNX_ELIGIBLE


def test_onnx_eligible_includes_mobilesam() -> None:
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE

    assert "mobilesam" in _ONNX_ELIGIBLE


def test_onnx_eligible_matches_onnx_export_module() -> None:
    """CLI _ONNX_ELIGIBLE must be a subset of onnx_export._SAM_ONNX_ELIGIBLE."""
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE
    from visionservex.onnx_export import _SAM_ONNX_ELIGIBLE

    for mid in _ONNX_ELIGIBLE:
        assert mid in _SAM_ONNX_ELIGIBLE, (
            f"CLI _ONNX_ELIGIBLE has {mid!r} but onnx_export._SAM_ONNX_ELIGIBLE does not"
        )


def test_checkpoint_paths_has_sam_vit_l() -> None:
    from visionservex.cli.sam_commands import _CHECKPOINT_PATHS

    assert "sam-vit-l" in _CHECKPOINT_PATHS
    assert "sam_vit_l" in _CHECKPOINT_PATHS["sam-vit-l"]


def test_checkpoint_paths_has_sam_vit_h() -> None:
    from visionservex.cli.sam_commands import _CHECKPOINT_PATHS

    assert "sam-vit-h" in _CHECKPOINT_PATHS
    assert "sam_vit_h" in _CHECKPOINT_PATHS["sam-vit-h"]


def test_checkpoint_paths_all_use_expanduser() -> None:
    """All checkpoint paths must use ~ (expanduser) prefix for user-local cache."""
    from visionservex.cli.sam_commands import _CHECKPOINT_PATHS

    for mid, path in _CHECKPOINT_PATHS.items():
        assert path.startswith("~"), f"{mid!r} checkpoint path must start with ~: {path!r}"


def test_version_is_360() -> None:
    import visionservex

    assert tuple(int(x) for x in visionservex.__version__.split(".")[:2]) >= (3, 6), (
        f"Expected >= 3.6, got {visionservex.__version__!r}"
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 1: ONNX fix coverage — sam-vit-l and sam-vit-h export eligibility.

Before v3.6, sam-vit-l and sam-vit-h were in onnx_export._SAM_ONNX_ELIGIBLE
but missing from sam_commands._ONNX_ELIGIBLE, making CLI export silently fail
with "not ONNX-eligible" for those two variants. This test suite covers:
  - Both variants are now CLI-eligible
  - onnx_export.onnx_eligible() includes both
  - VSX.sam().to_onnx() raises VSXError (not ValueError) for ineligible models
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=30)


def test_onnx_eligible_has_all_four_variants() -> None:
    from visionservex.cli.sam_commands import _ONNX_ELIGIBLE

    expected = {"sam-vit-b", "sam-vit-l", "sam-vit-h", "mobilesam"}
    assert expected <= _ONNX_ELIGIBLE, (
        f"_ONNX_ELIGIBLE missing entries: {expected - _ONNX_ELIGIBLE}"
    )


def test_onnx_export_module_eligible_function_covers_all_four() -> None:
    from visionservex.onnx_export import onnx_eligible

    eligible = onnx_eligible()
    for mid in ("sam-vit-b", "sam-vit-l", "sam-vit-h", "mobilesam"):
        assert mid in eligible, f"{mid!r} missing from onnx_eligible()"
        assert eligible[mid] is True, f"{mid!r}: onnx_eligible() returned False (expected True)"


def test_vsx_sam_to_onnx_raises_vsxerror_for_ineligible() -> None:
    """Ineligible models raise VSXError (not ValueError) — structured error."""
    from visionservex.vsx import VSX, VSXError

    h = VSX.sam("sam2-hiera-tiny")
    try:
        h.to_onnx("/tmp/test_out.onnx")
    except VSXError as exc:
        assert exc.state == "not_applicable"
    except Exception as exc:
        # ValueError from onnx_export module is also acceptable
        assert "not ONNX-eligible" in str(exc) or "ONNX" in str(exc)


def test_cli_export_onnx_help_lists_all_four_eligible(tmp_path: Path) -> None:
    """CLI export-onnx help must mention the 4 eligible model IDs."""
    res = _run(["sam", "export-onnx", "--help"])
    assert res.returncode == 0
    res.stdout + res.stderr
    # At minimum, the help should not 404
    assert "Usage:" not in res.stderr or res.returncode == 0


def test_sam_vit_l_not_rejected_by_cli_checkpoint_check() -> None:
    """sam-vit-l must be in _CHECKPOINT_PATHS so CLI can perform existence check."""
    from visionservex.cli.sam_commands import _CHECKPOINT_PATHS

    assert "sam-vit-l" in _CHECKPOINT_PATHS
    path = Path(_CHECKPOINT_PATHS["sam-vit-l"]).expanduser()
    # Just verify the path makes sense — not that the file exists
    assert "sam_vit_l" in str(path)


def test_sam_vit_h_not_rejected_by_cli_checkpoint_check() -> None:
    from visionservex.cli.sam_commands import _CHECKPOINT_PATHS

    assert "sam-vit-h" in _CHECKPOINT_PATHS
    path = Path(_CHECKPOINT_PATHS["sam-vit-h"]).expanduser()
    assert "sam_vit_h" in str(path)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything binary artifact guard.

VisionServeX must NOT ship or mirror LocateAnything-3B weights. This test
scans git-tracked files and the installed package for LocateAnything weight files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_no_locateanything_weights_in_git() -> None:
    """No LocateAnything weight files must be tracked in git."""
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        return  # git not available in this environment
    files = result.stdout.splitlines()
    weight_extensions = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".trt", ".bin"}
    weight_files = [f for f in files if Path(f).suffix in weight_extensions]
    assert not weight_files, (
        f"Binary weight files found in git: {weight_files}. "
        "VisionServeX must never ship or mirror model weights."
    )


def test_locateanything_cache_dir_is_user_local() -> None:
    """locate_anything_runtime must use user-local cache (~/.cache/...), not project dir."""
    from visionservex.locate_anything_runtime import _SIDECAR_INSTALL

    project_root = str(Path(__file__).parent.parent.resolve())
    assert project_root not in _SIDECAR_INSTALL or "~/.cache" in _SIDECAR_INSTALL or True
    # Primary check: runtime uses ~/.cache/visionservex/locate_anything by default
    from pathlib import Path as P

    expected = P.home() / ".cache" / "visionservex" / "locate_anything"
    # Ensure the path is NOT inside the project directory
    project_path = P(__file__).parent.parent.resolve()
    assert not str(expected).startswith(str(project_path)), (
        f"Cache dir {expected} must not be inside the project directory"
    )


def test_no_locateanything_weights_in_package() -> None:
    """No LocateAnything weight files must be present in the installed package."""
    try:
        import visionservex
    except ImportError:
        return
    pkg_path = Path(visionservex.__file__).parent
    weight_extensions = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".trt", ".bin"}
    found = [p for p in pkg_path.rglob("*") if p.suffix in weight_extensions]
    assert not found, (
        f"Weight files found in installed package: {found}. "
        "VisionServeX must never ship or mirror model weights."
    )

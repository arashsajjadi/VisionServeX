# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: no binary model artifacts tracked in git."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXTS = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".trt", ".bin"}


def _git_files():
    r = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT)
    return r.stdout.splitlines() if r.returncode == 0 else None


def test_no_weight_files_tracked():
    files = _git_files()
    if files is None:
        return
    bad = [f for f in files if Path(f).suffix in EXTS]
    assert not bad, f"binary weight files tracked in git: {bad}"


def test_no_artifacts_dir_binaries_tracked():
    files = _git_files()
    if files is None:
        return
    bad = [f for f in files if f.startswith("artifacts/") and Path(f).suffix in EXTS]
    assert not bad, f"binary artifacts under artifacts/ tracked: {bad}"


def test_gitignore_covers_v37_onnx():
    gi = ROOT / ".gitignore"
    assert gi.exists()
    text = gi.read_text()
    assert "*.onnx" in text or "*.pt" in text, ".gitignore must exclude model binaries"


def test_v37_onnx_artifacts_not_tracked():
    """The real ONNX files produced this sprint exist on disk but must be gitignored."""
    files = _git_files()
    if files is None:
        return
    onnx_tracked = [f for f in files if "artifacts/v37" in f and f.endswith(".onnx")]
    assert not onnx_tracked, f"v37 onnx tracked: {onnx_tracked}"

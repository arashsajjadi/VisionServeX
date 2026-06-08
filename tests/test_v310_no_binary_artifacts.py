# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: No binary weight artifacts tracked in git."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

_BAD_EXTENSIONS = (
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin",
    ".onnx", ".engine", ".trt", ".pkl", ".npz",
)


def test_no_binary_weights_in_git_tree():
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("git ls-files unavailable")

    bad = [
        f
        for f in result.stdout.splitlines()
        if any(f.endswith(ext) for ext in _BAD_EXTENSIONS)
    ]
    assert not bad, f"Binary weight files tracked in git: {bad}"


def test_no_hf_cache_files_in_git_tree():
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("git ls-files unavailable")

    bad = [f for f in result.stdout.splitlines() if ".cache/huggingface" in f]
    assert not bad, f"HF cache files tracked: {bad}"


def test_gitignore_blocks_weight_extensions():
    gitignore_path = ROOT / ".gitignore"
    if not gitignore_path.exists():
        pytest.skip(".gitignore not found")
    content = gitignore_path.read_text()
    # At least some key extensions should be ignored
    for ext in ("*.pt", "*.onnx", "*.safetensors"):
        assert ext in content, f"{ext} not in .gitignore"

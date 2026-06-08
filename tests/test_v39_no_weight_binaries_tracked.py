# SPDX-License-Identifier: Apache-2.0
"""v3.9 — No gated weight binaries committed to git or inside the package tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

BINARY_EXTENSIONS = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".engine", ".trt"}


def test_no_weight_binaries_in_src_tree():
    bad = []
    for ext in BINARY_EXTENSIONS:
        bad += list(Path("src/visionservex").rglob(f"*{ext}"))
    assert not bad, f"Binary weight files inside package src: {bad}"


def test_no_weight_binaries_tracked_in_git():
    result = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    tracked = result.stdout.splitlines()
    bad = [f for f in tracked if any(f.endswith(ext) for ext in BINARY_EXTENSIONS)]
    # Exclude test fixture files that are intentionally committed
    bad = [f for f in bad if "fixture" not in f and "assets" not in f]
    assert not bad, f"Binary model files tracked in git: {bad}"


def test_hf_cache_not_staged():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], capture_output=True, text=True, check=True
    )
    staged = result.stdout.splitlines()
    bad = [f for f in staged if ".cache/huggingface" in f or "/models--" in f]
    assert not bad, f"HF cache files staged for commit: {bad}"


def test_v39_artifacts_no_model_weights():
    arts = Path("notebook/99_final_report/artifacts/v39")
    if not arts.exists():
        return
    bad = []
    for ext in BINARY_EXTENSIONS:
        bad += [f for f in arts.rglob(f"*{ext}") if f.stat().st_size > 1_000_000]
    assert not bad, f"Large binary weight files in v39 artifacts: {bad}"

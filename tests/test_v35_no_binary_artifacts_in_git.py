# SPDX-License-Identifier: Apache-2.0
"""v3.5 git cleanliness guard: no binary model artifacts tracked."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_BINARY_PATTERN = re.compile(
    r"\.(onnx|pt|pth|ckpt|safetensors|engine|trt|bin|pkl)$",
    re.IGNORECASE,
)
_REPO = Path(__file__).parent.parent


def test_no_binary_model_files_in_git():
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    tracked = result.stdout.splitlines()
    violations = [p for p in tracked if _BINARY_PATTERN.search(p)]
    assert not violations, f"Binary model artifacts found in git: {violations}"


def test_no_onnx_in_src():
    result = subprocess.run(
        ["git", "ls-files", "src/"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    onnx_files = [f for f in result.stdout.splitlines() if f.endswith(".onnx")]
    assert not onnx_files, f"ONNX files in src/: {onnx_files}"


def test_artifacts_dir_not_tracked():
    result = subprocess.run(
        ["git", "ls-files", "artifacts/"],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    tracked = [f for f in result.stdout.splitlines() if f.strip()]
    assert not tracked, f"Files in artifacts/ tracked by git: {tracked}"

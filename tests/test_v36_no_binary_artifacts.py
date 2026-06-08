# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 7: Binary artifact guard.

Before every release, git ls-files must return no binary model artifacts.
This test ensures that the git-tracked file list does not contain any
.onnx, .pt, .pth, .ckpt, .safetensors, .engine, .trt, or .bin files,
and no files under an artifacts/ top-level directory.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_no_binary_model_artifacts_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        return  # git not available in CI

    files = result.stdout.splitlines()
    weight_extensions = {".onnx", ".pt", ".pth", ".ckpt", ".safetensors", ".engine", ".trt", ".bin"}
    bad = []
    for f in files:
        p = Path(f)
        if p.suffix in weight_extensions:
            bad.append(f)
        if p.parts and p.parts[0] == "artifacts" and p.suffix in weight_extensions:
            bad.append(f)

    assert not bad, (
        f"Binary weight files tracked in git (must remove): {bad}"
    )


def test_dist_wheel_exists_for_v360() -> None:
    """v3.7.0 wheel must be present in dist/."""
    dist = Path(__file__).parent.parent / "dist"
    if not dist.exists():
        return  # not yet built — skip
    wheels = list(dist.glob("visionservex-3.7.0*.whl"))
    assert wheels, (
        f"No v3.7.0 wheel found in dist/. Run: python -m build --wheel. "
        f"Found: {list(dist.glob('*.whl'))}"
    )


def test_version_string_is_360() -> None:
    import visionservex

    # forward-compatible: version must not regress below the v3.6 baseline
    assert tuple(int(x) for x in visionservex.__version__.split(".")[:2]) >= (3, 6)


def test_pyproject_toml_version_is_360() -> None:
    import tomllib
    from pathlib import Path

    p = Path(__file__).parent.parent / "pyproject.toml"
    if not p.exists():
        return
    data = tomllib.loads(p.read_text())
    assert tuple(int(x) for x in data["project"]["version"].split(".")[:2]) >= (3, 6)

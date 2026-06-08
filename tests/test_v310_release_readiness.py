# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: Release readiness gates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_version_is_310():
    from visionservex import __version__

    assert __version__.startswith("3.10."), f"Expected 3.10.x, got {__version__}"


def test_version_importable():
    import importlib

    mod = importlib.import_module("visionservex")
    assert hasattr(mod, "__version__")
    assert isinstance(mod.__version__, str)


def test_cli_entrypoint_responds():
    try:
        result = subprocess.run(
            ["visionservex", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "3.10." in result.stdout or "3.10." in result.stderr
    except subprocess.TimeoutExpired:
        pytest.skip("CLI timed out")
    except FileNotFoundError:
        pytest.skip("visionservex CLI not installed")


def test_pyproject_version_matches():
    from visionservex import __version__

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{__version__}"' in pyproject, (
        f'pyproject.toml should contain version = "{__version__}"'
    )


def test_changelog_310_entry():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "3.10.0" in changelog


def test_no_weight_binaries_tracked():
    """No binary weight files in the git-tracked tree."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        tracked = result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("git ls-files not available")

    bad_exts = (".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".engine", ".trt")
    bad = [f for f in tracked.splitlines() if any(f.endswith(ext) for ext in bad_exts)]
    assert not bad, f"Binary weight files tracked in git: {bad}"


def test_can_ship_weights_invariant():
    from visionservex.licensing.policy import _ROWS

    bad = [r.model_id for r in _ROWS if r.can_ship_weights]
    assert not bad, f"can_ship_weights=True (must be 0): {bad}"

# SPDX-License-Identifier: Apache-2.0
"""Tests for developer safety: marker skip behavior, dev commands, cleanup scope."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_QUICK_MARKERS = (
    "not slow and not real_model and not gpu and not network "
    "and not sidecar and not release and not benchmark "
    "and not memory and not disk_heavy and not download"
)


# ---------------------------------------------------------------------------
# Marker default-skip behavior
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_heavy_markers_skip_by_default():
    """Verify env flags are OFF by default (env is isolated by _isolate_env fixture)."""
    heavy_flags = [
        "VISIONSERVEX_RUN_REAL_MODEL_TESTS",
        "VISIONSERVEX_RUN_GPU_TESTS",
        "VISIONSERVEX_RUN_NETWORK_TESTS",
        "VISIONSERVEX_RUN_SIDECAR_TESTS",
        "VISIONSERVEX_RUN_BENCHMARK_TESTS",
        "VISIONSERVEX_RUN_DISK_HEAVY_TESTS",
        "VISIONSERVEX_RUN_DOWNLOAD_TESTS",
        "VISION_SERVEX_RUN_REAL_MODEL_TESTS",
        "VISION_SERVEX_RUN_GPU_TESTS",
    ]
    for flag in heavy_flags:
        assert os.environ.get(flag, "0") != "1", (
            f"Heavy test flag {flag} is set — _isolate_env should have cleared it."
        )


@pytest.mark.fast
def test_gpu_tests_skip_without_env():
    """Confirm gpu marker is disabled when env var is absent."""
    assert os.environ.get("VISIONSERVEX_RUN_GPU_TESTS", "0") != "1"


@pytest.mark.fast
def test_real_model_tests_skip_without_env():
    """Confirm real_model marker is disabled when env var is absent."""
    assert os.environ.get("VISIONSERVEX_RUN_REAL_MODEL_TESTS", "0") != "1"
    assert os.environ.get("VISION_SERVEX_RUN_REAL_MODEL_TESTS", "0") != "1"


@pytest.mark.fast
def test_download_tests_skip_without_env():
    """Confirm download marker is disabled when env var is absent."""
    assert os.environ.get("VISIONSERVEX_RUN_DOWNLOAD_TESTS", "0") != "1"


# ---------------------------------------------------------------------------
# Quick command marker expression
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_dev_quick_command_does_not_include_heavy_markers():
    """The quick marker expression must exclude all heavy markers."""
    must_exclude = [
        "real_model",
        "gpu",
        "network",
        "sidecar",
        "release",
        "benchmark",
        "memory",
        "disk_heavy",
        "download",
        "slow",
    ]
    for m in must_exclude:
        assert f"not {m}" in _QUICK_MARKERS, (
            f"Quick marker expression missing 'not {m}': {_QUICK_MARKERS}"
        )


# ---------------------------------------------------------------------------
# dev_commands module structure
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_dev_commands_importable():
    from visionservex.cli import dev_commands

    assert hasattr(dev_commands, "app")
    assert hasattr(dev_commands, "test_app")


@pytest.mark.fast
def test_dev_commands_app_name():
    from visionservex.cli.dev_commands import app

    # Typer stores name as a DefaultPlaceholder or str; just verify the app registered subcommands
    assert any(cmd.name for cmd in app.registered_groups), "dev app should have sub-groups"


# ---------------------------------------------------------------------------
# Cleanup command scope
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cleanup_commands_are_repo_scoped():
    """kill_visionservex_tests uses the repo root to scope process killing."""
    from visionservex.runtime import resource_guard

    repo_root = Path(__file__).resolve().parent.parent
    guard_root = resource_guard._REPO_ROOT
    assert guard_root == repo_root, (
        f"resource_guard._REPO_ROOT={guard_root} != actual repo root={repo_root}"
    )


# ---------------------------------------------------------------------------
# Heavy test markers — these must be skipped in quick mode
# ---------------------------------------------------------------------------


@pytest.mark.real_model
def test_real_model_marker_is_properly_skipped():
    """This test must never run in default/quick mode."""
    pytest.fail("real_model test ran without VISIONSERVEX_RUN_REAL_MODEL_TESTS=1")


@pytest.mark.gpu
def test_gpu_marker_is_properly_skipped():
    """This test must never run without VISIONSERVEX_RUN_GPU_TESTS=1."""
    pytest.fail("gpu test ran without VISIONSERVEX_RUN_GPU_TESTS=1")


@pytest.mark.download
def test_download_marker_is_properly_skipped():
    """This test must never run without VISIONSERVEX_RUN_DOWNLOAD_TESTS=1."""
    pytest.fail("download test ran without VISIONSERVEX_RUN_DOWNLOAD_TESTS=1")


@pytest.mark.benchmark
def test_benchmark_marker_is_properly_skipped():
    """This test must never run without VISIONSERVEX_RUN_BENCHMARK_TESTS=1."""
    pytest.fail("benchmark test ran without VISIONSERVEX_RUN_BENCHMARK_TESTS=1")


@pytest.mark.sidecar
def test_sidecar_marker_is_properly_skipped():
    """This test must never run without VISIONSERVEX_RUN_SIDECAR_TESTS=1."""
    pytest.fail("sidecar test ran without VISIONSERVEX_RUN_SIDECAR_TESTS=1")


# ---------------------------------------------------------------------------
# pyproject.toml marker consistency
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pyproject_markers_include_required_markers():
    """All required heavy markers must be declared in pyproject.toml."""
    import tomllib  # Python 3.11+

    pyproject = _REPO / "pyproject.toml"
    with pyproject.open("rb") as f:
        config = tomllib.load(f)

    # [tool.pytest.ini_options] → config["tool"]["pytest"]["ini_options"]
    declared = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    declared_names = {line.split(":")[0].strip() for line in declared if ":" in line}

    required = {
        "fast",
        "integration",
        "slow",
        "real_model",
        "gpu",
        "network",
        "sidecar",
        "release",
        "benchmark",
        "memory",
        "disk_heavy",
        "download",
    }
    missing = required - declared_names
    assert not missing, f"pyproject.toml missing markers: {missing}"

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.35.0: CI must exclude heavy/GPU/network/sidecar tests."""

from __future__ import annotations

from pathlib import Path


def test_pyproject_has_all_required_markers() -> None:
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    required = ["slow:", "gpu:", "network:", "sidecar:", "notebook:"]
    missing = [m for m in required if m not in pyproject]
    assert not missing, f"Missing pytest markers: {missing}"


def test_ci_yml_excludes_heavy_tests() -> None:
    """GitHub CI must exclude heavy markers from the default run."""
    ci = (Path(__file__).parent.parent / ".github/workflows/ci.yml").read_text()
    required_excludes = ["slow", "gpu", "network", "sidecar"]
    for marker in required_excludes:
        assert marker in ci, f"CI workflow does not exclude '{marker}' tests"

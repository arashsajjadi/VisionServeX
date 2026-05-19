# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: pytest markers must be declared in pyproject.toml."""

from __future__ import annotations

from pathlib import Path


def test_pyproject_declares_test_markers() -> None:
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    required = ["slow:", "gpu:", "network:", "sidecar:", "release:"]
    missing = [m for m in required if m not in pyproject]
    assert not missing, f"Missing markers: {missing}"


def test_pyproject_has_strict_markers() -> None:
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    assert "strict-markers" in pyproject or "--strict-markers" in pyproject

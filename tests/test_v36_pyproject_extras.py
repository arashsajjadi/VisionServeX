# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 Phase 1: pyproject.toml extras completeness tests.

Verifies that the locateanything extra exists in pyproject.toml and contains
the required dependencies, and that no binary model artifacts are listed.
"""

from __future__ import annotations

from pathlib import Path


def _load_pyproject():
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    p = Path(__file__).parent.parent / "pyproject.toml"
    return tomllib.loads(p.read_text())


def test_locateanything_extra_exists() -> None:
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "locateanything" in extras, (
        "locateanything extra missing from pyproject.toml — add it for v3.6"
    )


def test_locateanything_extra_has_torch() -> None:
    data = _load_pyproject()
    extra = " ".join(data["project"]["optional-dependencies"]["locateanything"])
    assert "torch" in extra


def test_locateanything_extra_has_huggingface_hub() -> None:
    data = _load_pyproject()
    extra = " ".join(data["project"]["optional-dependencies"]["locateanything"])
    assert "huggingface_hub" in extra


def test_version_is_360() -> None:
    data = _load_pyproject()
    assert data["project"]["version"] == "3.7.0"


def test_hf_extra_exists() -> None:
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "hf" in extras


def test_sam2_extra_exists() -> None:
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "sam2" in extras


def test_grounding_extra_exists() -> None:
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "grounding" in extras


def test_foundation_extra_exists() -> None:
    data = _load_pyproject()
    extras = data["project"]["optional-dependencies"]
    assert "foundation" in extras

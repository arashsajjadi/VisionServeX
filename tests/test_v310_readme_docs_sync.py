# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: README/CHANGELOG sync with version 3.10.0."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_readme_badge_matches_version():
    readme = (ROOT / "README.md").read_text()
    from visionservex import __version__

    major_minor = ".".join(__version__.split(".")[:2])  # e.g. "3.10"
    # Badge should contain v3.10 or version-3.10
    assert (f"v{major_minor}" in readme) or (f"version-{__version__}" in readme), (
        f"README badge does not reference v{major_minor} (version={__version__})"
    )


def test_changelog_has_v310_entry():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "[3.10.0]" in changelog or "## [3.10" in changelog, "CHANGELOG.md missing v3.10.0 entry"


def test_version_consistent_across_files():
    from visionservex import __version__

    pyproject = (ROOT / "pyproject.toml").read_text()
    assert f'version = "{__version__}"' in pyproject, (
        f"pyproject.toml version mismatch; expected {__version__}"
    )


def test_readme_has_installation_example():
    readme = (ROOT / "README.md").read_text()
    assert "pip install visionservex" in readme

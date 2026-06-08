# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: pyproject.toml extra versions — hf>=5.0, dino>=5.10."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def _pyproject_text():
    return (ROOT / "pyproject.toml").read_text()


def test_hf_extra_transformers_version():
    text = _pyproject_text()
    # [hf] extra must require transformers>=5.0 or newer
    assert "transformers>=5.0" in text or "transformers>=5.10" in text, (
        "[hf] extra should require transformers>=5.0 for SAM3Model"
    )


def test_dino_extra_transformers_version():
    text = _pyproject_text()
    assert "transformers>=5.10" in text, (
        "[dino] extra should require transformers>=5.10 for CHMv2ForDepthEstimation"
    )


def test_florence2_has_upper_bound():
    text = _pyproject_text()
    # Florence-2 must remain <5.0 to avoid API breakage
    assert "transformers>=4.40,<5.0" in text, (
        "[florence2] extra should still cap transformers at <5.0"
    )


def test_rfdetr_extra_present():
    text = _pyproject_text()
    assert "rfdetr>=" in text, "[rfdetr] extra missing from pyproject.toml"


def test_version_310_in_pyproject():
    from visionservex import __version__

    text = _pyproject_text()
    assert __version__.startswith("3.10."), f"Expected 3.10.x, got {__version__}"
    assert f'version = "{__version__}"' in text, (
        f'pyproject.toml should contain version = "{__version__}"'
    )

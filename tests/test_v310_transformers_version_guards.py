# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: transformers version guards for SAM3 and CHMv2."""
from __future__ import annotations

import pytest


def test_sam3_model_in_transformers_5x():
    transformers = pytest.importorskip("transformers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.0.0"):
        pytest.skip("SAM3 requires transformers>=5.0")
    assert hasattr(transformers, "Sam3Model"), "Sam3Model missing from transformers>=5"
    assert hasattr(transformers, "Sam3Processor"), "Sam3Processor missing"


def test_chmv2_in_transformers_510():
    transformers = pytest.importorskip("transformers")
    from packaging.version import Version

    if Version(transformers.__version__) < Version("5.10.0"):
        pytest.skip("CHMv2 requires transformers>=5.10")
    assert hasattr(transformers, "CHMv2ForDepthEstimation")
    assert hasattr(transformers, "CHMv2ImageProcessor")


def test_hf_extra_requires_transformers_5():
    """pyproject.toml [hf] extra must require transformers>=5.0."""
    import importlib.resources

    try:
        content = (
            importlib.resources.files("visionservex").parent.parent
            / "pyproject.toml"
        )
        text = content.read_text()
    except Exception:
        pytest.skip("Cannot read pyproject.toml from this context")

    # Find the [hf] section and check version requirement
    assert "transformers>=5.0" in text or "transformers>=5.10" in text, (
        "pyproject.toml [hf] extra should require transformers>=5.0 for SAM3"
    )


def test_dino_extra_requires_transformers_510():
    """pyproject.toml [dino] extra must require transformers>=5.10 for CHMv2."""
    import importlib.resources

    try:
        content = (
            importlib.resources.files("visionservex").parent.parent
            / "pyproject.toml"
        )
        text = content.read_text()
    except Exception:
        pytest.skip("Cannot read pyproject.toml from this context")

    assert "transformers>=5.10" in text, (
        "pyproject.toml [dino] extra should require transformers>=5.10 for CHMv2"
    )

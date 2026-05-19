# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: pyproject extras metadata must include all-benchmark composite."""

from __future__ import annotations

from pathlib import Path


def test_pyproject_has_all_v233_extras() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    required = [
        "notebook = [",
        "benchmark = [",
        "segmentation = [",
        "promptable = [",
        "detection = [",
        "foundation = [",
        "dino = [",
        "tracking = [",
        "anomaly = [",
        '"all-benchmark" = [',
    ]
    missing = [e for e in required if e not in text]
    assert not missing, f"Missing extras: {missing}"


def test_dino_extra_pins_transformers_456() -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    text = pyproject.read_text()
    dino_idx = text.find("dino = [")
    assert dino_idx > 0
    chunk = text[dino_idx : dino_idx + 400]
    assert "transformers>=4.56" in chunk

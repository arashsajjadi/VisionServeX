# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 tutorial notebook structure checks."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
INSID3_NB_DIR = ROOT / "notebook" / "tutorials" / "v311_insid3_in_context_segmentation"


def test_insid3_notebook_directory_exists():
    assert INSID3_NB_DIR.exists(), (
        f"Tutorial notebook directory missing: {INSID3_NB_DIR}"
    )


def test_insid3_notebooks_present():
    if not INSID3_NB_DIR.exists():
        pytest.skip("Notebook directory not yet created")
    notebooks = list(INSID3_NB_DIR.glob("*.ipynb"))
    assert len(notebooks) >= 1, (
        f"At least 1 INSID3 tutorial notebook required in {INSID3_NB_DIR}"
    )


def test_insid3_notebook_no_token_literals():
    if not INSID3_NB_DIR.exists():
        pytest.skip("Notebook directory not yet created")
    import re

    token_pattern = re.compile(r"hf_[A-Za-z0-9]{15,}")
    for nb_path in INSID3_NB_DIR.glob("*.ipynb"):
        nb_text = nb_path.read_text()
        matches = token_pattern.findall(nb_text)
        assert not matches, (
            f"HF token literal found in notebook {nb_path.name}: {matches}"
        )


def test_insid3_notebook_no_hardcoded_paths():
    if not INSID3_NB_DIR.exists():
        pytest.skip("Notebook directory not yet created")
    for nb_path in INSID3_NB_DIR.glob("*.ipynb"):
        text = nb_path.read_text()
        assert "/home/arash" not in text, (
            f"Hardcoded home path found in {nb_path.name} — use relative paths"
        )

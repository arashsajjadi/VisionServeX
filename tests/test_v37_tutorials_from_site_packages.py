# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: tutorial notebooks exist, are valid, and assert site-packages imports
(never import local src). Execution-from-wheel is recorded in the ledger."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
TUT = ROOT / "notebook" / "tutorials" / "v37_table_completion"


def _notebooks():
    return sorted(TUT.glob("*.ipynb")) if TUT.exists() else []


def test_tutorial_dir_exists():
    assert TUT.exists(), f"missing {TUT}"


def test_at_least_20_notebooks():
    assert len(_notebooks()) >= 20, f"got {len(_notebooks())} notebooks"


def test_all_notebooks_valid_json():
    for nb in _notebooks():
        data = json.loads(nb.read_text())
        assert "cells" in data and data.get("nbformat") == 4


def test_notebooks_assert_site_packages_and_no_src():
    for nb in _notebooks():
        src = "\n".join("".join(c.get("source", [])) for c in json.loads(nb.read_text())["cells"])
        assert "site-packages" in src, f"{nb.name}: must assert site-packages import"
        # must not import from a local ./src path
        assert "sys.path.insert(0, 'src')" not in src and "from src." not in src, (
            f"{nb.name}: imports local src"
        )


def test_notebooks_print_license_status():
    for nb in _notebooks():
        src = "\n".join(
            "".join(c.get("source", [])) for c in json.loads(nb.read_text())["cells"]
        ).lower()
        assert "license" in src or "commercial" in src, f"{nb.name}: must print license status"


def test_tutorial_execution_ledger_exists():
    led = ROOT / "notebook" / "99_final_report" / "reports" / "v37_tutorial_execution_ledger.csv"
    assert led.exists(), f"missing {led}"

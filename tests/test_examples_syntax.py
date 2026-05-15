# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Syntax-check the beginner examples so they cannot rot silently."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

EXAMPLES = sorted(Path("examples/beginner").glob("*.py"))


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def test_examples_directory_layout():
    assert (Path("examples/beginner") / "01_check_device.py").exists()
    assert (Path("examples/beginner") / "08_start_api.py").exists()
    assert (Path("examples/beginner") / "10_cloudflare_tunnel_safe.md").exists()
    assert (Path("examples/api") / "python_client.py").exists()
    assert (Path("examples/images") / "simple_shapes.jpg").exists()

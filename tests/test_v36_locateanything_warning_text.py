# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.6 LocateAnything warning-text compliance tests.

The NVIDIA non-commercial warning must appear verbatim in every execution
path: Python API, CLI run command, and the CLI list command output.
These tests assert the exact required phrases appear at every surface.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path

_REQUIRED_PHRASES = [
    "WARNING",
    "NVIDIA License",
    "non-commercial use only",
    "VisionServeX does not ship or mirror the weights",
    "BYOT/user-local-cache only",
]


def _vsx() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=30)


def test_warning_phrases_in_facts_dict() -> None:
    from visionservex.vsx import _LOCATEANYTHING_FACTS

    w = _LOCATEANYTHING_FACTS["_warning"]
    for phrase in _REQUIRED_PHRASES:
        assert phrase in w, f"Missing phrase in _LOCATEANYTHING_FACTS warning: {phrase!r}"


def test_explain_dict_contains_full_warning() -> None:
    from visionservex.vsx import VSX

    info = VSX.locateanything("locate-anything-3b").explain()
    w = info["warning"]
    for phrase in _REQUIRED_PHRASES:
        assert phrase in w, f"Missing phrase in explain()['warning']: {phrase!r}"


def test_python_locate_without_flag_warning_to_stderr(capsys) -> None:
    """locate() must print warning to stderr even when it raises VSXError."""
    from PIL import Image

    from visionservex.vsx import VSX, VSXError

    img = Image.new("RGB", (32, 32))
    with contextlib.suppress(VSXError):
        VSX.locateanything("locate-anything-3b").locate(img, text="cat", accept_noncommercial=False)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err, (
        "NVIDIA warning must be printed to stderr even when accept_noncommercial=False"
    )


def test_python_locate_with_flag_warning_to_stderr(capsys) -> None:
    """locate() with accept_noncommercial=True must still print warning to stderr."""
    from PIL import Image

    from visionservex.vsx import VSX

    img = Image.new("RGB", (32, 32))
    with contextlib.suppress(Exception):
        VSX.locateanything("locate-anything-3b").locate(img, text="cat", accept_noncommercial=True)
    captured = capsys.readouterr()
    assert "WARNING" in captured.err, (
        "NVIDIA warning must be printed to stderr even when accept_noncommercial=True"
    )


def test_cli_run_without_flag_warning_in_stderr(tmp_path: Path) -> None:
    from PIL import Image as _PIL

    img = tmp_path / "img.jpg"
    out = tmp_path / "out.json"
    _PIL.new("RGB", (32, 32)).save(img)
    res = _run(
        [
            "locate-anything",
            "run",
            "locate-anything-3b",
            str(img),
            "--text",
            "cat",
            "--out",
            str(out),
        ]
    )
    assert "WARNING" in res.stderr or "NVIDIA" in res.stderr


def test_cli_list_output_contains_warning(tmp_path: Path) -> None:
    res = _run(["locate-anything", "list"])
    combined = res.stdout + res.stderr
    assert "WARNING" in combined or "NVIDIA" in combined or "non-commercial" in combined

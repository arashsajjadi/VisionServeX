# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.25.0: robust CLI help assertion utilities.

The pre-v2.25 CLI tests asserted plain-string membership against
``subprocess.run(["visionservex", ..., "--help"]).stdout``. That stdout is
``rich``-rendered with ANSI escape sequences, soft-wrapping decisions, and
``╮ │ ╰`` box drawing — all of which depend on ``$COLUMNS`` and ``$TERM``.

CI hosts narrower terminals than the dev box, so a flag name like
``--require-gpu`` may get wrapped across two lines and the literal string
``--require-gpu`` no longer appears in stdout. The test fails even though
the flag still exists.

This helper:

1. Sets a deterministic CLI environment for help-text assertions
   (``COLUMNS=160``, ``FORCE_COLOR=0``, ``NO_COLOR=1``, ``TERM=dumb``).
2. Strips ANSI escape sequences from the captured output.
3. Collapses all whitespace (newlines, leading spaces) before substring
   matching so wrap-anywhere rendering can't break the test.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Mapping

__all__ = [
    "DETERMINISTIC_CLI_ENV",
    "assert_help_contains_all",
    "assert_help_contains_any",
    "deterministic_env",
    "run_help",
    "strip_ansi",
]

# Deterministic env for help-text assertions. COLUMNS=160 picks the same
# rich layout the dev box uses.
DETERMINISTIC_CLI_ENV: Mapping[str, str] = {
    "COLUMNS": "160",
    "FORCE_COLOR": "0",
    "NO_COLOR": "1",
    "TERM": "dumb",
    "PYTHONIOENCODING": "utf-8",
}


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX_CHARS = "─│┌┐└┘├┤┬┴┼╭╮╯╰═║╔╗╚╝╠╣╦╩╬"


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from ``text``."""
    return _ANSI_RE.sub("", text)


def _normalize_for_match(text: str) -> str:
    """Strip ANSI, drop box-drawing chars, collapse whitespace."""
    text = strip_ansi(text)
    for ch in _BOX_CHARS:
        text = text.replace(ch, " ")
    return re.sub(r"\s+", " ", text)


def deterministic_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(DETERMINISTIC_CLI_ENV)
    if extra:
        env.update(extra)
    return env


def _vsx_argv() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def run_help(
    args: Iterable[str],
    *,
    timeout: int = 30,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run ``visionservex <args> --help`` with deterministic env."""
    cmd = [*_vsx_argv(), *args, "--help"]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=deterministic_env(extra_env),
    )


def assert_help_contains_all(
    res: subprocess.CompletedProcess,
    needles: Iterable[str],
) -> None:
    """Assert each needle appears in ``res.stdout`` after ANSI/whitespace normalization."""
    assert res.returncode == 0, (res.stdout, res.stderr)
    norm = _normalize_for_match(res.stdout)
    missing = [n for n in needles if n not in norm]
    assert not missing, (
        f"Help output missing expected tokens {missing!r}. "
        f"Normalized stdout (first 800 chars):\n{norm[:800]}"
    )


def assert_help_contains_any(
    res: subprocess.CompletedProcess,
    needles: Iterable[str],
) -> None:
    """Assert at least one needle appears in ``res.stdout`` after normalization."""
    assert res.returncode == 0, (res.stdout, res.stderr)
    norm = _normalize_for_match(res.stdout)
    if not any(n in norm for n in needles):
        raise AssertionError(
            f"Help output contains none of {list(needles)!r}. "
            f"Normalized stdout (first 800 chars):\n{norm[:800]}"
        )

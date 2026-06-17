# SPDX-License-Identifier: Apache-2.0
"""v3.18 legal gate: no Ultralytics / AGPL import on any runtime or training path.

Benchmark-only comparison code (``cli/benchmark_commands.py``) may reference
Ultralytics for an opt-in baseline, but only behind a function-local, gated
import — never at module top level, and never imported by the package runtime.
"""

from __future__ import annotations

from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "visionservex"

# Every package that participates in loading, running, training, exporting, or
# describing a model at runtime. Benchmark/CLI comparison code is excluded by
# design and checked separately below.
_RUNTIME_DIRS = [
    _SRC / "engines",
    _SRC / "core",
    _SRC / "data",
    _SRC / "runtime",
    _SRC / "runtime_broker",
    _SRC / "registry",
    _SRC / "readiness",
    _SRC / "licensing",
    _SRC / "models",
    _SRC / "model_zoo",
    _SRC / "server",
    _SRC / "api",
]

_FORBIDDEN = ("import ultralytics", "from ultralytics")


def test_no_ultralytics_import_on_runtime_or_training_path():
    offenders = []
    for d in _RUNTIME_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*.py"):
            text = f.read_text()
            for needle in _FORBIDDEN:
                if needle in text:
                    offenders.append(f"{f}: {needle}")
    assert not offenders, f"Ultralytics import on a runtime/training path: {offenders}"


def test_benchmark_ultralytics_import_is_gated_not_top_level():
    """The only Ultralytics reference (benchmark comparison) must be a
    function-local optional import, never executed at import time."""
    bench = _SRC / "cli" / "benchmark_commands.py"
    if not bench.exists():
        return
    for line in bench.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(("from ultralytics", "import ultralytics")):
            # must be indented (inside a function), i.e. the raw line starts with
            # whitespace — a top-level import would start at column 0.
            assert line[:1] in (" ", "\t"), f"top-level Ultralytics import: {line!r}"


def test_no_ultralytics_in_declared_dependencies():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text().lower()
    # ultralytics must never be a declared runtime/optional dependency.
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert not s.startswith(("ultralytics", '"ultralytics')), f"declared dep: {line!r}"

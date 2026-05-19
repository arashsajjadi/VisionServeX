# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: libreyolo list-models + build-model-map must enumerate weights."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent

# Skip all LibreYOLO tests when the package is not installed
_LIBREYOLO_AVAILABLE = importlib.util.find_spec("libreyolo") is not None
pytestmark = pytest.mark.skipif(
    not _LIBREYOLO_AVAILABLE,
    reason="libreyolo not installed — install visionservex[libreyolo]",
)


def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


def test_list_models_returns_weights(tmp_path: Path) -> None:
    out = tmp_path / "models.json"
    proc = _run(["libreyolo", "list-models", "--format", "json", "--out", str(out)])
    assert proc.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["n_weights"] > 0


def test_build_model_map_writes_canonical_rows(tmp_path: Path) -> None:
    out = tmp_path / "map.json"
    proc = _run(["libreyolo", "build-model-map", "--out", str(out)])
    assert proc.returncode == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert "rows" in data
    rows = data["rows"]
    assert len(rows) > 0
    # Schema check
    required = {
        "hf_model_id",
        "weight_filename",
        "libreyolo_load_name",
        "visionservex_model_id",
        "family",
        "task",
        "license",
        "license_risk",
        "source_url",
        "default_safe",
        "smoke_command",
        "benchmark_command",
    }
    for row in rows[:3]:
        missing = required - set(row.keys())
        assert not missing, f"row missing keys: {missing}"


def test_build_model_map_has_default_safe() -> None:
    """At least some weights must be default_safe (Apache-2.0)."""
    proc = _run(["libreyolo", "build-model-map", "--out", "/tmp/test_map.json"])
    assert proc.returncode == 0
    data = json.loads(Path("/tmp/test_map.json").read_text())
    n_safe = data.get("n_default_safe", 0)
    assert n_safe > 0, "expected at least one default_safe weight (yolox/dfine/rfdetr/rtdetr)"

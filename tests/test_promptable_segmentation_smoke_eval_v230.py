# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: promptable-segmentation smoke/eval contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


def _run(args: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "visionservex", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO),
    )


def _payload(proc: subprocess.CompletedProcess) -> dict:
    try:
        obj = json.loads(proc.stdout.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                pass
    return {}


def test_promptable_segmentation_has_required_keys(tmp_path: Path) -> None:
    """Smoke eval payload must include task=promptable_segmentation and required fields."""
    ann = REPO / "tests/assets/smoke/coco_instance_sample.json"
    if not ann.exists():
        pytest.skip("smoke annotation missing")
    out = tmp_path / "pseg.json"
    proc = _run(
        [
            "benchmark-promptable-segmentation",
            "--dataset",
            str(ann),
            "--images-dir",
            str(REPO / "tests/assets/smoke"),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "1",
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    p = _payload(proc) or json.loads(out.read_text())
    assert "task" in p
    assert p["task"] == "promptable_segmentation"


def test_promptable_segmentation_metric_status_no_nan(tmp_path: Path) -> None:
    """metric_status must never be the string 'NaN'."""
    ann = REPO / "tests/assets/smoke/coco_instance_sample.json"
    if not ann.exists():
        pytest.skip("smoke annotation missing")
    out = tmp_path / "pseg_metric.json"
    proc = _run(
        [
            "benchmark-promptable-segmentation",
            "--dataset",
            str(ann),
            "--images-dir",
            str(REPO / "tests/assets/smoke"),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "2",
            "--out",
            str(out),
            "--format",
            "json",
        ]
    )
    text = proc.stdout + proc.stderr + (out.read_text() if out.exists() else "")
    assert "NaN" not in text
    assert "NOT_WIRED" not in text


def test_promptable_segmentation_max_instances_alias() -> None:
    """--max-instances must be accepted as the canonical v2.30 flag."""
    proc = _run(
        [
            "benchmark-promptable-segmentation",
            "--dataset",
            "/nonexistent/coco.json",
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "3",
            "--out",
            "/tmp/v230_promptable.json",
            "--format",
            "json",
        ],
        timeout=15,
    )
    # Must be either ok or expected_blocker, never a usage error
    assert proc.returncode != 2, proc.stderr[:200]

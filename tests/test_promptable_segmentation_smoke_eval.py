# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0: promptable segmentation smoke/eval tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SMOKE_ANN = Path("tests/assets/smoke/coco_instance_sample.json")
SMOKE_DIR = Path("tests/assets/smoke")
REPO_ROOT = Path(__file__).parent.parent


def _run(cmd: list[str], timeout: int = 90) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def _parse_payload(proc: subprocess.CompletedProcess) -> dict | None:
    # Try full stdout first (handles pretty-printed multi-line JSON)
    try:
        obj = json.loads(proc.stdout.strip())
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fall back to line-by-line for single-line JSON embedded in output
    for line in proc.stdout.splitlines():
        s = line.strip()
        if s.startswith("{"):
            try:
                return json.loads(s)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Command existence
# ---------------------------------------------------------------------------


def test_promptable_segmentation_command_exists() -> None:
    """benchmark-promptable-segmentation --help must not fail."""
    proc = _run(
        [sys.executable, "-m", "visionservex", "benchmark-promptable-segmentation", "--help"],
        timeout=15,
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr[:200]}"


def test_promptable_segmentation_with_missing_dataset_returns_blocker() -> None:
    """Calling with a non-existent dataset must return expected_blocker SMOKE_ASSET_MISSING."""
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            "/nonexistent/coco.json",
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "1",
            "--out",
            "/tmp/vsx_ps_missing_test.json",
            "--format",
            "json",
        ],
        timeout=15,
    )
    payload = _parse_payload(proc)
    assert payload is not None, (
        f"No structured payload for missing dataset\nstdout={proc.stdout[:200]}"
    )
    assert payload.get("status") == "expected_blocker"
    assert payload.get("code") in ("SMOKE_ASSET_MISSING",)


def test_promptable_segmentation_no_nan_output() -> None:
    """Output must not contain NaN string."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            str(SMOKE_ANN),
            "--images-dir",
            str(SMOKE_DIR),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "1",
            "--out",
            "/tmp/vsx_ps_nan_test.json",
            "--format",
            "json",
        ],
        timeout=90,
    )
    output = proc.stdout + proc.stderr
    assert "NaN" not in output, f"NaN found in output:\n{output[:500]}"
    assert "NOT_WIRED" not in output, f"NOT_WIRED found in output:\n{output[:500]}"


def test_promptable_segmentation_no_not_wired() -> None:
    """Output must not contain NOT_WIRED."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            str(SMOKE_ANN),
            "--images-dir",
            str(SMOKE_DIR),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "1",
            "--out",
            "/tmp/vsx_ps_notwired_test.json",
            "--format",
            "json",
        ],
        timeout=90,
    )
    combined = proc.stdout + proc.stderr
    assert "NOT_WIRED" not in combined


def test_promptable_segmentation_result_structure() -> None:
    """Output JSON must have required top-level keys."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            str(SMOKE_ANN),
            "--images-dir",
            str(SMOKE_DIR),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "2",
            "--out",
            "/tmp/vsx_ps_struct_test.json",
            "--format",
            "json",
        ],
        timeout=90,
    )
    payload = _parse_payload(proc)
    assert payload is not None, f"No JSON payload\nstdout={proc.stdout[:300]}"
    required_keys = {"status", "code", "dataset", "task"}
    missing = required_keys - set(payload.keys())
    assert not missing, f"Payload missing keys: {missing}"
    assert payload["task"] == "promptable_segmentation"


def test_promptable_segmentation_metric_status_not_nan() -> None:
    """metric_status field must never be NaN."""
    if not SMOKE_ANN.exists():
        pytest.skip("smoke annotation missing")
    proc = _run(
        [
            sys.executable,
            "-m",
            "visionservex",
            "benchmark-promptable-segmentation",
            "--dataset",
            str(SMOKE_ANN),
            "--images-dir",
            str(SMOKE_DIR),
            "--models",
            "sam2-hiera-tiny",
            "--max-instances",
            "2",
            "--out",
            "/tmp/vsx_ps_metric_test.json",
            "--format",
            "json",
        ],
        timeout=90,
    )
    payload = _parse_payload(proc)
    if payload is None:
        pytest.skip("no payload to validate")
    for row in payload.get("rows", []):
        for instance in row.get("rows", []):
            ms = instance.get("metric_status", "")
            assert ms not in ("nan", "NaN", ""), (
                f"metric_status is {ms!r} for instance {instance.get('annotation_id')}"
            )
            iou = instance.get("iou")
            if isinstance(iou, float):
                import math

                assert not math.isnan(iou), (
                    f"iou is NaN for instance {instance.get('annotation_id')}"
                )

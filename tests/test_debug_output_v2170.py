# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 4 (v2.17.0): `visionservex debug-output --out --format --draw`.

Tests use mock-detect so they run on CPU without downloading checkpoints.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def _make_image(tmp_path: Path) -> Path:
    img_path = tmp_path / "img.jpg"
    Image.new("RGB", (320, 240), (80, 80, 200)).save(img_path)
    return img_path


def test_debug_output_help_lists_v217_flags() -> None:
    res = subprocess.run(
        [*_vsx_cmd(), "debug-output", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    assert "--out" in res.stdout
    assert "--format" in res.stdout
    assert "--draw" in res.stdout
    assert "--threshold" in res.stdout


def test_debug_output_mock_detect_writes_json(tmp_path: Path) -> None:
    img = _make_image(tmp_path)
    out = tmp_path / "diag.json"
    res = subprocess.run(
        [
            *_vsx_cmd(),
            "debug-output",
            "mock-detect",
            str(img),
            "--threshold",
            "0.001",
            "--format",
            "json",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    payload = json.loads(out.read_text())
    # v2.17.0 schema
    for field in (
        "model_id",
        "image",
        "image_size",
        "device_requested",
        "device_actual",
        "threshold",
        "raw_predictions_count",
        "after_threshold_count",
        "after_nms_count",
        "final_normalized_count",
        "invalid_box_count",
        "dropped_prediction_count",
        "warnings",
        "errors",
    ):
        assert field in payload, field


def test_debug_output_emits_drawing_when_draw_supplied(tmp_path: Path) -> None:
    img = _make_image(tmp_path)
    out = tmp_path / "diag.json"
    draw_path = tmp_path / "diag.jpg"
    res = subprocess.run(
        [
            *_vsx_cmd(),
            "debug-output",
            "mock-detect",
            str(img),
            "--threshold",
            "0.001",
            "--format",
            "json",
            "--out",
            str(out),
            "--draw",
            str(draw_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    assert out.exists()
    # drawing file should be created if engine returned detections
    # (mock-detect always returns at least one detection)
    payload = json.loads(out.read_text())
    if payload.get("raw_predictions_count", 0) > 0:
        assert draw_path.exists()

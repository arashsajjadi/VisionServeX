# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: LibreYOLO smoke-test output must match the normalised schema."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_LIBREYOLO_AVAILABLE = importlib.util.find_spec("libreyolo") is not None
pytestmark = pytest.mark.skipif(not _LIBREYOLO_AVAILABLE, reason="libreyolo not installed")


REPO = Path(__file__).parent.parent
SMOKE_IMG = REPO / "tests/assets/smoke/coco_person_car.jpg"

REQUIRED_PREDICTION_KEYS = {
    "xyxy",
    "score",
    "class_id",
    "category_id",
    "class_name",
    "source_engine",
}


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


def test_libreyolo_smoke_output_schema() -> None:
    """LibreYOLO smoke output predictions must contain the v2.30 fields."""
    if not SMOKE_IMG.exists():
        pytest.skip("smoke image missing")
    proc = _run(
        [
            "libreyolo",
            "smoke-test",
            "libreyolo-yolox-n",
            str(SMOKE_IMG),
            "--device",
            "cpu",
            "--format",
            "json",
        ],
        timeout=120,
    )
    # accept ok or expected_blocker
    p = _payload(proc)
    assert "status" in p
    if p.get("status") != "ok":
        # expected_blocker is acceptable (e.g. weight not downloaded)
        assert p.get("status") == "expected_blocker" or p.get("code") in (
            "LIBREYOLO_REQUIRED",
            "LIBREYOLO_MODEL_NOT_FOUND",
            "DOWNLOAD_FAILED",
            "INPUT_NOT_FOUND",
        ), f"unexpected: {p}"
        return

    # smoke_passed → check schema
    preds = p.get("predictions", [])
    assert "source_engine" in p
    assert p["source_engine"] == "libreyolo"
    if preds:
        first = preds[0]
        missing = REQUIRED_PREDICTION_KEYS - set(first.keys())
        assert not missing, f"prediction missing keys: {missing}; got {first}"
        assert first["source_engine"] == "libreyolo"
        xyxy = first["xyxy"]
        assert isinstance(xyxy, list) and len(xyxy) == 4

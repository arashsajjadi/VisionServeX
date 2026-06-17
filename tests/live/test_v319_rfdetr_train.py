# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.19 LIVE: re-verify the RF-DETR native train lifecycle for one variant.

    VSX_LIVE_RFDTR_TRAIN=1 pytest tests/live/test_v319_rfdetr_train.py -q

Runs the real native trainer for rfdetr-nano via the matrix worker (GPU if
available). The full 8-variant matrix lives in
tools/qa/v319_rfdetr_live_train_matrix.py.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_RFDTR_TRAIN") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_RFDTR_TRAIN=1"),
    pytest.mark.real_model,
]

TOOL = Path("tools/qa/v319_rfdetr_live_train_matrix.py")
MATRIX = Path("docs/qa/v319_operationalize_all_models/rfdetr_live_train_matrix.json")


def test_rfdetr_nano_full_lifecycle():
    device = "cuda" if os.environ.get("VSX_LIVE_DEVICE", "cuda") == "cuda" else "cpu"
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--worker", "rfdetr-nano", "--device", device],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    assert row["status"] == "PASS", row
    assert row["final_state"] == "TRAIN_READY_LIVE"
    assert row["train"] and row["checkpoint_exists"] and row["reload"]
    assert row["predict_after_reload"] and row["output_schema_valid"] and row["export"]


def test_committed_matrix_has_all_8_passes():
    assert MATRIX.exists()
    data = json.loads(MATRIX.read_text())
    assert data["summary"]["PASS"] == 8

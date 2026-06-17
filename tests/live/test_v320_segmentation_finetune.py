# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.20 LIVE: RF-DETR-Seg train lifecycle (env-gated, GPU-preferred).

VSX_LIVE_SEGMENT_FINETUNE=1 pytest tests/live/test_v320_segmentation_finetune.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_SEGMENT_FINETUNE") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_SEGMENT_FINETUNE=1"),
    pytest.mark.real_model,
]

TOOL = Path("tools/qa/v319_rfdetr_live_train_matrix.py")


def test_rfdetr_seg_nano_train_lifecycle():
    device = os.environ.get("VSX_LIVE_DEVICE", "cuda")
    proc = subprocess.run(
        [sys.executable, str(TOOL), "--worker", "rfdetr-seg-nano", "--device", device],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    assert row["status"] == "PASS", row
    assert row["train"] and row["checkpoint_exists"] and row["reload"]
    assert row["predict_after_reload"] and row["output_schema_valid"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.18 LIVE: re-verify the train lifecycle for one fast detector + one classifier.

VSX_LIVE_TRAIN_MATRIX=1 pytest tests/live/test_v318_live_train_lifecycle_matrix.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

LIVE = os.environ.get("VSX_LIVE_TRAIN_MATRIX") == "1"
pytestmark = [
    pytest.mark.skipif(not LIVE, reason="set VSX_LIVE_TRAIN_MATRIX=1"),
    pytest.mark.real_model,
]

MATRIX = Path("docs/qa/v318_full_model_truth/live_train_lifecycle_matrix.json")
TOOL = Path("tools/qa/v318_live_train_lifecycle_matrix.py")


def _run_worker(model_id: str) -> dict:
    import subprocess
    import sys

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out = subprocess.run(
        [sys.executable, str(TOOL), "--worker", model_id],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mid", ["libreyolo-yolox-s", "torchvision-resnet18"])
def test_full_lifecycle_passes(mid):
    row = _run_worker(mid)
    assert row["status"] == "PASS", row
    assert row["final_state"] == "TRAIN_READY_LIVE"
    assert row["train"] and row["checkpoint_path_exists"] and row["reload"]
    assert row["predict_after_reload"] and row["output_schema_valid"]


def test_committed_matrix_matches_baked_evidence():
    from visionservex.readiness import live_evidence

    assert MATRIX.exists(), "run tools/qa/v318_live_train_lifecycle_matrix.py first"
    rows = json.loads(MATRIX.read_text())["results"]
    passed = {r["model_id"] for r in rows if r["status"] == "PASS"}
    assert passed == set(live_evidence.LIVE_TRAIN_VERIFIED)

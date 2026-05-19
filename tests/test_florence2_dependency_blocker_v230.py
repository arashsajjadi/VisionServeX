# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: Florence-2 must not be classified as failed_runtime when the only
issue is incompatible Transformers version."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))


def test_florence2_predict_failed_with_transformers_msg_is_expected_blocker() -> None:
    """A PREDICT_FAILED payload mentioning transformers version → DEPENDENCY_CONFLICT."""
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = json.dumps(
        {
            "error": {
                "code": "PREDICT_FAILED",
                "message": "Florence-2 is incompatible with transformers 5.3.0. "
                "Install: pip install transformers>=4.40,<5.0",
                "hint": "",
            }
        }
    )
    row = SmokeRow(model_id="florence-2-base", family="florence2", task="vlm")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker", f"got {row.final_state!r}"
    assert row.blocker_code in ("DEPENDENCY_CONFLICT", "DEPENDENCY_REQUIRED"), (
        f"got blocker_code={row.blocker_code!r}"
    )


def test_florence2_traceback_with_incompatible_is_expected_blocker() -> None:
    """Plain traceback mentioning 'incompatible with transformers' should still classify."""
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = (
        "Traceback (most recent call last):\n"
        "  File ...\n"
        "ImportError: Florence-2 is incompatible with transformers 5.3.0\n"
    )
    row = SmokeRow(model_id="florence-2-base", family="florence2", task="vlm")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker", f"got {row.final_state!r}"

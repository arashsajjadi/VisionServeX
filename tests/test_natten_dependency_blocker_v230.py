# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.30.0: OneFormer-DiNAT NATTEN dependency handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))


def test_dinat_natten_predict_failed_is_expected_blocker() -> None:
    """A PREDICT_FAILED payload mentioning natten library must be expected_blocker."""
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = json.dumps(
        {
            "error": {
                "code": "PREDICT_FAILED",
                "message": "DinatBackbone requires the natten library but it was not "
                "found in your environment. pip install natten",
                "hint": "",
            }
        }
    )
    row = SmokeRow(model_id="oneformer-dinat-large", family="oneformer", task="segment")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker"
    assert row.blocker_code in ("DEPENDENCY_REQUIRED", "NATTEN_REQUIRED"), (
        f"blocker_code={row.blocker_code!r}"
    )


def test_dinat_traceback_with_natten_is_expected_blocker() -> None:
    from run_model_smoke_matrix import SmokeRow, _classify_row

    stderr = (
        "Traceback (most recent call last):\n  File ...\nImportError: natten library not found\n"
    )
    row = SmokeRow(model_id="oneformer-dinat-large", family="oneformer", task="segment")
    row.returncode = 1
    row = _classify_row(row, "", stderr, Path("/nonexistent.json"))
    assert row.final_state == "expected_blocker"

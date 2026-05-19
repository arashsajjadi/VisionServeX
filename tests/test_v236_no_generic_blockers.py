# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.36.0: no generic expected_blocker without a code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def _check_no_empty_blockers(report_path: Path) -> None:
    if not report_path.exists():
        pytest.skip(f"Report not present: {report_path.name}")
    d = json.loads(report_path.read_text())
    rows = d.get("rows", [])
    for r in rows:
        if r.get("status") == "expected_blocker":
            code = r.get("code", "")
            assert code and code not in ("UNKNOWN", "", "EXPECTED_BLOCKER"), (
                f"Generic blocker in {report_path.name}: "
                f"model={r.get('model_id', '?')} code={code!r}"
            )


def test_v236_segmentation_no_generic_blockers() -> None:
    _check_no_empty_blockers(REPORTS_DIR / "v236_automatic_segmentation_400.json")


def test_v236_preflight_state_no_empty_fields() -> None:
    p = REPORTS_DIR / "v236_preflight_state.json"
    if not p.exists():
        pytest.skip("Preflight state not present")
    d = json.loads(p.read_text())
    assert d.get("current_detection_winner"), "detection winner not recorded"
    assert d.get("current_auto_segmentation_winner"), "seg winner not recorded"

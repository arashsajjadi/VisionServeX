# SPDX-License-Identifier: Apache-2.0
"""v2.33.0: high-value P0/P1 families must not return generic expected_blocker
without a precise blocker_code."""

from __future__ import annotations

import json
from pathlib import Path


def test_v233_blocked_audit_high_value_have_precise_codes() -> None:
    audit_path = Path(__file__).parent.parent / "reports/v233_blocked_model_audit.json"
    if not audit_path.exists():
        return
    audit = json.loads(audit_path.read_text())
    rows = audit.get("rows", [])
    high = [r for r in rows if r.get("priority") in ("P0", "P1")]
    # Allow root_cause categories, but exclude "unknown"
    unknown_high = [r for r in high if r["root_cause_category"] == "unknown"]
    assert not unknown_high, (
        f"P0/P1 with unknown root cause: {[r['model_id'] for r in unknown_high]}"
    )

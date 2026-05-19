# SPDX-License-Identifier: Apache-2.0
"""v2.38.0: no generic expected_blocker in v238 matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_v238_matrix_no_generic_blockers() -> None:
    p = Path(__file__).parent.parent / "reports/v238_49_blocked_resolution_matrix.json"
    if not p.exists():
        pytest.skip("v2.38 matrix not present")
    d = json.loads(p.read_text())
    bad = []
    for r in d["rows"]:
        final = r.get("final_state_after_v238") or r.get("final_state_after_v237") or ""
        if final in ("", "expected_blocker", "stub", "unknown"):
            bad.append((r["model_id"], final))
    assert not bad, f"Generic blockers: {bad}"


def test_v238_no_apache_marked_license_blocked() -> None:
    p = Path(__file__).parent.parent / "reports/v238_49_blocked_resolution_matrix.json"
    if not p.exists():
        pytest.skip("v2.38 matrix not present")
    d = json.loads(p.read_text())
    violations = []
    for r in d["rows"]:
        lic = (r["current_license_status"] or "").upper()
        final = (r.get("final_state_after_v238") or r.get("final_state_after_v237") or "").lower()
        if ("APACHE" in lic or lic == "MIT") and "license_blocked" in final:
            violations.append((r["model_id"], lic, final))
    assert not violations, f"Permissive licenses blocked: {violations}"

# SPDX-License-Identifier: Apache-2.0
"""v2.37.0: permissive licenses must never be license_blocked."""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "reports/v237_49_blocked_resolution_matrix.json"


def test_no_apache_or_mit_marked_license_blocked() -> None:
    d = json.loads(REPORT.read_text())
    violations = []
    for r in d["rows"]:
        lic = (r["current_license_status"] or "").upper()
        new_final = r["final_state_after_v237"]
        if ("APACHE" in lic or lic == "MIT") and "license_blocked" in new_final.lower():
            violations.append((r["model_id"], lic, new_final))
    assert not violations, f"Permissive licenses incorrectly blocked: {violations}"

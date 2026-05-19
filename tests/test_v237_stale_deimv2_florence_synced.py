# SPDX-License-Identifier: Apache-2.0
"""v2.37.0: stale DEIMv2 and Florence-2 rows must reflect v2.35/v2.36 evidence."""
from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "reports/v237_49_blocked_resolution_matrix.json"


def test_deimv2_s_m_l_x_benchmark_passed() -> None:
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    for size in ["s", "m", "l", "x"]:
        mid = f"deimv2-{size}"
        assert rows[mid]["final_state_after_v237"] == "benchmark_passed", (
            f"{mid}: stale state {rows[mid]['final_state_after_v237']}"
        )


def test_deimv2_smaller_variants_attempted() -> None:
    """DEIMv2 atto/femto/pico/n must have non-generic final state."""
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    for size in ["atto", "femto", "pico", "n"]:
        mid = f"deimv2-{size}"
        final = rows[mid]["final_state_after_v237"]
        assert final not in ("", "expected_blocker", "stub", "unknown"), (
            f"{mid}: {final!r}"
        )


def test_florence_sidecar_marked() -> None:
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    for mid in ["florence-2-base", "florence-2-large"]:
        assert rows[mid]["final_state_after_v237"] == "demo_passed_sidecar"

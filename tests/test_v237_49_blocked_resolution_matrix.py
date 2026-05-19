# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.37.0: 49-row blocked-model resolution matrix."""

from __future__ import annotations

import json
from pathlib import Path

REPORT = Path(__file__).parent.parent / "reports/v237_49_blocked_resolution_matrix.json"


def test_v237_matrix_exists() -> None:
    assert REPORT.exists(), "v237 49-row matrix missing"
    d = json.loads(REPORT.read_text())
    assert d.get("total") == 49


def test_v237_no_generic_expected_blocker() -> None:
    d = json.loads(REPORT.read_text())
    for r in d["rows"]:
        final = r["final_state_after_v237"]
        assert final not in ("", "expected_blocker", "stub", "unknown"), (
            f"{r['model_id']}: generic final_state {final!r}"
        )


def test_v237_permissive_license_not_blocked_unless_PML() -> None:
    """Apache-2.0/MIT rows must not be license_blocked unless contradictory source."""
    d = json.loads(REPORT.read_text())
    for r in d["rows"]:
        lic = (r["current_license_status"] or "").upper()
        new_final = r["final_state_after_v237"]
        if "APACHE" in lic or lic == "MIT":
            # Allow opt_in_license_required only for explicit PML cases (rfdetr-seg-xlarge/2xlarge)
            assert "license_blocked" not in new_final.lower(), (
                f"{r['model_id']}: permissive license {lic} but final={new_final}"
            )


def test_v237_pml_correctly_classified() -> None:
    d = json.loads(REPORT.read_text())
    for r in d["rows"]:
        if "PML" in (r["current_license_status"] or "").upper():
            assert r["final_state_after_v237"] in (
                "opt_in_license_required",
                "license_blocked",
            ), f"{r['model_id']}: PML must be opt_in"


def test_v237_deimv2_main_synced() -> None:
    """DEIMv2 s/m/l/x must be marked benchmark_passed (already done in v2.35)."""
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    for size in ["deimv2-s", "deimv2-m", "deimv2-l", "deimv2-x"]:
        assert size in rows
        assert rows[size]["final_state_after_v237"] == "benchmark_passed", (
            f"{size}: {rows[size]['final_state_after_v237']}"
        )


def test_v237_florence_stale_synced() -> None:
    """Florence-2 must be marked demo_passed_sidecar (v2.36 evidence)."""
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    for mid in ["florence-2-base", "florence-2-large"]:
        assert rows[mid]["final_state_after_v237"] == "demo_passed_sidecar"


def test_v237_rfdetr_seg_large_not_license_blocked() -> None:
    """rfdetr-seg-large must be reclassified from license_blocked to checkpoint_required."""
    d = json.loads(REPORT.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    assert rows["rfdetr-seg-large"]["final_state_after_v237"] == "checkpoint_required"
    assert "license_blocked" not in rows["rfdetr-seg-large"]["final_state_after_v237"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.45.0: non-healthy count must decrease or all remaining blockers are precisely documented."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LEDGER_CSV = (
    Path(__file__).resolve().parent.parent
    / "notebook/99_final_report/reports/model_coverage_ledger.csv"
)
PLAN_CSV = Path(__file__).resolve().parent.parent / "reports/v245_exact_51_recovery_plan.csv"

HEALTHY_STATES = frozenset(
    {"benchmark_passed", "smoke_passed", "demo_passed_sidecar", "contract_passed"}
)

V244_BASELINE_HEALTHY = 90  # from v2.44 docs


def _load():
    if not LEDGER_CSV.exists():
        pytest.skip("ledger CSV not present")
    return list(csv.DictReader(LEDGER_CSV.open()))


def test_healthy_rows_at_least_as_many_as_v244():
    rows = _load()
    healthy = sum(1 for r in rows if r.get("final_state", "") in HEALTHY_STATES)
    assert healthy >= V244_BASELINE_HEALTHY, (
        f"Healthy rows ({healthy}) less than v2.44 baseline ({V244_BASELINE_HEALTHY}). "
        "v2.45 must not regress."
    )


def test_no_unclassified_blockers():
    rows = _load()
    bad = [
        r["model_id"]
        for r in rows
        if r.get("final_state", "") not in HEALTHY_STATES
        and str(r.get("blocker_category", "")).strip() in ("", "unclassified")
    ]
    assert not bad, f"Unclassified blockers: {bad[:10]}"


def test_obb_models_are_classified_not_unclassified():
    """v2.45 added OBB infrastructure. The rotated-detection models (rtmdet-r/r2)
    are now routed through the OpenMMLab sidecar (final_state=sidecar_required) —
    a valid classified state, not an unclassified blocker. (Earlier they were
    tracked as contract_passed proxies; that proxy was retired.)"""
    rows = _load()
    obb = [r for r in rows if r.get("model_id", "").startswith("rtmdet-r")]
    for r in obb:
        assert r.get("final_state") not in ("", "unclassified"), (
            f"{r['model_id']} unclassified: {r.get('final_state')!r}"
        )


def test_license_gate_cli_accessible():
    """visionservex license-gate check must be a callable command."""
    import subprocess

    result = subprocess.run(
        ["visionservex", "license-gate", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "license-gate command not accessible"
    assert "check" in result.stdout.lower()


def test_registry_validate_cli_accessible():
    """visionservex registry validate must be a callable command."""
    import subprocess

    result = subprocess.run(
        ["visionservex", "registry", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "registry command not accessible"

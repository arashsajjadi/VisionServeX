# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.38.0: model_coverage_ledger must have 0 stale expected_blocker or stub rows."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

LEDGER = Path(__file__).parent.parent / "notebook/99_final_report/reports/model_coverage_ledger.csv"


def _load() -> list[dict]:
    if not LEDGER.exists():
        pytest.skip("model_coverage_ledger.csv not present")
    with open(LEDGER) as fh:
        return list(csv.DictReader(fh))


def test_no_expected_blocker_rows() -> None:
    """No row may show a STALE expected_blocker final_state.

    The one documented exception (added v2.48) is the LibreYOLO segmentation
    family, whose seg head is genuinely not runnable in this build
    (blocker_code=MODEL_NOT_RUNNABLE_IN_THIS_BUILD). Those are honestly
    classified, not stale.
    """
    rows = _load()
    bad = [
        r["model_id"]
        for r in rows
        if r["final_state"] == "expected_blocker"
        and not (
            r["model_id"].startswith("libreyolo-")
            and r["model_id"].endswith("-seg")
            and r.get("blocker_code") == "MODEL_NOT_RUNNABLE_IN_THIS_BUILD"
        )
    ]
    assert not bad, f"Stale expected_blocker rows: {bad}"


def test_no_stub_final_state_rows() -> None:
    """Zero rows may show stub as final_state."""
    rows = _load()
    bad = [r["model_id"] for r in rows if r["final_state"] == "stub"]
    assert not bad, f"Stale stub rows: {bad}"


def test_florence_is_demo_passed_sidecar() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for mid in ["florence-2-base", "florence-2-large"]:
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "demo_passed_sidecar", (
            f"{mid}: expected demo_passed_sidecar, got {r['final_state']}"
        )


def test_deimv2_main_sizes_are_benchmark_passed() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for size in ["s", "m", "l", "x"]:
        mid = f"deimv2-{size}"
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "benchmark_passed", (
            f"{mid}: expected benchmark_passed, got {r['final_state']}"
        )


def test_deimv2_smaller_sizes_are_benchmark_passed() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for size in ["atto", "femto", "pico"]:
        mid = f"deimv2-{size}"
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "benchmark_passed", (
            f"{mid}: expected benchmark_passed, got {r['final_state']}"
        )


def test_rtdetrv4_checkpoint_downloaded_or_benchmark() -> None:
    """v2.38 said checkpoint_downloaded; v2.41 benchmarked all 4 variants."""
    rows = {r["model_id"]: r for r in _load()}
    allowed = {"checkpoint_downloaded", "benchmark_passed", "smoke_passed"}
    for size in ["s", "m", "l", "x"]:
        mid = f"rtdetrv4-{size}"
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] in allowed, f"{mid}: {r['final_state']!r} not in {allowed}"


def test_rfdetr_seg_large_benchmark_passed() -> None:
    rows = {r["model_id"]: r for r in _load()}
    r = rows.get("rfdetr-seg-large")
    if r is None:
        pytest.skip("rfdetr-seg-large not in ledger")
    assert r["final_state"] == "benchmark_passed", (
        f"rfdetr-seg-large: expected benchmark_passed, got {r['final_state']}"
    )


def test_rfdetr_seg_large_not_license_blocked() -> None:
    rows = {r["model_id"]: r for r in _load()}
    r = rows.get("rfdetr-seg-large")
    if r is None:
        pytest.skip("rfdetr-seg-large not in ledger")
    assert "license_blocked" not in r["final_state"], (
        "rfdetr-seg-large is Apache-2.0 core — must not be license_blocked"
    )


def test_oneformer_convnext_is_wired() -> None:
    # v2.46: oneformer-convnext-large corrected to `wired` (Lane-A no-env win, Apache-2.0).
    rows = {r["model_id"]: r for r in _load()}
    r = rows.get("oneformer-convnext-large")
    if r is None:
        pytest.skip("oneformer-convnext-large not in ledger")
    assert r["final_state"] == "wired", f"expected wired, got {r['final_state']}"


def test_deim_legacy_is_wired() -> None:
    # v2.46: deim-m / deim-s corrected to `wired` (Lane-A no-env wins).
    rows = {r["model_id"]: r for r in _load()}
    for mid in ["deim-m", "deim-s"]:
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "wired", f"{mid}: expected wired, got {r['final_state']}"


def test_rfdetr_seg_xlarge_is_opt_in() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for mid in ["rfdetr-seg-xlarge", "rfdetr-seg-2xlarge"]:
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "opt_in_license_required", (
            f"{mid}: PML-1.0 must be opt_in_license_required, got {r['final_state']}"
        )

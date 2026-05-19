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
    """Zero rows may show expected_blocker as final_state."""
    rows = _load()
    bad = [r["model_id"] for r in rows if r["final_state"] == "expected_blocker"]
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


def test_oneformer_convnext_is_wrong_registry_entry() -> None:
    rows = {r["model_id"]: r for r in _load()}
    r = rows.get("oneformer-convnext-large")
    if r is None:
        pytest.skip("oneformer-convnext-large not in ledger")
    assert r["final_state"] == "wrong_registry_entry", (
        f"expected wrong_registry_entry, got {r['final_state']}"
    )


def test_deim_legacy_is_upstream_deprecated() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for mid in ["deim-m", "deim-s"]:
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "upstream_deprecated", (
            f"{mid}: expected upstream_deprecated, got {r['final_state']}"
        )


def test_rfdetr_seg_xlarge_is_opt_in() -> None:
    rows = {r["model_id"]: r for r in _load()}
    for mid in ["rfdetr-seg-xlarge", "rfdetr-seg-2xlarge"]:
        r = rows.get(mid)
        if r is None:
            pytest.skip(f"{mid} not in ledger")
        assert r["final_state"] == "opt_in_license_required", (
            f"{mid}: PML-1.0 must be opt_in_license_required, got {r['final_state']}"
        )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: post-v2.59 inventory completeness + integrity."""
from __future__ import annotations

import csv
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
INV = R / "v37_post_v259_inventory.csv"

REQUIRED_COLS = [
    "item_id", "family", "task", "introduced_version", "current_state", "default_safe",
    "commercial_safe", "license_status", "source", "evidence_artifact", "artifact_exists",
    "has_registry", "has_vsx_api", "has_cli", "has_explain", "has_test", "has_tutorial",
    "tutorial_executed", "has_docs", "fresh_install_verified", "product_grade_status",
    "missing_work", "exact_next_command",
]
VALID_PRODUCT_STATES = {
    "product_grade_pass", "runtime_pass_but_not_product_grade", "demo_only", "auth_required",
    "checkpoint_required", "sidecar_required", "legal_review_required", "excluded_restricted",
    "not_released", "failed_runtime", "external_api_only", "product_grade_candidate",
    "tool_available", "blocked_documented",
}


def _rows():
    return list(csv.DictReader(INV.open()))


def test_inventory_exists():
    assert INV.exists(), f"missing {INV}"


def test_all_required_columns_present():
    rows = _rows()
    assert rows
    for col in REQUIRED_COLS:
        assert col in rows[0], f"missing column {col}"


def test_at_least_40_items():
    assert len(_rows()) >= 40


def test_no_empty_state():
    for r in _rows():
        assert r["current_state"].strip(), f"empty state for {r['item_id']}"
        assert r["current_state"] not in ("unknown", "absent")


def test_product_grade_status_valid():
    for r in _rows():
        assert r["product_grade_status"] in VALID_PRODUCT_STATES, \
            f"{r['item_id']}: invalid product_grade_status {r['product_grade_status']!r}"


def test_benchmark_passed_rows_have_artifact():
    for r in _rows():
        if r["current_state"] == "benchmark_passed" and r["evidence_artifact"]:
            assert r["artifact_exists"] == "True", \
                f"{r['item_id']}: benchmark_passed but artifact missing: {r['evidence_artifact']}"


def test_locateanything_not_commercial_safe():
    for r in _rows():
        if "locateanything" in r["item_id"] or "locate-anything" in r["item_id"]:
            assert r["commercial_safe"] == "False"
            assert r["current_state"] == "excluded_restricted"


def test_every_item_has_next_command():
    for r in _rows():
        assert r["exact_next_command"].strip(), f"{r['item_id']} missing next command"

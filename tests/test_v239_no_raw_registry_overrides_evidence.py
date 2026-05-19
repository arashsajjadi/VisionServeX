# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: the reconciler must NEVER let raw registry state (stub /
expected_blocker / blocked) override real execution evidence."""

from __future__ import annotations

import json

from visionservex.reporting.v239_reconciler import reconcile


def test_real_evidence_beats_raw_registry_stub(tmp_path):
    # Fake a task report with deimv2-s = benchmark_passed
    reports = tmp_path / "notebook" / "01_object_detection" / "reports"
    reports.mkdir(parents=True)
    (reports / "detection_leaderboard.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "model_id": "deimv2-s",
                        "status": "ok",
                        "mAP50_95": 0.3266,
                        "AP50": 0.5,
                    }
                ]
            }
        )
    )
    payload = reconcile(
        task_reports_root=tmp_path / "notebook",
        resolution_matrix_path=None,
        notebook_call_ledger_path=None,
    )
    by_id = {r["model_id"]: r for r in payload["rows"]}
    assert by_id["deimv2-s"]["final_state"] == "benchmark_passed"
    # raw registry would have said stub/audit_only/non_core — none should override
    assert by_id["deimv2-s"]["registry_status"] not in ("benchmark_passed", "wired")


def test_known_correction_used_when_no_evidence(tmp_path):
    """If no live evidence, KNOWN_CORRECTIONS still wins over raw registry."""
    payload = reconcile(
        task_reports_root=tmp_path / "notebook",
        resolution_matrix_path=None,
        notebook_call_ledger_path=None,
    )
    by_id = {r["model_id"]: r for r in payload["rows"]}
    assert by_id["rfdetr-seg-large"]["final_state"] == "benchmark_passed"
    assert by_id["oneformer-convnext-large"]["final_state"] == "wrong_registry_entry"
    assert by_id["deim-m"]["final_state"] == "upstream_deprecated"


def test_evidence_higher_than_correction_when_actually_better(tmp_path):
    """If a model has both a correction and live benchmark evidence at the
    same level, the live evidence path is still respected (corrections do
    not downgrade)."""
    reports = tmp_path / "notebook" / "01_object_detection" / "reports"
    reports.mkdir(parents=True)
    # Florence-2 sees a benchmark — same priority as correction's
    # demo_passed_sidecar (lower in priority), so demo_passed_sidecar wins
    # if benchmark is missing, benchmark_passed wins if it exists.
    (reports / "vlm_benchmark.json").write_text(
        json.dumps({"rows": [{"model_id": "florence-2-base", "status": "ok", "mAP50_95": 0.42}]})
    )
    payload = reconcile(
        task_reports_root=tmp_path / "notebook",
        resolution_matrix_path=None,
        notebook_call_ledger_path=None,
    )
    by_id = {r["model_id"]: r for r in payload["rows"]}
    # benchmark_passed > demo_passed_sidecar
    assert by_id["florence-2-base"]["final_state"] == "benchmark_passed"

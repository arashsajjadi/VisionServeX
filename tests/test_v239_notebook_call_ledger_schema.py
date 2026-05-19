# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: notebook call ledger schema and helpers."""

from __future__ import annotations

import pytest

from visionservex.reporting.notebook_calls import (
    ALLOWED_CALL_TYPES,
    ALLOWED_EXECUTION_STATUS,
    ALLOWED_SKIP_REASONS,
    NotebookCall,
    NotebookCallLedger,
    record_model_call,
    record_skip,
)


def test_allowed_call_types_complete() -> None:
    required = {
        "benchmark",
        "contract",
        "smoke",
        "demo",
        "doctor",
        "status",
        "checkpoint_state",
        "auth_gate",
        "sidecar_status",
        "skipped_with_reason",
    }
    assert required.issubset(ALLOWED_CALL_TYPES)


def test_allowed_execution_status_complete() -> None:
    required = {
        "executed",
        "executed_blocked",
        "skipped_external_auth",
        "skipped_manual_checkpoint",
        "skipped_license_opt_in",
        "skipped_deprecated",
        "skipped_wrong_registry_entry",
    }
    assert required.issubset(ALLOWED_EXECUTION_STATUS)


def test_allowed_skip_reasons_complete() -> None:
    required = {
        "auth_required",
        "opt_in_license_required",
        "upstream_deprecated",
        "wrong_registry_entry",
        "manual_checkpoint_required",
        "sidecar_required",
    }
    assert required.issubset(ALLOWED_SKIP_REASONS)


def test_ledger_roundtrip(tmp_path):
    led = NotebookCallLedger.init(tmp_path / "ledger.json", run_id="test-run-1")
    record_model_call(
        model_id="dfine-x",
        notebook="01_object_detection/Object_Detection_Benchmark.ipynb",
        section="detection",
        task="detect",
        command="visionservex benchmark-detection ...",
        call_type="benchmark",
        status="executed",
        final_state="benchmark_passed",
        evidence_artifact="reports/v239_dfine_x.json",
        ledger=led,
    )
    record_skip(
        model_id="sam3-base",
        notebook="04_open_vocab_vlm/Open_Vocab_VLM_Demo.ipynb",
        section="vlm",
        task="vlm",
        reason="auth_required",
        ledger=led,
    )
    led2 = NotebookCallLedger.load(led.path)
    assert led2.run_id == "test-run-1"
    assert len(led2.calls) == 2
    summary = led2.coverage_summary()
    assert summary["total_models"] == 2
    assert "dfine-x" in summary["called_model_ids"]
    assert "sam3-base" in summary["skipped_model_ids"]


def test_invalid_call_type_rejected(tmp_path):
    led = NotebookCallLedger.init(tmp_path / "ledger.json")
    with pytest.raises(ValueError):
        record_model_call(
            model_id="x",
            notebook="x.ipynb",
            section="x",
            task="detect",
            command="",
            call_type="not_a_real_type",
            status="executed",
            ledger=led,
        )


def test_invalid_skip_reason_rejected(tmp_path):
    led = NotebookCallLedger.init(tmp_path / "ledger.json")
    with pytest.raises(ValueError):
        record_skip(
            model_id="x",
            notebook="x.ipynb",
            section="x",
            task="detect",
            reason="i_just_dont_want_to",
            ledger=led,
        )


def test_csv_export(tmp_path):
    led = NotebookCallLedger.init(tmp_path / "ledger.json")
    record_model_call(
        model_id="dinov2-base",
        notebook="06_embedding_similarity/Embedding_Similarity_Demo.ipynb",
        section="embed",
        task="embed",
        command="visionservex demo dinov2-base",
        call_type="demo",
        status="executed",
        final_state="demo_passed",
        ledger=led,
    )
    csv_path = tmp_path / "ledger.csv"
    led.write_csv(csv_path)
    text = csv_path.read_text()
    assert "model_id" in text.splitlines()[0]
    assert "dinov2-base" in text


def test_notebook_call_dataclass_to_dict():
    c = NotebookCall(
        model_id="x",
        family="fam",
        task="detect",
        notebook_path="p",
        notebook_section="s",
        call_type="smoke",
        command_or_api="cmd",
        called_in_notebook=True,
        call_count=1,
        execution_status="executed",
    )
    d = c.to_dict()
    assert d["model_id"] == "x"
    assert d["called_in_notebook"] is True

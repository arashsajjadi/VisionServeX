# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.42.0: provenance sidecar and verify-generated-artifacts tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from visionservex.reporting.v242_provenance import (
    OLD_11_COLUMN_SCHEMA,
    REQUIRED_LEDGER_COLUMNS,
    verify_artifact,
    verify_generated_artifacts,
    write_provenance,
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_write_provenance_creates_sidecar(tmp_path):
    artifact = tmp_path / "ledger.csv"
    artifact.write_text("model_id,final_state\ndfine-x,benchmark_passed\n")
    prov_path = write_provenance(artifact, run_id="testrun001")
    assert prov_path.exists()
    prov = json.loads(prov_path.read_text())
    assert prov["run_id"] == "testrun001"
    assert prov["manual_edit_detected"] is False
    assert prov["artifact_sha256"] == _sha256(artifact)
    assert prov["generated_by"].startswith("visionservex")


def test_verify_artifact_detects_manual_edit(tmp_path):
    artifact = tmp_path / "ledger.csv"
    artifact.write_text("model_id,final_state\ndfine-x,benchmark_passed\n")
    write_provenance(artifact, run_id="run001")
    # Simulate manual edit
    artifact.write_text("model_id,final_state\ndfine-x,MANUALLY_PATCHED\n")
    result = verify_artifact(artifact, run_id="run001")
    assert result["manual_edit_detected"] is True
    assert not result["ok"]
    assert any("HASH_MISMATCH" in i for i in result["issues"])


def test_verify_artifact_passes_when_untouched(tmp_path):
    import csv as _csv

    artifact = tmp_path / "ledger.csv"
    # Write a proper CSV with required columns
    cols = [*sorted(REQUIRED_LEDGER_COLUMNS), "engine", "run_mode"]
    with artifact.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for mid in ["dfine-x", "rtdetrv4-x"]:
            row = dict.fromkeys(cols, "")
            row["model_id"] = mid
            row["final_state"] = "benchmark_passed"
            row["current_run_id"] = "run001"
            w.writerow(row)
    write_provenance(artifact, run_id="run001")
    result = verify_artifact(artifact, run_id="run001", min_rows=2)
    assert result["hash_match"] is True
    assert result["required_columns_present"] is True
    assert result["old_schema_detected"] is False


def test_verify_artifact_rejects_old_11_column_schema(tmp_path):
    import csv as _csv

    artifact = tmp_path / "stale_ledger.csv"
    with artifact.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=sorted(OLD_11_COLUMN_SCHEMA))
        w.writeheader()
        for mid in ["dfine-x"]:
            row = dict.fromkeys(OLD_11_COLUMN_SCHEMA, "")
            row["model_id"] = mid
            w.writerow(row)
    write_provenance(artifact, run_id="run001")
    result = verify_artifact(artifact, run_id="run001")
    assert result["old_schema_detected"] is True
    assert not result["ok"]
    assert any("OLD_11_COLUMN_SCHEMA" in i for i in result["issues"])


def test_required_ledger_columns_superset_of_old():
    # The required set must be STRICTLY LARGER than the old set
    assert not OLD_11_COLUMN_SCHEMA.issubset(REQUIRED_LEDGER_COLUMNS) or len(
        REQUIRED_LEDGER_COLUMNS
    ) > len(OLD_11_COLUMN_SCHEMA)
    assert "current_run_id" in REQUIRED_LEDGER_COLUMNS
    assert "called_in_current_notebook_run" in REQUIRED_LEDGER_COLUMNS
    assert "evidence_source_kind" in REQUIRED_LEDGER_COLUMNS
    assert "implementation_status" not in REQUIRED_LEDGER_COLUMNS  # old column


def test_verify_generated_artifacts_missing_dir(tmp_path):
    payload = verify_generated_artifacts(tmp_path / "does_not_exist", run_id="x")
    assert payload["status"] == "failed"


def test_verify_generated_artifacts_all_pass(tmp_path):
    import csv as _csv

    # Create all 4 required artifacts
    for name in [
        "model_coverage_ledger.csv",
        "model_coverage_ledger.json",
        "final_winners.json",
        "notebook_model_call_ledger.json",
    ]:
        p = tmp_path / name
        if name.endswith(".csv"):
            cols = [*sorted(REQUIRED_LEDGER_COLUMNS), "engine"]
            with p.open("w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for i in range(141):
                    row = dict.fromkeys(cols, "")
                    row["model_id"] = f"model_{i}"
                    row["current_run_id"] = "run001"
                    w.writerow(row)
        elif name == "model_coverage_ledger.json":
            base_row = dict.fromkeys(REQUIRED_LEDGER_COLUMNS, "")
            base_row["current_run_id"] = "run001"
            rows_j = [{**base_row, "model_id": f"m{i}"} for i in range(141)]
            p.write_text(json.dumps({"total": 141, "run_id": "run001", "rows": rows_j}))
        else:
            p.write_text(json.dumps({"run_id": "run001"}))
        write_provenance(p, run_id="run001")
    payload = verify_generated_artifacts(tmp_path, run_id="run001", min_rows=141)
    assert payload["artifacts_ok"] == 4
    assert payload["status"] == "ok"

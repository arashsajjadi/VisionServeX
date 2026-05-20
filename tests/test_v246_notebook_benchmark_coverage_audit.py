# SPDX-License-Identifier: Apache-2.0
"""Tests for `visionservex notebook audit-benchmark-coverage`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from visionservex.cli.main import app

RUNNER = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "notebook" / "99_final_report" / "reports" / "model_coverage_ledger.csv"


def test_audit_runs_against_repo_ledger_and_emits_schema() -> None:
    if not LEDGER.exists():
        # Allow the test to skip in environments without the notebook tree.
        import pytest

        pytest.skip("ledger not present in this checkout")
    result = RUNNER.invoke(
        app,
        [
            "notebook",
            "audit-benchmark-coverage",
            "--ledger",
            str(LEDGER),
            "--notebook-root",
            str(REPO_ROOT / "notebook"),
        ],
    )
    assert result.exit_code == 0, result.output
    # The output is a single pretty-printed JSON document. Find the first '{'
    # and parse from there to be robust against any incidental rich-style
    # leader the runner may emit.
    text = result.output
    start = text.find("{")
    assert start != -1, result.output
    payload = json.loads(text[start:])
    assert payload["schema_version"].startswith("v246.notebook_benchmark_coverage_audit")
    assert "missing_counts" in payload
    assert isinstance(payload["missing_counts"], dict)


def test_audit_fails_when_ledger_missing(tmp_path: Path) -> None:
    fake_ledger = tmp_path / "no_such_ledger.csv"
    result = RUNNER.invoke(
        app,
        [
            "notebook",
            "audit-benchmark-coverage",
            "--ledger",
            str(fake_ledger),
            "--notebook-root",
            str(tmp_path),
            "--fail-on-missing",
        ],
    )
    # exit code 2 (ledger missing) when --fail-on-missing is set
    assert result.exit_code == 2


def test_audit_does_not_fail_silently_when_notebook_root_missing(tmp_path: Path) -> None:
    fake_root = tmp_path / "no_such_notebook_dir"
    if not LEDGER.exists():
        import pytest

        pytest.skip("ledger not present in this checkout")
    result = RUNNER.invoke(
        app,
        [
            "notebook",
            "audit-benchmark-coverage",
            "--ledger",
            str(LEDGER),
            "--notebook-root",
            str(fake_root),
        ],
    )
    assert "notebook_root_missing" in result.output

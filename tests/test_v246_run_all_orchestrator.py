# SPDX-License-Identifier: Apache-2.0
"""RUN_ALL.ipynb must be a real orchestrator, not a 3-cell static wrapper.

It must read no `archive_legacy/` ledger and must call the reconciler +
benchmark-coverage audit + Final_Report. The notebook must hard-validate
the regenerated ledger schema (old 11-column schema raises).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ALL_NB = REPO_ROOT / "notebook" / "RUN_ALL.ipynb"
FINAL_REPORT_NB = REPO_ROOT / "notebook" / "99_final_report" / "Final_Report.ipynb"


def _nb_source(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    parts: list[str] = []
    for cell in data.get("cells") or []:
        src = cell.get("source")
        if isinstance(src, list):
            parts.append("".join(src))
        elif isinstance(src, str):
            parts.append(src)
    return "\n".join(parts)


def test_run_all_has_more_than_3_cells() -> None:
    assert RUN_ALL_NB.exists()
    data = json.loads(RUN_ALL_NB.read_text())
    assert len(data["cells"]) >= 8, "RUN_ALL.ipynb should be a real orchestrator (>= 8 cells)"


def test_run_all_calls_reconciler() -> None:
    text = _nb_source(RUN_ALL_NB)
    assert "reconcile-model-states" in text


def test_run_all_calls_benchmark_coverage_audit() -> None:
    text = _nb_source(RUN_ALL_NB)
    assert "audit-benchmark-coverage" in text


def test_run_all_calls_final_report() -> None:
    text = _nb_source(RUN_ALL_NB)
    assert "Final_Report.ipynb" in text


def test_run_all_validates_old_schema_rejection() -> None:
    text = _nb_source(RUN_ALL_NB)
    assert "OLD_SCHEMA_COLS" in text or "implementation_status" in text
    assert "raise" in text  # the hard-validation cell must raise on the old schema


def test_run_all_requires_v246_columns() -> None:
    text = _nb_source(RUN_ALL_NB)
    for col in (
        "runtime_id",
        "current_run_id",
        "called_in_current_notebook_run",
        "current_run_artifact_exists",
        "command_attempted",
        "next_iteration_command",
    ):
        assert col in text, f"RUN_ALL.ipynb missing required v2.46 column: {col}"


def test_run_all_mints_fresh_run_id() -> None:
    text = _nb_source(RUN_ALL_NB)
    assert "VISIONSERVEX_NOTEBOOK_RUN_ID" in text


def test_final_report_does_not_read_archive_legacy() -> None:
    text = _nb_source(FINAL_REPORT_NB)
    # The notebook may MENTION archive_legacy in a "do not read" comment but
    # must not USE it as a ledger source.
    bad_pattern = "pd.read_csv(NB_ROOT / \"archive_legacy"
    bad_pattern_alt = "pd.read_csv(NB_ROOT / 'archive_legacy"
    bad_pattern_assign = "ledger_path = NB_ROOT / \"archive_legacy"
    assert bad_pattern not in text
    assert bad_pattern_alt not in text
    assert bad_pattern_assign not in text


def test_final_report_validates_old_schema() -> None:
    text = _nb_source(FINAL_REPORT_NB)
    assert "OLD_SCHEMA_COLS" in text or "implementation_status" in text
    assert "raise" in text or "assert" in text

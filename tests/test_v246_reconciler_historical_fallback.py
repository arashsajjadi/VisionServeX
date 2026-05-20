# SPDX-License-Identifier: Apache-2.0
"""v2.46 reconciler historical_fallback_ledger.

When a previous run's ledger is supplied as fallback, models without current
evidence should carry forward their previous final_state with
``metric_origin=historical_validated`` set transparently.
"""

from __future__ import annotations

import csv
from pathlib import Path

from visionservex.reporting.v239_reconciler import (
    _load_historical_fallback_ledger,
    reconcile,
)


def _write_fallback_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_id", "final_state", "blocker_code"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_historical_fallback_ledger_returns_empty_when_path_missing(tmp_path: Path) -> None:
    assert _load_historical_fallback_ledger(tmp_path / "no-such.csv") == {}
    assert _load_historical_fallback_ledger(None) == {}


def test_load_historical_fallback_ledger_loads_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "v245_ledger.csv"
    _write_fallback_ledger(
        ledger,
        [
            {"model_id": "alpha", "final_state": "benchmark_passed", "blocker_code": ""},
            {"model_id": "beta", "final_state": "smoke_passed", "blocker_code": ""},
        ],
    )
    loaded = _load_historical_fallback_ledger(ledger)
    assert "alpha" in loaded
    assert loaded["alpha"]["final_state"] == "benchmark_passed"
    assert loaded["beta"]["final_state"] == "smoke_passed"


def test_reconcile_accepts_historical_fallback_kwarg(tmp_path: Path) -> None:
    """Smoke test the reconciler kwarg threading. Full integration is the CLI test."""
    payload = reconcile(
        task_reports_root=tmp_path / "no-such",
        historical_fallback_ledger=tmp_path / "no-such-ledger.csv",
    )
    assert payload["total"] >= 0
    assert "rows" in payload

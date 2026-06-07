"""Validate the smart-tool coverage ledger (classic tools, separate from models)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPORTS = Path(__file__).resolve().parents[1] / "notebook" / "99_final_report" / "reports"
CSV = REPORTS / "smart_tool_coverage_ledger.csv"
JSON = REPORTS / "smart_tool_coverage_ledger.json"


@pytest.mark.skipif(not CSV.exists(), reason="smart_tool_coverage_ledger.csv not generated yet")
def test_ledger_separate_and_commercial_safe():
    rows = list(csv.DictReader(CSV.open()))
    assert len(rows) == 8
    ids = {r["tool_id"] for r in rows}
    assert all(i.startswith("classic-") for i in ids)
    for r in rows:
        assert r["requires_model_weights"] in ("False", "false"), r["tool_id"]
        assert r["commercial_safe"] in ("True", "true"), r["tool_id"]
        assert r["gated_or_auth_required"] in ("False", "false"), r["tool_id"]
        lic = r["dependency_license"].upper()
        assert "GPL" not in lic, f"{r['tool_id']} copyleft dep {lic}"
        assert "cpu" in r["device"]
        # benchmarked with a real metric
        assert 0.0 <= float(r["mean_iou"]) <= 1.0


@pytest.mark.skipif(not JSON.exists(), reason="ledger json not generated yet")
def test_ledger_json_no_model_weights_and_real_dataset():
    data = json.loads(JSON.read_text())
    assert "tools" in data and len(data["tools"]) == 8
    assert "real_images" in data["dataset"] or "promptable" in data["dataset"]
    for t in data["tools"]:
        assert t["requires_model_weights"] is False
        assert t["commercial_safe"] is True
        assert t["final_state"] == "benchmark_passed"

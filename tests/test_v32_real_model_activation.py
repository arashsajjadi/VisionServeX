"""v3.2 real-model-activation tests: new SAM runtime/video/ONNX modes + honest
blocker table for everything that can't lawfully ship."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "notebook" / "99_final_report" / "reports"


def _csv(name):
    p = R / name
    if not p.exists():
        pytest.skip(f"{name} not generated")
    return list(csv.DictReader(p.open()))


def test_real_activations_have_evidence_and_new_mode():
    rows = _csv("v32_real_model_activation_plan.csv")
    assert len(rows) >= 4
    valid_modes = {"onnx_cpu_runtime", "video_object_tracking", "transformers_image_backend"}
    for r in rows:
        assert r["new_mode"] in valid_modes, r
        assert r["evidence"], f"{r['model_id']} no evidence"
        assert r["final_state"] == "benchmark_passed"
        assert "Apache-2.0" in r["license"], f"{r['model_id']} not commercial-safe: {r['license']}"


def test_no_fake_benchmark_passed_without_evidence():
    rows = _csv("v32_real_model_activation_plan.csv")
    for r in rows:
        # every benchmark_passed real activation must reference a real artifact json
        assert ".json" in r["evidence"]


def test_blocker_table_has_escalation_and_next_command():
    rows = _csv("v32_failed_model_blockers.csv")
    assert len(rows) >= 8
    for r in rows:
        assert r["exact_blocker"].strip(), f"{r['model_id']} vague blocker"
        assert r["exact_next_command"].strip(), f"{r['model_id']} no next command"
        assert "->" in r["escalation_ladder"]
        assert r["final_state"] in {
            "sidecar_required", "checkpoint_required", "not_released",
            "legal_review_required", "external_api_only", "auth_required",
        }


def test_byot_no_token_leak_and_no_mirroring():
    rows = _csv("v32_byot_execution_ledger.csv")
    for r in rows:
        assert r["weights_mirrored"] in ("False", "false"), f"{r['model_id']} mirrors gated weights!"
        assert r["token_logged"] in ("False", "false"), f"{r['model_id']} logs token!"
        # without a token, BYOT models stay auth_required (never benchmark_passed)
        if r["token_present"] in ("False", "false"):
            assert r["final_state"] in ("auth_required", "external_api_only")


def test_onnx_export_runs_for_eligible_only():
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    from visionservex.onnx_export import onnx_eligible
    from visionservex.vsx import VSX, VSXError

    elig = onnx_eligible()
    assert "mobilesam" in elig and "edge-sam" not in elig  # non-commercial excluded
    with pytest.raises(VSXError):
        VSX.sam("edge-sam").to_onnx("/tmp/x.onnx")  # non-commercial refused


def test_sidecar_ledger_logs_exact_blocker():
    rows = _csv("v32_sidecar_execution_ledger.csv")
    mmlab = [r for r in rows if "openmmlab" in r["family"]]
    assert mmlab and "mmcv" in mmlab[0]["result"].lower()
    assert "conda" in mmlab[0]["next_command"]


def test_no_bad_license_in_core_still_holds():
    rows = _csv("model_coverage_ledger.csv")
    import re

    for r in rows:
        if str(r.get("default_safe", "")) == "True":
            assert not re.search("AGPL|GPL|S-Lab|non-commercial|NonCommercial", r.get("license_status", ""), re.I), r["model_id"]

"""v3.1 SAM/DINO model-expansion sprint tests.

Covers: activation KPI, SAM/DINO matrix completeness, VSX API contract, CV2-Pro
tools (license-safe, no GPL), no bad-license/EdgeSAM regression in core.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "notebook" / "99_final_report" / "reports"


def _csv(name):
    p = R / name
    if not p.exists():
        pytest.skip(f"{name} not generated yet")
    return list(csv.DictReader(p.open()))


# ---------------- activation KPI ----------------
def test_activation_kpi_at_least_20():
    rows = _csv("v31_model_activation_ledger.csv")
    counting = [r for r in rows if str(r.get("counts_toward_20", "")).lower() == "true"]
    assert len(counting) >= 20, f"only {len(counting)} counting activations: {[r['entry_id'] for r in counting]}"


def test_no_fake_benchmark_in_activations():
    # nothing may claim benchmark_passed without being a tool/pipeline with real evidence
    rows = _csv("v31_model_activation_ledger.csv")
    for r in rows:
        if r["new_state"] == "benchmark_passed":
            assert r["evidence"], f"{r['entry_id']} benchmark_passed without evidence"


# ---------------- matrices complete, no omitted/unknown rows ----------------
@pytest.mark.parametrize("name", ["v31_sam_family_matrix.csv", "v31_dino_family_matrix.csv"])
def test_matrix_no_absent_or_unknown_state(name):
    rows = _csv(name)
    assert len(rows) >= 20
    forbidden = {"absent", "unknown", "not_in_manifest", "missing", "", "nan"}
    for r in rows:
        st = r["target_state_after"].strip()
        assert st not in forbidden, f"{r['model_id']} has forbidden state {st!r}"
        assert r["exact_user_command"].strip(), f"{r['model_id']} missing next command"


def test_sam_family_covers_all_generations():
    rows = _csv("v31_sam_family_matrix.csv")
    gens = {r["generation"] for r in rows}
    assert {"SAM1", "SAM2", "SAM2.1", "SAM3", "SAM3.1", "lightweight"} <= gens


def test_pipelines_present_and_classified():
    rows = _csv("v31_sam_dino_pipeline_ledger.csv")
    assert len(rows) == 11
    for r in rows:
        assert r["pipeline_state"] in {
            "pipeline_demo_ready", "auth_required", "legal_review_required", "external_api_only", "blocked_on_part",
        }


# ---------------- VSX API contract ----------------
def test_vsx_facades_explain_contract():
    from visionservex import VSX, VSXError

    sam = VSX.sam("sam2.1-hiera-small").explain()
    assert {"model_id", "family", "task", "state", "license", "next_command"} <= set(sam)
    assert sam["state"] == "benchmark_passed"
    # gated SAM3 -> structured error with the exact lawful next command.
    # v3.8: segment() now performs a REAL access check, so the precise state is
    # auth_required (no token) or auth_required_license_pending (token present but
    # the upstream license not yet accepted). The next command always routes the
    # user to connect a token / accept the license.
    with pytest.raises(VSXError) as ei:
        VSX.sam("sam3-base").segment("x.jpg", box=[0, 0, 10, 10])
    assert ei.value.state in ("auth_required", "auth_required_license_pending")
    nxt = ei.value.next_command.lower()
    assert ("hf connect" in nxt) or ("hf_token" in nxt) or ("huggingface" in nxt) \
        or ("accept" in nxt)
    # DINO
    d = VSX.dino("dinov2-base").explain()
    assert d["state"] == "benchmark_passed" and d["task"] == "embed"
    assert VSX.dino("dino-x-api").explain()["state"] == "external_api_only"
    # pipeline inherits gated state
    assert VSX.pipeline("grounding-dino-1.6+sam3-base").status() == "auth_required"
    assert VSX.pipeline("grounding-dino-swin-t+sam2.1-hiera-small").status() == "pipeline_demo_ready"


def test_edgesam_not_regressed():
    from visionservex import VSX

    assert VSX.sam("edge-sam").explain()["state"] == "excluded_restricted"
    assert "NON-COMMERCIAL" in VSX.sam("edge-sam").explain()["license"]


# ---------------- CV2-Pro ----------------
def test_cv2_pro_tools_license_safe_and_runnable():
    pytest.importorskip("cv2")
    import numpy as np

    from visionservex.cv2_pro import TOOL_LICENSE, list_tools, run_tool, tool_available

    tools = list_tools()
    assert len(tools) >= 11
    for t, lic in TOOL_LICENSE().items():
        assert "GPL" not in lic.upper(), f"{t} copyleft: {lic}"
        assert "Apache-2.0" in lic
    img = (np.random.default_rng(0).normal(120, 30, (96, 96, 3))).clip(0, 255).astype("uint8")
    ran = 0
    for t in tools:
        if not tool_available(t)[0]:
            continue
        params = {"box": [10, 10, 80, 80]} if t == "opencv-grabcut-plus" else {}
        r = run_tool(t, img, **params)
        assert r["license_safe"] is True and r["device"] == "cpu"
        ran += 1
    assert ran >= 9


def test_cv2_pro_ledger_benchmarked():
    rows = _csv("v31_cv2_pro_tool_ledger.csv")
    bench = [r for r in rows if r["final_state"] == "tool_benchmark_passed"]
    assert len(bench) >= 11


# ---------------- no bad-license regression ----------------
def test_no_bad_license_in_default_safe_core():
    rows = _csv("model_coverage_ledger.csv")
    bad = "AGPL|GPL|non-commercial|NonCommercial|S-Lab|proprietary|Enterprise"
    import re

    for r in rows:
        if str(r.get("default_safe", "")) == "True" and re.search(bad, r.get("license_status", ""), re.I):
            raise AssertionError(f"bad license in default-safe core: {r['model_id']} {r['license_status']}")


def test_gated_models_not_default_safe():
    rows = {r["model_id"]: r for r in _csv("model_coverage_ledger.csv")}
    for mid in ("sam3-base", "grounding-dino-1.5", "grounding-dino-1.6", "dino-x-api"):
        if mid in rows:
            assert rows[mid]["default_safe"] == "False", f"{mid} must not be default_safe"

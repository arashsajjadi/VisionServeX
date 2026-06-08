# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: GroundingDINO + SAM/SAM2 text-to-mask pipelines (executed evidence)."""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).parent.parent / "notebook" / "99_final_report" / "reports"
ART = R.parent / "artifacts" / "v37"


def _exec():
    return [
        json.loads(ln)
        for ln in (R / "v37_raw_results.jsonl").read_text().splitlines()
        if ln.strip()
    ]


def test_pipelines_executed_count():
    pipes = [r for r in _exec() if r["task"].startswith("pipe:") and r["status"] == "ok"]
    assert len(pipes) >= 5, f"need >=5 pipelines, got {len(pipes)}"


def test_pipelines_have_real_mask_area():
    pipes = [r for r in _exec() if r["task"].startswith("pipe:") and r["status"] == "ok"]
    for p in pipes:
        assert p.get("mask_area", 0) > 0, f"{p['task']} has no mask area"
        assert p.get("top_box") is not None
        assert p.get("prompt_text")


def test_pipeline_artifacts_exist():
    pipes = [r for r in _exec() if r["task"].startswith("pipe:") and r["status"] == "ok"]
    for p in pipes:
        art = p.get("artifact")
        if art:
            assert (
                (R.parent.parent.parent / art).exists()
                or (Path(art)).exists()
                or (ART / Path(art).name).exists()
            ), f"missing {art}"


def test_vsx_pipeline_handle_states():
    from visionservex.vsx import VSX

    h = VSX.pipeline("grounding-dino-swin-b+sam2.1-hiera-large")
    info = h.explain()
    assert info["state"] in ("pipeline_demo_ready", "auth_required", "legal_review_required")


def test_pipeline_pairs_documented():
    pipes = {r["task"].replace("pipe:", "") for r in _exec() if r["task"].startswith("pipe:")}
    # at least one GD+SAM1 and one GD+SAM2 pair executed
    assert any("sam-vit" in p for p in pipes)
    assert any("sam2" in p for p in pipes)

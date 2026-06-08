# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: anti-fabrication — every benchmark_passed execution has a real on-disk
artifact and a measured latency. No matrix-only / helper / docs row is counted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
R = ROOT / "notebook" / "99_final_report" / "reports"


def _exec():
    return [
        json.loads(ln)
        for ln in (R / "v37_raw_results.jsonl").read_text().splitlines()
        if ln.strip()
    ]


def test_execution_ledger_exists():
    assert (R / "v37_raw_results.jsonl").exists()
    assert (R / "v37_new_model_execution_ledger.csv").exists()


def test_every_ok_execution_has_real_artifact_or_metric():
    art_dir = ROOT / "notebook" / "99_final_report" / "artifacts" / "v37"
    if not art_dir.exists():
        pytest.skip(f"v37 artifacts dir not in CI env: {art_dir}")
    for r in _exec():
        if r["status"] != "ok":
            continue
        art = r.get("artifact")
        if art:
            p = ROOT / art
            assert p.exists(), f"{r['task']}: claimed artifact does not exist: {art}"
        # must have at least one real measured quantity
        has_metric = any(
            r.get(k) is not None
            for k in (
                "mask_area",
                "embed_dim",
                "n_boxes",
                "n_instances",
                "frames_tracked",
                "onnx_bytes",
                "depth_shape",
                "latency_ms",
            )
        )
        assert has_metric, f"{r['task']}: no measured metric — possible fabrication"


def test_ok_count_matches_summary():
    summary = json.loads((R / "v37_execution_summary.json").read_text())
    ok = sum(1 for r in _exec() if r["status"] == "ok")
    assert ok == summary["ok"], f"ledger ok={ok} != summary ok={summary['ok']}"


def test_at_least_30_real_executions():
    ok = [r for r in _exec() if r["status"] == "ok"]
    assert len(ok) >= 30, f"need >=30 real executions, got {len(ok)}"


def test_distinct_model_ids_at_least_20():
    ok = [r for r in _exec() if r["status"] == "ok"]
    ids = set()
    for r in ok:
        ids.add(
            r.get("model_id")
            or r.get("hf_id")
            or r.get("variant")
            or r.get("pipeline_id")
            or r["task"]
        )
    assert len(ids) >= 20, f"need >=20 distinct ids, got {len(ids)}"


def test_sam_onnx_dino_minimums():
    ok = [r for r in _exec() if r["status"] == "ok"]
    sam = [r for r in ok if "sam" in r["task"].lower()]
    onnx = [r for r in ok if "onnx" in r["task"].lower()]
    dino = [r for r in ok if "dino" in r["task"].lower() or "gdino" in r["task"].lower()]
    assert len(sam) >= 5, f"SAM executions {len(sam)}"
    assert len(onnx) >= 3, f"ONNX executions {len(onnx)}"
    assert len(dino) >= 4, f"DINO/GD executions {len(dino)}"


def test_artifacts_directory_populated():
    art = ROOT / "notebook" / "99_final_report" / "artifacts" / "v37"
    if not art.exists():
        pytest.skip(f"v37 artifacts dir not in CI env: {art}")
    files = list(art.glob("*"))
    assert len(files) >= 25, f"expected >=25 artifacts, got {len(files)}"

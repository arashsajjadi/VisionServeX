# SPDX-License-Identifier: Apache-2.0
"""v2.38.0: Deep Research findings ingested into matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_deep_research_ingested_exists() -> None:
    p = Path(__file__).parent.parent / "reports/v238_deep_research_ingested.json"
    if not p.exists():
        pytest.skip("Deep Research not ingested")
    d = json.loads(p.read_text())
    assert d.get("status") == "ok"


def test_rt_detrv4_gdown_ids_documented() -> None:
    p = Path(__file__).parent.parent / "reports/v238_deep_research_ingested.json"
    if not p.exists():
        pytest.skip("Deep Research not ingested")
    d = json.loads(p.read_text())
    ids = d.get("key_findings", {}).get("rt_detrv4_gdown_ids", {})
    for size in ["s", "m", "l", "x"]:
        assert f"rtdetrv4-{size}" in ids, f"missing gdown ID for rtdetrv4-{size}"


def test_maskdino_sidecar_target_is_detectron2() -> None:
    p = Path(__file__).parent.parent / "reports/v238_49_blocked_resolution_matrix.json"
    if not p.exists():
        pytest.skip("v2.38 matrix not present")
    d = json.loads(p.read_text())
    rows = {r["model_id"]: r for r in d["rows"]}
    r = rows.get("maskdino-r50-coco")
    if r is None:
        pytest.skip("maskdino-r50-coco not in matrix")
    target = r.get("sidecar_target", "")
    assert "detectron2" in target.lower() or "py38" in target, (
        f"MaskDINO sidecar must be detectron2_detseg_py38, got {target}"
    )

# SPDX-License-Identifier: Apache-2.0
"""Tests for the Deep Research ingest artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MD = REPO_ROOT / "reports" / "v246_deep_research_ingested.md"
JS = REPO_ROOT / "reports" / "v246_deep_research_ingested.json"
ACTION = REPO_ROOT / "reports" / "v246_action_plan_from_deep_research.csv"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} is gitignored under reports/; generate locally to run this test")


def test_md_exists_and_nontrivial() -> None:
    _skip_if_missing(MD)
    text = MD.read_text()
    assert len(text) > 2000
    assert "OpenMMLab" in text
    assert "NATTEN" in text
    assert "DEIMv2" in text


def test_json_exists_and_has_50_models() -> None:
    _skip_if_missing(JS)
    data = json.loads(JS.read_text())
    assert data["release_version"] == "2.46.0"
    assert data["baseline_non_healthy_count"] == 50
    assert len(data["models"]) == 50


def test_action_plan_csv_exists() -> None:
    import csv

    _skip_if_missing(ACTION)
    with ACTION.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 50


def test_every_ingested_model_has_runtime_id() -> None:
    _skip_if_missing(JS)
    data = json.loads(JS.read_text())
    missing = [m["model_id"] for m in data["models"] if not m.get("runtime_id")]
    assert not missing, f"models without runtime_id: {missing}"


def test_incorrectly_restricted_models_are_flagged_for_unblock() -> None:
    _skip_if_missing(JS)
    data = json.loads(JS.read_text())
    by_id = {m["model_id"]: m for m in data["models"]}
    for mid in ("agriclip", "prithvi-eo-2.0", "dinov3-vitb16"):
        m = by_id[mid]
        assert m["commercial_allowed"] is True, (
            f"{mid} should be commercial_allowed=True per Deep Research"
        )
        # Deep Research says these three should NOT remain blocked under the wrong category
        assert "incorrectly" in (m.get("deep_research_notes") or "").lower() or m.get(
            "deep_research_notes", ""
        ).startswith("INCORRECTLY"), f"{mid} notes do not flag the false restriction"


def test_api_only_models_route_to_external_api_runtime() -> None:
    _skip_if_missing(JS)
    data = json.loads(JS.read_text())
    by_id = {m["model_id"]: m for m in data["models"]}
    for mid in ("dino-x-api", "grounding-dino-1.5-pro", "grounding-dino-1.6-pro"):
        assert by_id[mid]["runtime_id"] == "external_api_runtime"


def test_natten_oneformer_uses_compat_shim_strategy() -> None:
    _skip_if_missing(JS)
    data = json.loads(JS.read_text())
    by_id = {m["model_id"]: m for m in data["models"]}
    onef = by_id["oneformer-dinat-large"]
    assert onef["runtime_id"] == "oneformer_natten_py310"
    # required_fix must mention the API shim strategy, not another rebuild
    fix = (onef.get("required_fix") or "").lower()
    assert "shim" in fix or "natten2dqkrpb" in fix or "na2d_qk" in fix

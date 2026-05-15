# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for the recommend command and recommendation engine."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from visionservex.cli.main import app
from visionservex.registry import default_registry
from visionservex.runtime.recommendations import first_beginner_pick, recommend


runner = CliRunner()


def test_recommend_returns_results():
    recs = recommend(task="detect", limit=3)
    assert len(recs) <= 3
    assert recs, "expected at least one recommendation for `detect`"
    for r in recs:
        assert r.entry.task == "detect"
        assert r.score is not None
        assert isinstance(r.reasons, list)


def test_recommend_simple_prefers_beginner():
    recs = recommend(task="detect", simple=True, limit=5)
    assert recs
    top = recs[0].entry
    # The very top recommendation in `simple` mode should be a wired model
    # that auto-downloads (no manual install path).
    assert top.implementation_status in {"wired", "partial"}, top.id


def test_first_beginner_pick_returns_model():
    pick = first_beginner_pick(task="detect")
    assert pick is not None
    assert pick.task == "detect"


def test_recommend_cli_json():
    r = runner.invoke(app, ["recommend", "--task", "detect", "--simple", "--limit", "3", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, list)
    assert all("model_id" in item for item in data)


def test_recommend_filters_by_device():
    recs = recommend(task="detect", device="cpu", limit=5)
    for r in recs:
        assert "cpu" in {d.lower() for d in r.entry.supported_devices}


def test_registry_required_fields():
    reg = default_registry()
    for e in reg.list():
        assert e.id
        assert e.task
        assert e.license
        assert e.upstream_url
        assert e.engine
        assert e.difficulty in {"very_easy", "easy", "medium", "hard", "expert"}
        assert e.implementation_status in {"wired", "partial", "stub"}
        assert e.download_type in {
            "huggingface", "github_release", "direct_url",
            "manual", "external_api", "not_available", "synthetic",
        }

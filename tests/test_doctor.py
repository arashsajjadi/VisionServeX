# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for the doctor CLI command and system probes."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from visionservex.cli.main import app
from visionservex.utils.system import collect, probe_dependencies

runner = CliRunner()


def test_collect_returns_basic_fields():
    info = collect().to_dict()
    assert "os" in info
    assert "python" in info
    assert "cache_path" in info
    assert "cpu" in info and info["cpu"]["logical_cores"] >= 1


def test_probe_dependencies_includes_known_packages():
    deps = probe_dependencies()
    assert "torch" in deps
    assert "transformers" in deps
    assert "huggingface_hub" in deps
    for _name, info in deps.items():
        assert "installed" in info
        assert info["installed"] in (True, False)


def test_doctor_json_smoke():
    r = runner.invoke(app, ["doctor", "--json"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert "system" in data
    assert "devices" in data
    assert "dependencies" in data
    assert "best_device" in data
    assert "beginner_pick" in data
    assert isinstance(data["warnings"], list)


def test_devices_command():
    r = runner.invoke(app, ["devices", "--json"])
    assert r.exit_code == 0
    items = json.loads(r.output)
    assert any(d["name"] == "cpu" and d["available"] for d in items)

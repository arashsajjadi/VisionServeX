# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7: CLI commands expose --explain / status with required fields."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest


def _vsx():
    b = shutil.which("visionservex")
    return [b] if b else [sys.executable, "-m", "visionservex"]


def _run(args, timeout=60):
    return subprocess.run(_vsx() + args, capture_output=True, text=True, timeout=timeout)


def test_cli_loads():
    r = _run(["--help"])
    assert r.returncode == 0


@pytest.mark.parametrize(
    "group", ["interactive", "segment-instances", "locate-anything", "sam", "dino"]
)
def test_subcommand_help(group):
    r = _run([group, "--help"])
    assert r.returncode == 0, (group, r.stderr[:200])
    assert "No such command" not in r.stderr


def test_interactive_list_json():
    r = _run(["interactive", "list", "--format", "json"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    ids = {x["model_id"] for x in data}
    assert {"ritm", "clickseg", "simpleclick", "focalclick"} <= ids


def test_interactive_status_explain():
    r = _run(["interactive", "status", "simpleclick", "--explain"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["state"] == "legal_review_required"
    assert data["commercial_safe"] is False


def test_segment_instances_explain():
    r = _run(["segment-instances", "--model", "rfdetr-seg-small", "--explain"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["state"] == "benchmark_passed"
    assert data["license"] == "Apache-2.0"
    # required explain fields
    for k in ["state", "license", "commercial_safe", "output_schema", "next_command"]:
        assert k in data, f"missing {k}"


def test_locate_anything_explain_excluded():
    r = _run(["locate-anything", "status", "locate-anything-3b"])
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["state"] == "excluded_restricted"

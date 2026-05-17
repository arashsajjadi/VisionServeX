# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.13.0: seg alias, audit validate, notebook manifest, Docker fixes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.fast
def test_seg_alias_exists():
    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["seg", "--help"])
    assert result.exit_code == 0, result.output
    assert "seg" in result.output.lower() or "segment" in result.output.lower()


@pytest.mark.fast
def test_audit_validate_returns_valid_for_docs_audit(tmp_path):

    # Copy current docs/audit into tmp and validate it
    import shutil

    shutil.copytree(ROOT / "docs" / "audit", tmp_path / "audit")
    from visionservex.cli.audit_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--audit-dir", str(tmp_path / "audit"), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["verdict"] == "VALID", f"audit has issues: {payload.get('issues')}"


@pytest.mark.fast
def test_audit_validate_invalid_on_bad_manifest(tmp_path):
    from visionservex.cli.audit_commands import app

    bad = tmp_path / "visionservex_notebook_input_manifest.json"
    bad.write_text("{}")  # missing required keys
    runner = CliRunner()
    result = runner.invoke(app, ["validate", "--audit-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["verdict"] == "INVALID"
    assert len(payload["issues"]) > 0


@pytest.mark.fast
def test_notebook_manifest_consumption_script_exists_and_passes():
    p = ROOT / "scripts" / "test_notebook_manifest_consumption.py"
    assert p.exists()
    # Run with --json to get structured output
    res = subprocess.run(
        [sys.executable, str(p), "--json"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    assert res.returncode == 0, f"Consumption script failed: {res.stdout[-500:]}"
    payload = json.loads(res.stdout)
    assert payload["verdict"] == "PASS"
    assert payload["model_counts"]["total"] >= 100


@pytest.mark.fast
def test_gated_models_have_sidecar_colab_mode():
    """Auth-required models must not be marked quick/balanced in Colab."""
    from visionservex.audit.builder import export_model_inventory

    inv = export_model_inventory()
    bad = [
        m["model_id"]
        for m in inv["models"]
        if m.get("requires_auth") and m.get("recommended_colab_mode") in ("quick", "balanced")
    ]
    assert not bad, f"gated models should have colab_mode=sidecar: {bad}"


@pytest.mark.fast
def test_mmrotate_dockerfile_uses_available_base():
    p = ROOT / "docker" / "mmrotate-legacy" / "Dockerfile"
    assert p.exists()
    body = p.read_text()
    assert "1.13.1-cuda11.6" in body, "should use pytorch 1.13.1+cu116 (1.13.0-cu117 was removed)"
    assert "cu116" in body, "mim install URL should use cu116 index"


@pytest.mark.fast
def test_maskdino_dockerfile_uses_no_build_isolation():
    p = ROOT / "docker" / "maskdino" / "Dockerfile"
    assert p.exists()
    body = p.read_text()
    assert "--no-build-isolation" in body, (
        "Detectron2 needs --no-build-isolation so torch is visible"
    )


@pytest.mark.fast
def test_version_is_at_least_2130():
    import visionservex

    parts = tuple(int(p) for p in visionservex.__version__.split(".")[:3])
    assert parts >= (2, 13, 0)

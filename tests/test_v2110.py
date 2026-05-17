# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.11.0 release infrastructure.

Covers: load-matrix-run, cli-audit, Docker/GHCR workflow, clean install
script, updated sidecar image tags, v3 readiness doc, and optional-extras
workflow pytest fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# load-matrix-run
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_load_matrix_run_returns_v3_gate_pass_for_blockers():
    from visionservex.cli.model_health_commands import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "load-matrix-run",
            "--mode",
            "unavailable_blocker_validate",
            "--ci-safe",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["v3_gate_pass"] is True
    assert payload["core_failures"] == 0
    assert "UNAVAILABLE_CONFIRMED" in payload["status_counts"]


@pytest.mark.fast
def test_load_matrix_run_all_modes_no_core_failures():
    from visionservex.cli.model_health_commands import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["load-matrix-run", "--mode", "all", "--ci-safe", "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["core_failures"] == 0
    assert payload["v3_gate_pass"] is True
    # 113 registry models
    assert payload["n_rows"] >= 100


@pytest.mark.fast
def test_load_matrix_run_writes_out_file(tmp_path):
    from visionservex.cli.model_health_commands import app

    out = tmp_path / "lmr.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "load-matrix-run",
            "--mode",
            "gated_auth_validate",
            "--ci-safe",
            "--format",
            "json",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["n_rows"] >= 1


@pytest.mark.fast
def test_load_matrix_run_expected_status_set():
    """Every row must produce a known status code."""
    from visionservex.cli.model_health_commands import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["load-matrix-run", "--mode", "all", "--ci-safe", "--format", "json"],
    )
    payload = json.loads(result.output)
    valid_statuses = {
        "PASS",
        "SKIP_EXPECTED",
        "GATED_AUTH_REQUIRED",
        "NON_CORE_BLOCKED",
        "UNAVAILABLE_CONFIRMED",
        "DEPENDENCY_MISSING",
        "RESOURCE_BLOCKED",
        "FAIL",
        "not_run",
    }
    for row in payload["rows"]:
        assert row["tested_result"] in valid_statuses, (
            f"{row['model_id']}: unexpected status {row['tested_result']!r}"
        )


# ---------------------------------------------------------------------------
# cli-audit
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_cli_audit_all_pass():
    import shutil

    if not shutil.which("visionservex"):
        pytest.skip("visionservex console script not installed")

    from visionservex.cli.dev_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["cli-audit", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["all_pass"], (
        f"CLI audit failures: {[r for r in payload['results'] if r['status'] != 'PASS']}"
    )


# ---------------------------------------------------------------------------
# Docker GHCR workflow and scripts
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_publish_sidecars_workflow_exists_with_packages_write():
    import yaml

    wf = ROOT / ".github" / "workflows" / "publish-sidecars.yml"
    assert wf.exists()
    data = yaml.safe_load(wf.read_text())
    assert data["permissions"]["packages"] == "write"
    assert "build-openmmlab" in data["jobs"]
    assert "build-mmrotate-legacy" in data["jobs"]
    assert "build-maskdino" in data["jobs"]


@pytest.mark.fast
def test_build_and_push_sidecars_script_exists_and_executable():
    p = ROOT / "scripts" / "build_and_push_sidecars.sh"
    assert p.exists() and p.stat().st_mode & 0o100
    # Script handles --dry-run and emits correct JSON structure
    import subprocess

    res = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert res.returncode == 0, f"Syntax error: {res.stderr}"


@pytest.mark.fast
def test_maskdino_dockerfile_exists():
    p = ROOT / "docker" / "maskdino" / "Dockerfile"
    assert p.exists()
    body = p.read_text()
    assert "detectron2" in body.lower()
    assert "MaskDINO" in body


@pytest.mark.fast
def test_docker_image_tags_updated_to_v3():
    """No sidecar script should reference the old v2.9.0 tag."""
    stale_tag = "visionservex-openmmlab:v2.9.0"
    stale_tag2 = "visionservex-mmrotate-legacy:v2.9.0"
    for fname in (
        "docker/openmmlab/Dockerfile",
        "docker/mmrotate-legacy/Dockerfile",
        "scripts/build_openmmlab_sidecar.sh",
        "scripts/run_openmmlab_sidecar_smoke.sh",
        "scripts/build_mmrotate_legacy_sidecar.sh",
        "scripts/run_mmrotate_oriented_rcnn_smoke.sh",
    ):
        body = (ROOT / fname).read_text()
        assert stale_tag not in body, f"{fname} still references stale {stale_tag}"
        assert stale_tag2 not in body, f"{fname} still references stale {stale_tag2}"


# ---------------------------------------------------------------------------
# Clean install script
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_clean_wheel_install_script_exists_and_executable():
    p = ROOT / "scripts" / "test_clean_wheel_install.sh"
    assert p.exists() and p.stat().st_mode & 0o100
    import subprocess

    res = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
    assert res.returncode == 0


# ---------------------------------------------------------------------------
# v3 readiness doc
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_v3_readiness_doc_exists():
    p = ROOT / "docs" / "release_readiness" / "v3.0.0.md"
    assert p.exists()
    body = p.read_text()
    assert "v3 gate checklist" in body
    assert "load-matrix-run" in body
    assert "optional-extras" in body


@pytest.mark.fast
def test_latest_readiness_pointer_updated():
    p = ROOT / "docs" / "release_readiness" / "latest.md"
    body = p.read_text()
    assert "v3.0.0" in body


# ---------------------------------------------------------------------------
# optional-extras workflow has pytest install
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_optional_extras_tracking_job_installs_pytest():
    import yaml

    wf = ROOT / ".github" / "workflows" / "optional-extras-smoke.yml"
    data = yaml.safe_load(wf.read_text())
    job = data["jobs"]["tracking-smoke"]
    steps = job["steps"]
    install_step = next((s for s in steps if "Install" in (s.get("name") or "")), None)
    assert install_step is not None, "No install step found"
    run_text = install_step.get("run", "")
    assert "pytest" in run_text, "pytest not installed in tracking-smoke job"


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_at_least_2110():
    import visionservex

    parts = tuple(int(p) for p in visionservex.__version__.split(".")[:3])
    assert parts >= (2, 11, 0)

# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.9.0: 90% readiness across all factors.

Every row in the readiness table must be release-ready, MaskDINO must
ship real checkpoint URLs, certified blockers must include the v2.9
schema, the OpenMMLab Dockerfile + sidecar scripts must exist, and the
MMRotate legacy sidecar artifacts must be in place."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Readiness table — release rule
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_readiness_verdict_is_release_ok():
    from visionservex.cli.readiness_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["verdict", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["all_ready"] is True
    assert payload["verdict"] == "RELEASE_OK"
    assert payload["n_rows"] >= 18, f"readiness table shrunk to {payload['n_rows']}"
    assert payload["not_ready"] == []


@pytest.mark.fast
def test_every_readiness_row_passes_release_rule():
    from visionservex.readiness import READINESS_ROWS, is_row_release_ready

    failing = [r.factor for r in READINESS_ROWS if not is_row_release_ready(r)]
    assert failing == [], f"rows still below release rule: {failing}"


@pytest.mark.fast
def test_readiness_table_writes_markdown(tmp_path):
    from visionservex.cli.readiness_commands import app

    md = tmp_path / "readiness.md"
    out = tmp_path / "readiness.json"
    runner = CliRunner()
    result = runner.invoke(app, ["table", "--md", str(md), "--out", str(out), "--json"])
    assert result.exit_code == 0
    assert md.exists() and out.exists()
    body = md.read_text()
    assert "| Factor |" in body
    assert "Runnable practical capacity" in body
    payload = json.loads(out.read_text())
    assert payload["all_ready"] is True


# ---------------------------------------------------------------------------
# MaskDINO checkpoint URLs registered
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_maskdino_swinl_coco_has_real_checkpoint_url():
    from visionservex.cli.maskdino_commands import _MASKDINO_MODELS

    entry = _MASKDINO_MODELS["maskdino-swinl-coco"]
    assert entry["checkpoint_url"].startswith(
        "https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/"
    )
    assert entry["checkpoint_filename"].endswith(".pth")
    assert entry["license"] == "Apache-2.0"
    assert entry["checkpoint_source"] == "official_upstream"


@pytest.mark.fast
def test_maskdino_validate_now_surfaces_checkpoint_url():
    from visionservex.cli.maskdino_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["validate", "maskdino-swinl-coco", "--json"])
    payload = json.loads(result.stdout)
    assert payload["checkpoint_url"].startswith("https://github.com/IDEA-Research/")
    assert payload["release_page"].endswith("maskdino-v0.1.0")


@pytest.mark.fast
def test_maskdino_panoptic_entries_registered():
    from visionservex.cli.maskdino_commands import _MASKDINO_MODELS

    panoptic = [k for k, v in _MASKDINO_MODELS.items() if v.get("task") == "panoptic"]
    assert "maskdino-r50-coco-panoptic" in panoptic
    assert "maskdino-swinl-coco-panoptic" in panoptic


# ---------------------------------------------------------------------------
# Certified blockers — full v2.9 schema
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_certified_blocker_maskdino_schema_complete():
    from visionservex.cli.model_zoo_commands import CERTIFIED_BLOCKERS

    cert = CERTIFIED_BLOCKERS["maskdino"]
    for key in (
        "family",
        "variants",
        "official_repo",
        "license",
        "install_route",
        "checkpoint_status",
        "loader_status",
        "config_status",
        "exact_missing_piece",
        "tested_commands",
        "source_files_checked",
        "date_checked",
        "future_unblock_condition",
        "recommended_route",
        "status",
        "blocker_certainty",
    ):
        assert key in cert, f"maskdino certified blocker missing {key!r}"
    assert cert["blocker_certainty"] >= 95


@pytest.mark.fast
def test_model_zoo_blockers_refresh_emits_certified_payload(tmp_path):
    from visionservex.cli.model_zoo_commands import app

    out = tmp_path / "blocker.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["blockers", "--family", "maskdino", "--refresh", "--out", str(out)],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text())
    assert payload["family"] == "maskdino"
    assert payload["blocker_certainty"] >= 95
    assert payload["status"] == "optional_sidecar"


@pytest.mark.fast
def test_certified_blockers_all_have_certainty_above_90():
    from visionservex.cli.model_zoo_commands import CERTIFIED_BLOCKERS

    for family, cert in CERTIFIED_BLOCKERS.items():
        assert cert["blocker_certainty"] >= 90, f"{family} certainty {cert['blocker_certainty']}"


# ---------------------------------------------------------------------------
# OpenMMLab Dockerfile + sidecar CLI plumbing
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_dockerfile_exists_and_pins_recipe():
    p = ROOT / "docker" / "openmmlab" / "Dockerfile"
    assert p.exists()
    body = p.read_text()
    for pin in ("setuptools<72", "mmcv==2.1.0", "mmpose==1.3.2", "mmdet==3.3.0", "numpy==1.26.4"):
        assert pin in body, f"OpenMMLab Dockerfile missing pin {pin!r}"


@pytest.mark.fast
def test_openmmlab_dockerfile_cli_emits_pin_recipe():
    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["dockerfile", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["exists"] is True
    pin = payload["pin_recipe"]
    assert pin["python"] == "3.10"
    assert pin["mmcv"] == "2.1.0"
    assert pin["numpy"] == "1.26.4"


@pytest.mark.fast
def test_openmmlab_sidecar_smoke_blocked_when_docker_missing():
    """If Docker isn't on PATH, sidecar-smoke must return DOCKER_REQUIRED."""
    import shutil

    if shutil.which("docker"):
        pytest.skip("Docker available on this host; blocker path doesn't apply")
    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["sidecar-smoke", "rtmpose-m", "--image", "examples/images/person.jpg", "--json"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["code"] == "DOCKER_REQUIRED"


# ---------------------------------------------------------------------------
# MMRotate legacy sidecar
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_mmrotate_legacy_dockerfile_exists():
    p = ROOT / "docker" / "mmrotate-legacy" / "Dockerfile"
    assert p.exists()
    body = p.read_text()
    assert "mmcv-full==1.7.2" in body
    assert "mmdet==2.28.2" in body
    assert "mmrotate==0.3.4" in body
    assert "torch" in body


@pytest.mark.fast
def test_mmrotate_legacy_build_and_run_scripts_executable():
    for name in ("build_mmrotate_legacy_sidecar.sh", "run_mmrotate_oriented_rcnn_smoke.sh"):
        p = ROOT / "scripts" / name
        assert p.exists(), name
        assert p.stat().st_mode & 0o100, f"{name} must be executable"


@pytest.mark.fast
def test_obb_smoke_test_payload_includes_legacy_blocker():
    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["smoke-test", "oriented-rcnn", "--json"])
    payload = json.loads(result.stdout)
    assert payload["structured_error_code"] == "OBB_INFERENCER_UNAVAILABLE"
    assert payload["blocker_certainty"] >= 95
    assert "scripts/run_mmrotate_oriented_rcnn_smoke.sh" in payload["fix"]
    assert payload["legacy_image_tag"].startswith("visionservex-mmrotate-legacy")
    for key in ("x_center", "theta", "score", "label", "image", "runtime_ms"):
        assert key in payload["obb_schema"]


# ---------------------------------------------------------------------------
# TotalSegmentator sidecar
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_totalsegmentator_script_exists_and_blocks_missing_input(tmp_path):
    import subprocess

    p = ROOT / "scripts" / "run_totalsegmentator_smoke.sh"
    assert p.exists() and p.stat().st_mode & 0o100
    res = subprocess.run(
        ["bash", str(p), "/tmp/__definitely_not_a_real_path__.nii.gz"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 3
    assert "INPUT_NOT_FOUND" in res.stdout


# ---------------------------------------------------------------------------
# Sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_290():
    import visionservex

    assert visionservex.__version__ == "2.9.0"

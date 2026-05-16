# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.8.0: OpenMMLab real RTMPose/RTMDet smoke shape, OBB structured
blocker, optional-extras CI workflow, and HF real-smoke status."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# OpenMMLab CLI — model-card + smoke-test plumbing
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_rtmdet_tiny_coco_model_card_has_real_url():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "rtmdet-tiny-coco", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "detect"
    assert payload["inferencer"] == "mmdet.apis.DetInferencer"
    assert payload["config_name"] == "rtmdet_tiny_8xb32-300e_coco"
    assert payload["download_url"].startswith("https://download.openmmlab.com/mmdetection/")
    assert payload["checkpoint_filename"].endswith(".pth")


@pytest.mark.fast
def test_openmmlab_rtmpose_m_model_card_unchanged_url():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["model-card", "rtmpose-m", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["task"] == "pose"
    assert payload["inferencer"] == "mmpose.apis.MMPoseInferencer"
    assert "rtmpose-m" in payload["checkpoint_filename"]


@pytest.mark.fast
def test_openmmlab_smoke_test_obb_returns_schema():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(oml_app, ["smoke-test", "oriented-rcnn", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["structured_error_code"] == "OBB_INFERENCER_UNAVAILABLE"
    schema = payload["obb_schema"]
    # OBB must not be flattened to xyxy — theta is the load-bearing field.
    for key in ("x_center", "y_center", "width", "height", "theta", "score", "label"):
        assert key in schema, f"OBB schema missing {key!r}"


@pytest.mark.fast
def test_openmmlab_smoke_test_accepts_device_and_out_flags():
    """Even with mmpose missing on host, --device/--out must parse."""
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(
        oml_app,
        [
            "smoke-test",
            "rtmpose-m",
            "--device",
            "cpu",
            "--out",
            "/tmp/vsx_rtmpose_unused.json",
            "--json",
        ],
    )
    # Host has no mmpose; expect a structured API blocker, not a crash.
    assert result.exit_code in (0, 4)
    payload = json.loads(result.stdout)
    # Either constructed inferencer dry-run path or blocker.
    assert payload.get("status") in {"dry_run", "error", "ok"} or payload.get(
        "structured_error_code"
    )


@pytest.mark.fast
def test_openmmlab_smoke_test_unknown_model_id_returns_skipped():
    from visionservex.cli.openmmlab_commands import app as oml_app

    runner = CliRunner()
    result = runner.invoke(
        oml_app, ["smoke-test", "not-a-real-mm-model", "--device", "cpu", "--json"]
    )
    payload = json.loads(result.stdout)
    # Skipped (no metadata + unknown task) is the honest path.
    assert payload.get("status") in {"skipped", "error", "dry_run"}


# ---------------------------------------------------------------------------
# Optional-extras CI workflow
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_optional_extras_smoke_workflow_exists():
    wf = ROOT / ".github" / "workflows" / "optional-extras-smoke.yml"
    assert wf.exists()


@pytest.mark.fast
def test_optional_extras_smoke_workflow_jobs_complete():
    import yaml

    wf = ROOT / ".github" / "workflows" / "optional-extras-smoke.yml"
    data = yaml.safe_load(wf.read_text())
    jobs = data["jobs"]
    for name in (
        "tracking-smoke",
        "reid-smoke",
        "anomaly-smoke",
        "openmmlab-rtmpose-smoke",
    ):
        assert name in jobs, f"{name} missing from optional-extras workflow"
        # Make each job opt-in by continuing on error so heavy envs don't break merges.
        assert jobs[name].get("continue-on-error") is True


# ---------------------------------------------------------------------------
# Sidecar scripts ship and are executable
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_rtmpose_sidecar_script_exists_and_executable():
    p = ROOT / "scripts" / "run_openmmlab_rtmpose_smoke.sh"
    assert p.exists()
    assert p.stat().st_mode & 0o100


@pytest.mark.fast
def test_rtmpose_sidecar_script_pins_setuptools_below_72():
    p = ROOT / "scripts" / "run_openmmlab_rtmpose_smoke.sh"
    body = p.read_text()
    # The whole point of the script is the pinned recipe — verify the pins.
    assert "setuptools<72" in body
    assert "mmcv==2.1.0" in body
    assert "numpy==1.26.4" in body
    assert "torch==2.1.0" in body
    assert "mmdet==3.3.0" in body


# ---------------------------------------------------------------------------
# Existing v2.8 sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_version_is_at_least_280():
    import visionservex

    parts = tuple(int(p) for p in visionservex.__version__.split(".")[:3])
    assert parts >= (2, 8, 0)

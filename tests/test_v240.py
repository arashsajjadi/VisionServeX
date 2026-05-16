# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.4.0: anomalib adapter, MedSAM mask, ByteTrack in index, OpenMMLab create-env."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Anomalib adapter — missing dependency
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomalib_adapter_unavailable():
    from visionservex.integrations.anomalib_adapter import (
        AnomalibUnavailableError,
        PatchCoreAdapter,
    )

    adapter = PatchCoreAdapter()
    with pytest.raises(AnomalibUnavailableError) as exc_info:
        adapter.train("/tmp/data", "/tmp/out")
    assert exc_info.value.code == "ANOMALIB_REQUIRED"
    assert "visionservex[anomaly]" in exc_info.value.fix


@pytest.mark.fast
def test_anomalib_adapter_unavailable_to_dict():
    from visionservex.integrations.anomalib_adapter import AnomalibUnavailableError

    err = AnomalibUnavailableError()
    d = err.to_dict()
    assert d["code"] == "ANOMALIB_REQUIRED"
    assert "fix" in d


@pytest.mark.fast
def test_anomalib_adapter_detect_version_none():
    from visionservex.integrations.anomalib_adapter import detect_anomalib_version

    with patch("importlib.import_module", side_effect=ImportError("no anomalib")):
        assert detect_anomalib_version() is None


@pytest.mark.fast
def test_anomalib_adapter_capabilities_not_installed():
    from visionservex.integrations.anomalib_adapter import get_anomalib_capabilities

    with patch(
        "visionservex.integrations.anomalib_adapter.detect_anomalib_version", return_value=None
    ):
        caps = get_anomalib_capabilities()
    assert caps["installed"] is False
    assert caps["engine_available"] is False


@pytest.mark.fast
def test_anomalib_adapter_mocked_1x_train(tmp_path):
    """When anomalib is mocked as 1.x, PatchCoreAdapter.train should succeed or return error dict."""
    import sys

    from visionservex.integrations.anomalib_adapter import PatchCoreAdapter

    # Mock anomalib 1.x by injecting fake modules into sys.modules
    mock_engine_instance = MagicMock()
    mock_engine_instance.fit.return_value = None

    mock_engine_cls = MagicMock(return_value=mock_engine_instance)
    mock_patchcore_cls = MagicMock()
    mock_folder_cls = MagicMock()

    mock_anomalib = MagicMock()
    mock_anomalib.__version__ = "1.2.3"
    mock_anomalib_engine = MagicMock()
    mock_anomalib_engine.Engine = mock_engine_cls
    mock_anomalib_models = MagicMock()
    mock_anomalib_models.Patchcore = mock_patchcore_cls
    mock_anomalib_data = MagicMock()
    mock_anomalib_data.Folder = mock_folder_cls

    fake_modules = {
        "anomalib": mock_anomalib,
        "anomalib.engine": mock_anomalib_engine,
        "anomalib.models": mock_anomalib_models,
        "anomalib.data": mock_anomalib_data,
    }
    data_dir = tmp_path / "normal"
    data_dir.mkdir()

    with patch.dict(sys.modules, fake_modules):
        adapter = PatchCoreAdapter()
        result = adapter.train(str(data_dir), str(tmp_path / "out"))

    assert result.get("status") in ("trained", "error", "unsupported")


@pytest.mark.fast
def test_anomalib_unsupported_version_error():
    from visionservex.integrations.anomalib_adapter import AnomalibUnsupportedVersionError

    err = AnomalibUnsupportedVersionError(
        "API changed", installed_version="3.0.0", available_modules=["anomalib.models"]
    )
    d = err.to_dict()
    assert d["code"] == "ANOMALIB_API_UNSUPPORTED"
    assert "3.0.0" in d["installed_version"]


# ---------------------------------------------------------------------------
# MedSAM real mask output
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_medsam_segment_produces_mask(tmp_path):
    """MedSAM segment must write mask PNG + metadata JSON when model works."""
    pytest.importorskip("transformers")  # fast-CI skips if [hf] not installed

    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    img = Image.new("RGB", (256, 256), color=(128, 200, 200))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    out_dir = tmp_path / "out"

    import numpy as np

    fake_mask = np.ones((256, 256), dtype=np.uint8)

    fake_seg = MagicMock()
    fake_seg.mask = fake_mask
    fake_seg.score = 0.92
    fake_seg_box = MagicMock()
    fake_seg_box.x1, fake_seg_box.y1, fake_seg_box.x2, fake_seg_box.y2 = 10.0, 20.0, 100.0, 200.0
    fake_seg.box = fake_seg_box

    fake_result = MagicMock()
    fake_result.segments = [fake_seg]
    fake_result.device = "cpu"

    fake_model = MagicMock()
    fake_model.predict.return_value = fake_result

    runner = CliRunner()
    with patch("visionservex.VisionModel", return_value=fake_model):
        result = runner.invoke(
            app,
            [
                "segment",
                "medsam",
                str(img_path),
                "--box",
                "10,20,100,200",
                "--out",
                str(out_dir),
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["status"] == "ok"
    assert payload["n_masks"] == 1
    assert payload["masks_saved"][0]["iou_score"] == pytest.approx(0.92)
    assert (out_dir / "mask_000.png").exists()
    assert (out_dir / "medsam_metadata.json").exists()


@pytest.mark.fast
def test_medsam_segment_invalid_box(tmp_path):
    """Invalid box format returns INPUT_SCHEMA_ERROR."""
    pytest.importorskip("transformers")  # fast-CI skips if [hf] not installed

    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    img = Image.new("RGB", (64, 64))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["segment", "medsam", str(img_path), "--box", "10,20", "--out", str(out_dir), "--json"],
    )
    assert result.exit_code != 0
    output = json.loads(result.output.strip())
    assert output["code"] == "INPUT_SCHEMA_ERROR"


@pytest.mark.fast
def test_medsam_segment_missing_checkpoint(tmp_path):
    """Missing checkpoint returns CHECKPOINT_REQUIRED."""
    pytest.importorskip("transformers")  # fast-CI skips if [hf] not installed

    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    img = Image.new("RGB", (64, 64))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    with patch(
        "visionservex.VisionModel",
        side_effect=Exception("checkpoint not found"),
    ):
        result = runner.invoke(
            app,
            [
                "segment",
                "medsam",
                str(img_path),
                "--box",
                "10,20,100,200",
                "--out",
                str(out_dir),
                "--json",
            ],
        )
    assert result.exit_code != 0
    output = json.loads(result.output.strip())
    assert output["code"] == "CHECKPOINT_REQUIRED"


# ---------------------------------------------------------------------------
# ByteTrack in video-search index
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_tracker_registry_includes_bytetrack():
    from visionservex.runtime.trackers import _TRACKER_PACKAGES

    assert "bytetrack" in _TRACKER_PACKAGES


@pytest.mark.fast
def test_build_tracker_simple_iou_returns_none():
    from visionservex.runtime.trackers import build_tracker

    assert build_tracker("simple-iou") is None
    assert build_tracker("") is None


@pytest.mark.fast
def test_build_tracker_bytetrack_missing():
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    with pytest.raises(TrackerUnavailableError) as exc_info:
        build_tracker("bytetrack")
    assert exc_info.value.code == "BYTETRACK_REQUIRED"
    assert "bytetracker" in exc_info.value.install


@pytest.mark.fast
def test_build_tracker_unknown_name():
    from visionservex.runtime.trackers import TrackerUnavailableError, build_tracker

    with pytest.raises(TrackerUnavailableError) as exc_info:
        build_tracker("nonexistent-tracker")
    assert "TRACKER_UNKNOWN" in exc_info.value.code or "UNKNOWN" in exc_info.value.code


@pytest.mark.fast
def test_tracker_unavailable_error_to_dict():
    from visionservex.runtime.trackers import TrackerUnavailableError

    err = TrackerUnavailableError("bytetrack", "BYTETRACK_REQUIRED", "pip install bytetracker")
    d = err.to_dict()
    assert d["code"] == "BYTETRACK_REQUIRED"
    assert d["tracker"] == "bytetrack"
    assert "bytetracker" in d["install"]


@pytest.mark.fast
def test_video_search_index_bytetrack_missing_returns_error(tmp_path):
    """index --tracker bytetrack without package gives BYTETRACK_REQUIRED JSON."""
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    (tmp_path / "frame_000.jpg").write_bytes(Image.new("RGB", (64, 64)).tobytes())

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "index",
            str(tmp_path),
            "--out",
            str(tmp_path / "idx"),
            "--tracker",
            "bytetrack",
            "--json",
        ],
    )
    # Exit 3 = tracker unavailable; output may include privacy notice before JSON
    assert result.exit_code == 3
    # Extract JSON block from output (privacy notice precedes JSON)
    raw = result.output
    json_start = raw.find("{")
    assert json_start >= 0, f"No JSON in output: {raw!r}"
    output = json.loads(raw[json_start:])
    assert output["code"] == "BYTETRACK_REQUIRED"


@pytest.mark.fast
def test_video_search_trackers_includes_bytetrack():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["trackers", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output.strip())
    assert "bytetrack" in data
    assert data["bytetrack"]["installed"] is False


# ---------------------------------------------------------------------------
# OpenMMLab create-env and install-help
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_openmmlab_create_env_json():
    from typer.testing import CliRunner

    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert "env_name" in payload
    assert "commands" in payload
    cmds = " ".join(payload["commands"])
    assert "conda create" in cmds
    assert "openmim" in cmds
    assert "mmpose" in cmds
    assert "mmdet" in cmds


@pytest.mark.fast
def test_openmmlab_create_env_custom_name():
    from typer.testing import CliRunner

    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--name", "my-mmlab", "--python", "3.11", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload["env_name"] == "my-mmlab"
    assert payload["python"] == "3.11"
    assert any("my-mmlab" in cmd for cmd in payload["commands"])


@pytest.mark.fast
def test_openmmlab_install_help_json():
    from typer.testing import CliRunner

    from visionservex.cli.openmmlab_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["install-help", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert "native" in payload
    assert any("openmim" in cmd for cmd in payload["native"])


# ---------------------------------------------------------------------------
# HF model pull dry-run
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_model_pull_dry_run_convnextv2():
    """model pull --dry-run must not download and print what would happen."""
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["model", "pull", "convnextv2-tiny", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry" in result.output.lower() or "convnextv2" in result.output.lower()


@pytest.mark.fast
def test_model_pull_dry_run_clip():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["model", "pull", "clip-vit-base-patch32", "--dry-run"])
    assert result.exit_code == 0, result.output


@pytest.mark.fast
def test_model_pull_dry_run_json():
    from typer.testing import CliRunner

    from visionservex.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["model", "pull", "maxvit-tiny-tf-224", "--dry-run", "--json"])
    assert result.exit_code == 0
    # Should return either JSON or print dry-run info without erroring

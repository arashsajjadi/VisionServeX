# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.3.0: Florence-2 smoke, benchmark-anomaly mock, tracker/ReID doctor."""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Florence-2 create-env — updated pin
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_florence2_create_env_validated_pin():
    """create-env must include the validated transformers==4.46.3 pin."""
    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert payload["transformers_pin"] == "transformers==4.46.3"
    assert "einops" in payload["required_extras"]
    assert "timm" in payload["required_extras"]
    assert "validated_smoke_result" in payload
    assert "PASSED" in payload["validated_smoke_result"]


@pytest.mark.fast
def test_florence2_create_env_includes_einops_timm():
    """create-env commands must include einops and timm."""
    from typer.testing import CliRunner

    from visionservex.cli.florence2_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--json"])
    payload = json.loads(result.output.strip())
    cmds = " ".join(payload["commands"])
    assert "einops" in cmds
    assert "timm" in cmds


# ---------------------------------------------------------------------------
# benchmark-anomaly mock-anomaly path (no anomalib required)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_benchmark_anomaly_mock_no_anomalib(tmp_path):
    """mock-anomaly must produce valid JSON without anomalib installed."""
    from PIL import Image
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    # Create tiny simple dataset
    (tmp_path / "normal").mkdir()
    (tmp_path / "test").mkdir()
    for i in range(3):
        Image.new("RGB", (32, 32), color=(200, 200, 200)).save(tmp_path / "normal" / f"n{i}.png")
    for i in range(2):
        Image.new("RGB", (32, 32), color=(50, 0, 0)).save(tmp_path / "test" / f"t{i}.png")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--dataset", f"simple:{tmp_path}", "--model", "mock-anomaly", "--json"],
    )
    assert result.exit_code == 0, f"CLI crashed: {result.output}"
    payload = json.loads(result.output.strip())
    assert payload["benchmark"] == "anomaly"
    r = payload["result"]
    assert r["model"] == "mock-anomaly"
    assert r["n_normal_train"] == 3
    assert r["n_anomaly_test"] == 2
    assert r["error"] is None
    assert "mock-anomaly" in r["notes"]


@pytest.mark.fast
def test_benchmark_anomaly_patchcore_returns_alternative_tip(tmp_path):
    """patchcore without anomalib must include 'alternative' key pointing to mock-anomaly."""
    from unittest.mock import patch

    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    (tmp_path / "normal").mkdir()
    (tmp_path / "normal" / "n.png").write_bytes(b"")

    runner = CliRunner()
    with patch("importlib.import_module", side_effect=ImportError("no anomalib")):
        result = runner.invoke(
            app, ["--dataset", f"simple:{tmp_path}", "--model", "patchcore", "--json"]
        )
    payload = json.loads(result.output.strip())
    assert payload["code"] == "ANOMALIB_REQUIRED"
    assert "alternative" in payload
    assert "mock-anomaly" in payload["alternative"]


@pytest.mark.fast
def test_benchmark_anomaly_mvtec_mock(tmp_path):
    """mock-anomaly on MVTec layout must read train/good and test sub-dirs."""
    from PIL import Image
    from typer.testing import CliRunner

    from visionservex.cli.benchmark_anomaly_cmd import app

    (tmp_path / "train" / "good").mkdir(parents=True)
    (tmp_path / "test" / "good").mkdir(parents=True)
    (tmp_path / "test" / "crack").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (32, 32), color=(180, 180, 180)).save(
            tmp_path / "train" / "good" / f"n{i}.png"
        )
        Image.new("RGB", (32, 32), color=(180, 180, 180)).save(
            tmp_path / "test" / "good" / f"g{i}.png"
        )
    for i in range(2):
        Image.new("RGB", (32, 32), color=(30, 0, 0)).save(tmp_path / "test" / "crack" / f"c{i}.png")

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["--dataset", f"mvtec:{tmp_path}", "--model", "mock-anomaly", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    r = payload["result"]
    assert r["n_normal_train"] == 3
    assert r["n_normal_test"] == 3
    assert r["n_anomaly_test"] == 2


# ---------------------------------------------------------------------------
# video-search tracker / ReID commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_video_search_trackers_json():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["trackers", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "simple-iou" in data
    assert data["simple-iou"]["installed"] is True
    assert "bytetrack" in data
    assert data["bytetrack"]["installed"] is False
    assert "BYTETRACK_REQUIRED" in data["bytetrack"].get("blocker", "")


@pytest.mark.fast
def test_video_search_reid_models_json():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["reid-models", "--json"])
    assert result.exit_code == 0, result.output
    import importlib.util

    data = json.loads(result.output.strip())
    assert "cosine-siglip2" in data
    assert data["cosine-siglip2"]["installed"] is True
    assert "osnet" in data
    # osnet availability tracks whether torchreid is installed on the host (absent in clean CI).
    assert data["osnet"]["installed"] is (importlib.util.find_spec("torchreid") is not None)


@pytest.mark.fast
def test_video_search_doctor_bytetrack_missing():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--tracker", "bytetrack", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert data["tracker"]["installed"] is False
    assert data["tracker"]["code"] == "BYTETRACK_REQUIRED"
    assert "bytetracker" in data["tracker"]["install"]


@pytest.mark.fast
def test_video_search_doctor_osnet_missing():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--reid", "osnet", "--json"])
    assert result.exit_code == 0, result.output
    import importlib.util

    data = json.loads(result.output.strip())
    if importlib.util.find_spec("torchreid") is not None:
        # torchreid installed on the host -> osnet reports available, no missing-dep blocker.
        assert data["reid"]["installed"] is True
    else:
        assert data["reid"]["installed"] is False
        assert data["reid"]["code"] == "TORCHREID_REQUIRED"
        assert "torchreid" in data["reid"]["install"]


@pytest.mark.fast
def test_video_search_doctor_unknown_tracker():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--tracker", "nonexistent", "--json"])
    assert result.exit_code == 2
    data = json.loads(result.output.strip())
    assert data["code"] == "TRACKER_UNKNOWN"

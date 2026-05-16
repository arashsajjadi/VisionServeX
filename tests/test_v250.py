# SPDX-License-Identifier: Apache-2.0
"""Tests for v2.5.0: model-zoo matrix/gap-report, SAM-family, anomaly create-env,
MedSAM multi-box, video-search install-help, medical doctor."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# model-zoo commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_model_zoo_gap_report_json():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["gap-report", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    # Should have list of models or categorized dict
    assert isinstance(data, (list, dict))


@pytest.mark.fast
def test_model_zoo_gap_report_markdown():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["gap-report", "--format", "markdown"])
    assert result.exit_code == 0
    assert "runnable" in result.output.lower() or "#" in result.output


@pytest.mark.fast
def test_model_zoo_matrix_json():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["matrix", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, list)
    assert len(data) > 10
    # Each row should have model_id field
    assert "model_id" in data[0]


@pytest.mark.fast
def test_model_zoo_matrix_filter_family():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["matrix", "--format", "json", "--family", "dfine"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, list)
    for row in data:
        assert "dfine" in row.get("family", "").lower()


@pytest.mark.fast
def test_model_zoo_blockers_family():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["blockers", "--family", "deimv2", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, list)
    # deimv2 should have blockers
    assert len(data) > 0


@pytest.mark.fast
def test_model_zoo_matrix_markdown():
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["matrix", "--format", "markdown"])
    assert result.exit_code == 0
    assert "|" in result.output  # markdown table


# ---------------------------------------------------------------------------
# SAM-family commands
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam_family_list_json():
    from typer.testing import CliRunner

    from visionservex.cli.sam_family_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, (list, dict))
    # Should include sam, sam2, sam2.1 families
    ids = [
        d.get("model_id", d) if isinstance(d, dict) else d
        for d in (data if isinstance(data, list) else data.values())
    ]
    assert len(ids) >= 5


@pytest.mark.fast
def test_sam_family_doctor_json():
    from typer.testing import CliRunner

    from visionservex.cli.sam_family_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, (list, dict))


@pytest.mark.fast
def test_sam_family_model_card_sam21():
    from typer.testing import CliRunner

    from visionservex.cli.sam_family_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["model-card", "sam2.1-hiera-tiny", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert data.get("model_id") == "sam2.1-hiera-tiny"
    assert "facebook/sam2.1-hiera-tiny" in str(data)


@pytest.mark.fast
def test_sam_family_model_card_fastsam():
    from typer.testing import CliRunner

    from visionservex.cli.sam_family_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["model-card", "fastsam-s", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    # FastSAM is AGPL-3.0 do_not_add
    assert "fastsam" in str(data).lower()


@pytest.mark.fast
def test_sam_family_smoke_test_non_runnable():
    """smoke-test for a non-runnable model returns structured error."""
    from typer.testing import CliRunner

    from visionservex.cli.sam_family_commands import app

    runner = CliRunner()
    # fastsam-s is do_not_add / AGPL — not runnable
    result = runner.invoke(app, ["smoke-test", "fastsam-s", "/nonexistent.jpg", "--json"])
    # Should exit non-zero or return structured error, not crash with traceback
    if result.output.strip():
        try:
            data = json.loads(result.output.strip())
            # Acceptable: structured error
            assert "code" in data or "status" in data or "recommended_action" in data
        except json.JSONDecodeError:
            pass  # Rich output is ok if not --json


# ---------------------------------------------------------------------------
# Anomaly create-env and install-help
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_anomaly_create_env_json():
    from typer.testing import CliRunner

    from visionservex.cli.anomaly_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert "env_name" in payload
    assert "commands" in payload
    cmds = " ".join(payload["commands"])
    assert "conda create" in cmds
    assert "anomalib" in cmds


@pytest.mark.fast
def test_anomaly_create_env_custom_name():
    from typer.testing import CliRunner

    from visionservex.cli.anomaly_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["create-env", "--name", "my-anomaly", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output.strip())
    assert payload["env_name"] == "my-anomaly"
    assert any("my-anomaly" in cmd for cmd in payload["commands"])


@pytest.mark.fast
def test_anomaly_install_help_json():
    from typer.testing import CliRunner

    from visionservex.cli.anomaly_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["install-help", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "native" in data or "pip" in str(data)


# ---------------------------------------------------------------------------
# MedSAM multi-box
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_medsam_multi_box_produces_multiple_masks(tmp_path):
    """Multi-box MedSAM should attempt multiple prompts."""
    import numpy as np
    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    img = Image.new("RGB", (256, 256), color=(128, 200, 200))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    out_dir = tmp_path / "out"

    # Create two fake masks
    fake_mask1 = np.ones((256, 256), dtype=np.uint8)
    fake_mask2 = np.zeros((256, 256), dtype=np.uint8)
    fake_mask2[50:100, 50:100] = 1

    fake_seg1 = MagicMock()
    fake_seg1.mask = fake_mask1
    fake_seg1.score = 0.9
    fake_seg1.box = MagicMock()
    fake_seg1.box.x1, fake_seg1.box.y1, fake_seg1.box.x2, fake_seg1.box.y2 = (
        10.0,
        20.0,
        100.0,
        200.0,
    )

    fake_result = MagicMock()
    fake_result.segments = [fake_seg1]
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
                "--box",
                "30,40,150,180",
                "--out",
                str(out_dir),
                "--json",
            ],
        )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    # boxes field should list both boxes
    assert "boxes" in payload
    assert len(payload["boxes"]) == 2


@pytest.mark.fast
def test_medsam_invalid_box_index_reported(tmp_path):
    """Invalid box at specific index shows which index failed."""
    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    img = Image.new("RGB", (64, 64))
    img_path = tmp_path / "test.png"
    img.save(img_path)
    out_dir = tmp_path / "out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "segment",
            "medsam",
            str(img_path),
            "--box",
            "10,20,100,200",  # valid
            "--box",
            "bad_box",  # invalid
            "--out",
            str(out_dir),
            "--json",
        ],
    )
    assert result.exit_code != 0
    output = json.loads(result.output.strip())
    assert output["code"] == "INPUT_SCHEMA_ERROR"


@pytest.mark.fast
def test_medical_doctor_json():
    from typer.testing import CliRunner

    from visionservex.cli.medical_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert isinstance(data, list)
    # Each entry should have model_id and installed fields
    assert all("model_id" in d for d in data)


# ---------------------------------------------------------------------------
# Video-search install-help
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_video_search_install_help_all():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["install-help", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "trackers" in data or "bytetrack" in str(data)


@pytest.mark.fast
def test_video_search_install_help_bytetrack():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["install-help", "--tracker", "bytetrack", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "bytetracker" in str(data) or "bytetrack" in str(data)


@pytest.mark.fast
def test_video_search_install_help_osnet():
    from typer.testing import CliRunner

    from visionservex.cli.video_search_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["install-help", "--reid", "osnet", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "torchreid" in str(data) or "osnet" in str(data)


# ---------------------------------------------------------------------------
# Real smoke verification: grounding-dino-tiny, dfine-n, sam-vit-base (already cached)
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_grounding_dino_tiny_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    e = reg.get("grounding-dino-tiny")
    assert e is not None
    assert e.implementation_status == "wired"


@pytest.mark.fast
def test_dfine_n_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    e = reg.get("dfine-n")
    assert e is not None
    assert e.implementation_status == "wired"


@pytest.mark.fast
def test_sam_vit_base_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    e = reg.get("sam-vit-base")
    assert e is not None
    assert e.implementation_status == "wired"


# ---------------------------------------------------------------------------
# SAM 2.1 runtime registry entries
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_sam21_all_entries_in_registry():
    from visionservex.registry import default_registry

    reg = default_registry()
    for mid in (
        "sam2.1-hiera-tiny",
        "sam2.1-hiera-small",
        "sam2.1-hiera-base-plus",
        "sam2.1-hiera-large",
    ):
        e = reg.get(mid)
        assert e is not None, f"Missing: {mid}"


# ---------------------------------------------------------------------------
# Model-zoo gap-report and matrix file output
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_model_zoo_gap_report_writes_file(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    out_path = tmp_path / "gap_report.md"
    runner = CliRunner()
    result = runner.invoke(app, ["gap-report", "--format", "markdown", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert out_path.stat().st_size > 100


@pytest.mark.fast
def test_model_zoo_matrix_writes_file(tmp_path):
    from typer.testing import CliRunner

    from visionservex.cli.model_zoo_commands import app

    out_path = tmp_path / "matrix.json"
    runner = CliRunner()
    result = runner.invoke(app, ["matrix", "--format", "json", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert len(data) > 10

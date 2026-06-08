# SPDX-License-Identifier: Apache-2.0
"""v3.4 SAM runtime unblock tests — 7 tests covering CLI and VisionModel SAM paths."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"
_runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. sam_commands.app is importable and non-None
# ---------------------------------------------------------------------------


def test_sam_cli_app_importable():
    from visionservex.cli.sam_commands import app

    assert app is not None


# ---------------------------------------------------------------------------
# 2. sam status sam-vit-b --json returns model_id and onnx_eligible fields
# ---------------------------------------------------------------------------


def test_sam_status_sam_vit_b_json():
    from visionservex.cli.sam_commands import app

    result = _runner.invoke(app, ["status", "sam-vit-b", "--json"])
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}\n{result.output}"
    data = json.loads(result.output)
    assert "model_id" in data, "Response must contain 'model_id'"
    assert "onnx_eligible" in data, "Response must contain 'onnx_eligible'"


# ---------------------------------------------------------------------------
# 3. sam status sam3-base --json has auth_required=True and GATED/AUTH code
# ---------------------------------------------------------------------------


def test_sam_status_sam3_returns_auth_required():
    from visionservex.cli.sam_commands import app

    result = _runner.invoke(app, ["status", "sam3-base", "--json"])
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}\n{result.output}"
    data = json.loads(result.output)
    assert data.get("auth_required") is True, "sam3-base must report auth_required=True"
    blocker = (data.get("blocker") or data.get("code") or "").upper()
    assert "GATED" in blocker or "AUTH" in blocker, (
        f"Expected blocker to contain 'GATED' or 'AUTH', got: {blocker!r}"
    )


# ---------------------------------------------------------------------------
# 4. sam list --json returns at least one sam-family model
# ---------------------------------------------------------------------------


def test_sam_list_includes_expected_families():
    from visionservex.cli.sam_commands import app

    result = _runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}\n{result.output}"
    data = json.loads(result.output)
    assert isinstance(data, list), "sam list --json must return a JSON array"
    assert len(data) >= 1, "sam list must return at least one model"
    # At least one entry must belong to a sam family
    sam_families = {"sam", "sam2", "sam2.1", "sam3", "efficientsam", "mobilesam"}
    families_present = {entry.get("family", "") for entry in data}
    assert families_present & sam_families, (
        f"No sam-family model found. Families present: {families_present}"
    )


# ---------------------------------------------------------------------------
# 5. visionservex sam export-onnx --help exits 0
# ---------------------------------------------------------------------------


def test_sam_export_onnx_help():
    result = subprocess.run(
        [sys.executable, "-m", "visionservex", "sam", "export-onnx", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"export-onnx --help exited {result.returncode}\n"
        f"stdout: {result.stdout[:300]}\nstderr: {result.stderr[:300]}"
    )


# ---------------------------------------------------------------------------
# 6. VisionModel("sam-vit-b") predict returns SegmentationResult with segments
# ---------------------------------------------------------------------------


def test_sam_vit_b_runnable_via_vision_model():
    if not _IMG.exists():
        pytest.skip(f"Test image not found: {_IMG}")
    from PIL import Image as PILImage

    from visionservex import VisionModel

    img = PILImage.open(_IMG).convert("RGB")
    model = VisionModel("sam-vit-base")
    result = model.predict(img, boxes=[[10, 20, 200, 220]])
    assert result is not None, "predict() must return a result"
    segments = getattr(result, "segments", None)
    assert segments is not None, "Result must have a 'segments' attribute"
    assert len(segments) > 0, "Result must contain at least one segment"


# ---------------------------------------------------------------------------
# 7. sam3-base must NOT return status=ok (never fake success for gated model)
# ---------------------------------------------------------------------------


def test_sam3_never_returns_fake_success():
    from visionservex.cli.sam_commands import app

    result = _runner.invoke(app, ["status", "sam3-base", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    # Must not report that the model is runnable or ok
    assert data.get("runnable") is not True, "sam3-base must not report runnable=True — it is gated"
    status = (data.get("status") or "").lower()
    assert status != "ok", f"sam3-base must never return status='ok', got: {status!r}"
    auth_required = data.get("auth_required")
    assert auth_required is True, (
        f"sam3-base must always have auth_required=True, got: {auth_required!r}"
    )

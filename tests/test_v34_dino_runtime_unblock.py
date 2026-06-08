# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.4 DINO family runtime unblock tests.

Covers:
  1. CLI app importable
  2. dino status dinov2-base --json => runnable=True
  3. dino status dinov3-vitb16 --json => auth_required=True
  4. dino status grounding-dino-1.5 --json => api_required=True
  5. dino status dino-x-api --json => api_required=True
  6. VisionModel("dinov2-base").predict returns result with embedding
  7. VisionModel("grounding-dino-swin-t").predict with text returns detections
  8. VSX().dino("dinov3-vitb16").status() returns auth_required status
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"


# ---------------------------------------------------------------------------
# 1. CLI app importable
# ---------------------------------------------------------------------------


def test_dino_cli_app_importable() -> None:
    from visionservex.cli.dino_commands import app

    assert app is not None


# ---------------------------------------------------------------------------
# 2. dino status dinov2-base --json => runnable=True
# ---------------------------------------------------------------------------


def test_dino_status_dinov2_base_passes() -> None:
    from typer.testing import CliRunner

    from visionservex.cli.dino_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["status", "dinov2-base", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["runnable"] is True
    assert data["auth_required"] is False
    assert data["api_required"] is False
    assert data["model_id"] == "dinov2-base"


# ---------------------------------------------------------------------------
# 3. dino status dinov3-vitb16 --json => auth_required=True
# ---------------------------------------------------------------------------


def test_dino_status_dinov3_returns_auth_required() -> None:
    from typer.testing import CliRunner

    from visionservex.cli.dino_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["status", "dinov3-vitb16", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["auth_required"] is True
    assert data["runnable"] is False
    assert data["model_id"] == "dinov3-vitb16"


# ---------------------------------------------------------------------------
# 4. dino status grounding-dino-1.5 --json => api_required=True
# ---------------------------------------------------------------------------


def test_dino_status_grounding_dino_1_5_api_required() -> None:
    from typer.testing import CliRunner

    from visionservex.cli.dino_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["status", "grounding-dino-1.5", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["api_required"] is True
    assert data["model_id"] == "grounding-dino-1.5"


# ---------------------------------------------------------------------------
# 5. dino status dino-x-api --json => api_required=True
# ---------------------------------------------------------------------------


def test_dino_status_dino_x_api_required() -> None:
    from typer.testing import CliRunner

    from visionservex.cli.dino_commands import app

    runner = CliRunner()
    result = runner.invoke(app, ["status", "dino-x-api", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["api_required"] is True
    assert data["model_id"] == "dino-x-api"


# ---------------------------------------------------------------------------
# 6. VisionModel("dinov2-base").predict returns result with embedding
# ---------------------------------------------------------------------------


def test_dinov2_embed_returns_array() -> None:
    if not _IMG.exists():
        pytest.skip("test image not found")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import PIL.Image

    from visionservex import VisionModel

    img = PIL.Image.open(_IMG).convert("RGB")
    model = VisionModel("dinov2-base")
    result = model.predict(img)
    assert result is not None
    # EmbeddingResult must carry an embedding attribute or embedding_dim
    has_embedding = hasattr(result, "embedding") or hasattr(result, "embedding_dim")
    assert has_embedding, (
        f"result {type(result)} has no embedding attribute; expected EmbeddingResult or similar"
    )


# ---------------------------------------------------------------------------
# 7. VisionModel("grounding-dino-swin-t").predict with text returns detections
# ---------------------------------------------------------------------------


def test_grounding_dino_detects_text() -> None:
    if not _IMG.exists():
        pytest.skip("test image not found")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import PIL.Image

    from visionservex import VisionModel

    img = PIL.Image.open(_IMG).convert("RGB")
    model = VisionModel("grounding-dino-swin-t")
    result = model.predict(img, prompts=["person"])
    assert result is not None
    # Should return a result with boxes or detections
    has_detections = (
        hasattr(result, "boxes") or hasattr(result, "detections") or hasattr(result, "predictions")
    )
    assert has_detections, (
        f"result {type(result)} has no boxes/detections attribute; "
        "expected DetectionResult or OpenVocabularyResult"
    )


# ---------------------------------------------------------------------------
# 8. VSX().dino("dinov3-vitb16").status() returns auth_required status
# ---------------------------------------------------------------------------


def test_dinov3_never_produces_embedding() -> None:
    from visionservex.vsx import VSX

    handle = VSX.dino("dinov3-vitb16")
    info = handle.explain()
    assert isinstance(info, dict), "explain() must return a dict"
    state = info.get("state", "")
    assert state in ("auth_required", "legal_review_required"), (
        f"dinov3-vitb16 must be auth_required or legal_review_required, got state={state!r}"
    )
    assert state != "benchmark_passed", (
        f"dinov3-vitb16 must never be benchmark_passed, got state={state!r}"
    )

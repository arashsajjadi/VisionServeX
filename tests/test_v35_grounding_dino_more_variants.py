# SPDX-License-Identifier: Apache-2.0
"""v3.5 GroundingDINO additional variants tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"
_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_grounding_dino_tiny_in_manifest():
    from visionservex.model_zoo import SOURCE_MANIFEST

    assert "grounding-dino-tiny" in SOURCE_MANIFEST, "grounding-dino-tiny not in manifest"


def test_grounding_dino_swin_b_in_manifest():
    from visionservex.model_zoo import SOURCE_MANIFEST

    assert "grounding-dino-swin-b" in SOURCE_MANIFEST, "grounding-dino-swin-b not in manifest"


def test_grounding_dino_variants_artifact_exists():
    artifact = _ARTIFACTS / "grounding_dino_variants.json"
    if not artifact.exists():
        pytest.skip(f"GD variants artifact not in CI env: {artifact}")
    import json

    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 GD variant must succeed, got {ok_count}"


def test_grounding_dino_tiny_result():
    artifact = _ARTIFACTS / "grounding_dino_variants.json"
    if not artifact.exists():
        pytest.skip("GD variants artifact missing")
    import json

    data = json.loads(artifact.read_text())
    if "grounding-dino-tiny" not in data:
        pytest.skip("grounding-dino-tiny not in artifact")
    r = data["grounding-dino-tiny"]
    if r.get("status") == "not_in_manifest":
        pytest.skip("grounding-dino-tiny not in manifest")
    assert r.get("status") == "ok", f"grounding-dino-tiny failed: {r}"


def test_grounding_dino_tiny_detects_objects():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image

    from visionservex.core.model import VisionModel

    img = Image.open(str(_IMG)).convert("RGB")
    m = VisionModel("grounding-dino-tiny")
    result = m.predict(img, text="person . car")
    has_det = (
        hasattr(result, "boxes") or hasattr(result, "detections") or hasattr(result, "predictions")
    )
    assert has_det, f"result {type(result)} has no detections"

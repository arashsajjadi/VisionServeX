# SPDX-License-Identifier: Apache-2.0
"""v3.5 SAM2-hiera native runtime tests."""
from __future__ import annotations
from pathlib import Path
import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"
_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_sam2_hiera_tiny_runs():
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image
    from visionservex.core.model import VisionModel
    img = Image.open(str(_IMG)).convert("RGB")
    m = VisionModel("sam2-hiera-tiny")
    result = m.predict(img, boxes=[[50, 50, 300, 300]])
    assert result is not None


def test_sam2_hiera_tiny_returns_segments():
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image
    from visionservex.core.model import VisionModel
    img = Image.open(str(_IMG)).convert("RGB")
    m = VisionModel("sam2-hiera-tiny")
    result = m.predict(img, boxes=[[50, 50, 300, 300]])
    segs = getattr(result, "segments", None) or getattr(result, "masks", None)
    assert segs is not None


def test_sam2_hiera_segmentation_artifact_exists():
    artifact = _ARTIFACTS / "sam2_hiera_segmentation.json"
    assert artifact.exists(), f"SAM2 hiera execution artifact missing: {artifact}"
    import json
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 2, f"Expected >=2 OK SAM2 executions, got {ok_count}"


def test_sam21_video_artifact_exists():
    artifact = _ARTIFACTS / "sam21_video_tracking.json"
    assert artifact.exists(), f"SAM2.1 video artifact missing"
    import json
    data = json.loads(artifact.read_text())
    assert data.get("status") == "ok", f"SAM2.1 video failed: {data}"
    assert data.get("frame1_segments", 0) >= 1


def test_sam2_hiera_4_variants_in_artifact():
    artifact = _ARTIFACTS / "sam2_hiera_segmentation.json"
    if not artifact.exists():
        pytest.skip("SAM2 hiera artifact missing")
    import json
    data = json.loads(artifact.read_text())
    expected = ["sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large"]
    for mid in expected:
        assert mid in data, f"Missing {mid} in artifact"
        assert data[mid].get("status") == "ok", f"{mid} not ok: {data[mid]}"

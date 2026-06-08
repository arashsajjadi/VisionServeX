# SPDX-License-Identifier: Apache-2.0
"""v3.4 no-sidecar regression tests.

Verifies that existing models (SAM, DINOv2, Grounding DINO) are not broken
by v3.4 additions. Each test uses the real inference path when the test image
is available, and falls back to a VSX status-only check otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"


# ---------------------------------------------------------------------------
# a. SAM-ViT-B still works
# ---------------------------------------------------------------------------


def test_existing_sam_models_still_pass() -> None:
    """VisionModel('sam-vit-b').predict(img) still works (or handles report benchmark_passed)."""
    from visionservex.vsx import _SAMHandle

    # State must remain benchmark_passed regardless of env
    info = _SAMHandle("sam-vit-b").explain()
    assert info["state"] == "benchmark_passed", f"sam-vit-b regressed: state={info['state']!r}"

    if not _IMG.exists():
        pytest.skip("test image not found; skipping live inference check")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import PIL.Image

    from visionservex import VisionModel

    img = PIL.Image.open(_IMG).convert("RGB")
    model = VisionModel("sam-vit-base")
    result = model.predict(img, boxes=[[10, 20, 200, 220]])
    assert result is not None, "VisionModel('sam-vit-base').predict returned None"


# ---------------------------------------------------------------------------
# b. DINOv2-base still returns an embedding
# ---------------------------------------------------------------------------


def test_existing_dinov2_still_passes() -> None:
    """VisionModel('dinov2-base').predict(img) returns an embedding result."""
    from visionservex.vsx import _DINOHandle

    info = _DINOHandle("dinov2-base").explain()
    assert info["state"] == "benchmark_passed", f"dinov2-base regressed: state={info['state']!r}"

    if not _IMG.exists():
        pytest.skip("test image not found; skipping live inference check")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import PIL.Image

    from visionservex import VisionModel

    img = PIL.Image.open(_IMG).convert("RGB")
    model = VisionModel("dinov2-base")
    result = model.predict(img)
    assert result is not None, "VisionModel('dinov2-base').predict returned None"
    has_embedding = hasattr(result, "embedding") or hasattr(result, "embedding_dim")
    assert has_embedding, (
        f"dinov2-base result {type(result)} has no embedding attribute; "
        "expected EmbeddingResult or similar"
    )


# ---------------------------------------------------------------------------
# c. Grounding DINO still detects with text prompt
# ---------------------------------------------------------------------------


def test_existing_grounding_dino_still_passes() -> None:
    """VisionModel('grounding-dino-swin-t').predict(img, text='person') works."""
    from visionservex.vsx import _DINOHandle

    info = _DINOHandle("grounding-dino-swin-t").explain()
    assert info["state"] == "benchmark_passed", (
        f"grounding-dino-swin-t regressed: state={info['state']!r}"
    )

    if not _IMG.exists():
        pytest.skip("test image not found; skipping live inference check")

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import PIL.Image

    from visionservex import VisionModel

    img = PIL.Image.open(_IMG).convert("RGB")
    model = VisionModel("grounding-dino-swin-t")
    result = model.predict(img, prompts=["person"])
    assert result is not None, "VisionModel('grounding-dino-swin-t').predict returned None"
    has_detections = (
        hasattr(result, "boxes") or hasattr(result, "detections") or hasattr(result, "predictions")
    )
    assert has_detections, (
        f"grounding-dino-swin-t result {type(result)} has no boxes/detections attribute"
    )

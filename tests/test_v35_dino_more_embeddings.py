# SPDX-License-Identifier: Apache-2.0
"""v3.5 DINOv2-large/giant embedding tests."""
from __future__ import annotations
from pathlib import Path
import pytest

_IMG = Path(__file__).parent / "assets" / "smoke" / "coco_person_car.jpg"
_ARTIFACTS = Path(__file__).parent.parent / "notebook/99_final_report/artifacts/v35"


def test_dinov2_large_artifact_exists():
    npy = _ARTIFACTS / "dinov2_dinov2_large_embed.npy"
    assert npy.exists(), f"DINOv2-large embedding artifact missing: {npy}"
    import numpy as np
    emb = np.load(str(npy))
    assert emb.shape[-1] == 1024, f"Expected dim=1024, got {emb.shape}"


def test_dinov2_giant_artifact_exists():
    npy = _ARTIFACTS / "dinov2_dinov2_giant_embed.npy"
    assert npy.exists(), f"DINOv2-giant embedding artifact missing: {npy}"
    import numpy as np
    emb = np.load(str(npy))
    assert emb.shape[-1] == 1536, f"Expected dim=1536, got {emb.shape}"


def test_dinov2_large_runs():
    if not _IMG.exists():
        pytest.skip("test image not found")
    from PIL import Image
    from visionservex.core.model import VisionModel
    img = Image.open(str(_IMG)).convert("RGB")
    m = VisionModel("dinov2-large")
    result = m.predict(img)
    assert result is not None
    has_embedding = hasattr(result, "embedding") or hasattr(result, "embedding_dim")
    assert has_embedding, f"result {type(result)} has no embedding attribute"


def test_dinov2_embedding_dim_progression():
    import numpy as np
    dims = {}
    for mid, fname in [("large", "dinov2_dinov2_large_embed.npy"), ("giant", "dinov2_dinov2_giant_embed.npy")]:
        npy = _ARTIFACTS / fname
        if npy.exists():
            emb = np.load(str(npy))
            dims[mid] = int(emb.shape[-1])
    if len(dims) == 2:
        assert dims["large"] == 1024
        assert dims["giant"] == 1536
        assert dims["large"] < dims["giant"]


def test_dinov2_lg_results_json_exists():
    artifact = _ARTIFACTS / "dinov2_lg_embed_results.json"
    assert artifact.exists(), "DINOv2 L/G results JSON missing"
    import json
    data = json.loads(artifact.read_text())
    ok_count = sum(1 for v in data.values() if isinstance(v, dict) and v.get("status") == "ok")
    assert ok_count >= 1, f"At least 1 DINOv2 must succeed, got {ok_count}"

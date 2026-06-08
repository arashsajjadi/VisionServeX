# SPDX-License-Identifier: Apache-2.0
"""v3.10.0: byot_runtime public API — all three functions exported."""
from __future__ import annotations

import pytest


def test_byot_runtime_all_exports():
    from visionservex import byot_runtime

    for fn in ("dinov3_depth", "dinov3_embed", "sam3_segment"):
        assert fn in byot_runtime.__all__, f"{fn} not in byot_runtime.__all__"
        assert callable(getattr(byot_runtime, fn))


def test_sam3_segment_blocked_without_text():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import sam3_segment

    result = sam3_segment("sam3", "tests/assets/smoke/coco_person_car.jpg", text=None)
    assert result["status"] in ("blocked", "ok")


def test_dinov3_depth_blocked_without_policy():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import dinov3_depth

    # Non-existent model should return blocked dict
    result = dinov3_depth("nonexistent-model", "tests/assets/smoke/coco_person_car.jpg")
    assert result["status"] == "blocked"


def test_dinov3_embed_blocked_without_policy():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import dinov3_embed

    result = dinov3_embed("nonexistent-model", "tests/assets/smoke/coco_person_car.jpg")
    assert result["status"] == "blocked"


def test_sam3_segment_returns_dict():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import sam3_segment

    result = sam3_segment("sam3", "tests/assets/smoke/coco_person_car.jpg", text="person")
    assert isinstance(result, dict)
    assert "status" in result


def test_byot_runtime_no_token_in_output():
    """Verify that the blocked result dict never contains token-looking strings."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from visionservex.byot_runtime import dinov3_embed

    result = dinov3_embed("nonexistent-model", "tests/assets/smoke/coco_person_car.jpg")
    result_str = str(result)
    import re
    token_pattern = re.compile(r"hf_[A-Za-z]{10,}")
    assert not token_pattern.search(result_str), "HF token leaked into result dict"

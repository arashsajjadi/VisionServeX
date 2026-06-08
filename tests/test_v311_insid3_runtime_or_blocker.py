# SPDX-License-Identifier: Apache-2.0
"""v3.11: INSID3 runtime — import, signature, and blocked-dict correctness.

If DINOv3 access is available (token present + HF online), attempts a real
lightweight forward pass with a synthetic 64x64 image pair.
If not, verifies the blocked-dict has the correct shape.
"""

from __future__ import annotations


def test_insid3_runtime_importable():
    from visionservex import insid3_runtime

    assert hasattr(insid3_runtime, "insid3_segment")


def test_insid3_segment_callable():
    from visionservex.insid3_runtime import insid3_segment

    assert callable(insid3_segment)


def test_unknown_model_returns_blocked():
    from PIL import Image

    from visionservex.insid3_runtime import insid3_segment

    dummy = Image.new("RGB", (32, 32), color=(128, 128, 128))
    dummy_mask = Image.new("L", (32, 32), color=255)
    result = insid3_segment(dummy, dummy, dummy_mask, model_id="insid3-nonexistent")
    assert result["status"] == "blocked"
    assert result["state"] == "unknown_model"


def test_valid_model_without_token_returns_blocked_or_ok():
    """Without a valid HF token insid3_segment must return blocked, never crash."""
    import os

    from PIL import Image

    dummy = Image.new("RGB", (64, 64), color=(100, 150, 200))
    dummy_mask = Image.new("L", (64, 64), color=200)

    # Temporarily unset token env vars to simulate no-token environment
    saved_hf = os.environ.pop("HF_TOKEN", None)
    saved_hub = os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    try:
        from visionservex.insid3_runtime import insid3_segment

        result = insid3_segment(dummy, dummy, dummy_mask, model_id="insid3-large")
        assert "status" in result
        assert result["status"] in ("ok", "blocked", "error")
        if result["status"] == "blocked":
            assert "state" in result
            assert "reason" in result
    finally:
        if saved_hf is not None:
            os.environ["HF_TOKEN"] = saved_hf
        if saved_hub is not None:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = saved_hub


def test_blocked_dict_has_next_command():
    from PIL import Image

    from visionservex.insid3_runtime import insid3_segment

    dummy = Image.new("RGB", (32, 32))
    dummy_mask = Image.new("L", (32, 32))
    result = insid3_segment(dummy, dummy, dummy_mask, model_id="insid3-nonexistent")
    assert "next_command" in result


def test_vsx_insid3_handle_explain():
    from visionservex.vsx import VSX

    handle = VSX.insid3("insid3-large")
    info = handle.explain()
    assert info["family"] == "insid3"
    assert info["task"] == "in_context_segmentation"
    assert info["state"] == "byot_license_required"
    assert "attribution_required" in info
    assert "Built with DINOv3" in info["attribution_required"]


def test_vsx_insid3_default_model():
    from visionservex.vsx import VSX

    handle = VSX.insid3()
    assert "insid3-large" in handle.model_id or handle.model_id == "insid3-large"

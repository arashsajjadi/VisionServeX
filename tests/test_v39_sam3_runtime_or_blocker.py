# SPDX-License-Identifier: Apache-2.0
"""v3.9 — SAM3/SAM3.1 BYOT runtime: real segmentation or honest blocker.

Unit tests: no live model required.
Live tests: VISIONSERVEX_RUN_GATED_HF=1 → sam3-base forward pass must succeed.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.getenv("VISIONSERVEX_RUN_GATED_HF") == "1"


def test_sam3_segment_requires_text_prompt():
    from visionservex import byot_runtime

    result = byot_runtime.sam3_segment("sam3-base", "x.jpg")
    assert result["status"] == "blocked"
    assert result["state"] in ("prompt_required", "auth_required", "auth_required_license_pending")


def test_sam3_segment_unknown_model():
    from visionservex import byot_runtime

    result = byot_runtime.sam3_segment("sam3-nonexistent-xyz", "x.jpg", text="person")
    assert result["status"] == "blocked"
    assert result["state"] == "unknown_model"


def test_sam3_segment_blocked_without_token(monkeypatch):
    from visionservex import byot_runtime, hf_auth

    monkeypatch.setattr(hf_auth, "_detect", lambda: (None, None))
    result = byot_runtime.sam3_segment("sam3-base", "x.jpg", text="person")
    assert result["status"] == "blocked"
    assert result["state"] in ("auth_required", "auth_required_license_pending", "dependency_required")


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live SAM3 tests")
def test_sam3_base_live_segmentation():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image
    from visionservex import byot_runtime

    img = Image.new("RGB", (640, 480), (100, 150, 200))
    result = byot_runtime.sam3_segment("sam3-base", img, text="car")
    assert result["status"] == "ok", f"Expected ok, got: {result}"
    assert result["state"] == "benchmark_passed_byot"
    assert "token" not in str(result.get("warning", "")).replace("hf_***", "")


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live SAM3.1 tests")
def test_sam31_base_live_segmentation():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image
    from visionservex import byot_runtime

    img = Image.new("RGB", (640, 480), (100, 150, 200))
    result = byot_runtime.sam3_segment("sam3.1-base", img, text="person")
    assert result["status"] == "ok", f"Expected ok, got: {result}"
    assert result["state"] == "benchmark_passed_byot"

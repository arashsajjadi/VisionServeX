# SPDX-License-Identifier: Apache-2.0
"""v3.9 — DINOv3 BYOT runtime: real execution or honest resource blocker.

Unit tests: verify preflight blocks without token (no live model).
Live tests: VISIONSERVEX_RUN_GATED_HF=1 → at least vits16 and convnext-tiny must pass.
Heavy live: VISIONSERVEX_RUN_HEAVY_GATED=1 → vitl16, vith16plus, vit7b16.
"""

from __future__ import annotations

import os

import pytest

LIVE = os.getenv("VISIONSERVEX_RUN_GATED_HF") == "1"
HEAVY = os.getenv("VISIONSERVEX_RUN_HEAVY_GATED") == "1"


def test_byot_runtime_importable():
    from visionservex import byot_runtime

    assert hasattr(byot_runtime, "dinov3_embed")
    assert hasattr(byot_runtime, "sam3_segment")


def test_dinov3_embed_returns_blocked_without_token(monkeypatch):
    from visionservex import byot_runtime, hf_auth

    monkeypatch.setattr(hf_auth, "_detect", lambda: (None, None))
    result = byot_runtime.dinov3_embed("dinov3-vits16", "x.jpg")
    assert result["status"] == "blocked"
    assert result["state"] in (
        "auth_required",
        "auth_required_license_pending",
        "dependency_required",
    )


def test_dinov3_unknown_model_returns_blocked():
    from visionservex import byot_runtime

    result = byot_runtime.dinov3_embed("dinov3-nonexistent-xyz", "x.jpg")
    assert result["status"] == "blocked"
    assert result["state"] == "unknown_model"


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live BYOT tests")
def test_dinov3_vits16_live_embedding():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image

    from visionservex import byot_runtime

    img = Image.new("RGB", (224, 224), (128, 64, 32))
    result = byot_runtime.dinov3_embed("dinov3-vits16", img)
    assert result["status"] == "ok", f"Expected ok, got: {result}"
    assert result["state"] == "benchmark_passed_byot"
    assert result["embedding_dim"] > 0
    assert result["embedding_norm"] > 0
    assert "token" not in str(result.get("token_redacted", "")).replace("hf_***", "")


@pytest.mark.skipif(not LIVE, reason="Set VISIONSERVEX_RUN_GATED_HF=1 for live BYOT tests")
def test_dinov3_convnext_tiny_live_embedding():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image

    from visionservex import byot_runtime

    img = Image.new("RGB", (224, 224), (200, 100, 50))
    result = byot_runtime.dinov3_embed("dinov3-convnext-tiny", img)
    assert result["status"] == "ok"
    assert result["embedding_dim"] > 0


@pytest.mark.skipif(not HEAVY, reason="Set VISIONSERVEX_RUN_HEAVY_GATED=1 for heavy BYOT tests")
def test_dinov3_vitl16_heavy_or_resource_blocked():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from PIL import Image

    from visionservex import byot_runtime

    img = Image.new("RGB", (224, 224))
    result = byot_runtime.dinov3_embed("dinov3-vitl16", img)
    assert result["status"] in ("ok", "blocked"), f"Unexpected: {result}"
    if result["status"] == "blocked":
        assert result["state"] in ("resource_blocked_byot", "dependency_required")

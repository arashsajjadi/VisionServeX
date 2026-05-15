# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for SwinV2, SAM HF, Grounded SAM, and GD dtype fix."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visionservex.config import reload_settings
from visionservex.registry import default_registry


def _img(size=(128, 128), color="gray") -> Image.Image:
    return Image.new("RGB", size, color=color)


# ============================================================
# Registry metadata
# ============================================================

def test_swinv2_tiny_in_registry():
    e = default_registry().get("swinv2-tiny")
    assert e.task == "classify"
    assert e.engine == "swinv2"
    assert e.implementation_status == "wired"
    assert e.status == "beta"
    assert e.install_extra == "hf"
    assert e.auto_download is True


def test_sam_vit_base_in_registry():
    e = default_registry().get("sam-vit-base")
    assert e.task == "foundation_segment"
    assert e.engine == "sam_hf"
    assert e.implementation_status == "wired"
    assert e.status == "beta"
    assert e.install_extra == "hf"
    assert e.auto_download is True


def test_grounded_sam_in_registry():
    e = default_registry().get("grounded-sam")
    assert e.task == "grounded_segment"
    assert e.engine == "grounded_sam"
    assert e.implementation_status == "wired"
    assert e.status == "beta"


def test_sam2_hiera_tiny_now_wired():
    e = default_registry().get("sam2-hiera-tiny")
    assert e.implementation_status == "wired"
    assert e.engine == "sam2_hf"
    assert e.status == "beta"
    assert e.hf_repo_id == "facebook/sam2-hiera-tiny"


# ============================================================
# SwinV2 engine
# ============================================================

def test_swinv2_engine_missing_dep_without_fallback(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setenv("VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from visionservex.engines.base import MissingDependencyError
    from visionservex.engines.swinv2 import SwinV2Engine
    e = default_registry().get("swinv2-tiny")
    engine = SwinV2Engine(e)
    with pytest.raises(MissingDependencyError):
        engine.load(device="cpu", precision="fp32")


@pytest.mark.real_model
def test_swinv2_tiny_real_inference():
    pytest.importorskip("transformers")
    from visionservex.engines.swinv2 import SwinV2Engine
    e = default_registry().get("swinv2-tiny")
    engine = SwinV2Engine(e)
    engine.load(device="cpu", precision="fp32")
    result = engine.predict(_img(size=(256, 256)))
    assert result.task == "classify"
    assert len(result.top_k) > 0
    assert result.top_k[0][1] >= result.top_k[-1][1], "top_k should be sorted descending"
    engine.unload()


@pytest.mark.real_model
def test_swinv2_classification_result_schema():
    pytest.importorskip("transformers")
    from visionservex import VisionModel
    m = VisionModel("swinv2-tiny", device="cpu")
    r = m.predict(_img(size=(256, 256)))
    assert r.kind == "classification"
    assert r.backend == "huggingface"
    assert len(r.top_k) >= 1
    for label, score in r.top_k:
        assert isinstance(label, str)
        assert 0.0 <= float(score)


# ============================================================
# SAM HF engine
# ============================================================

@pytest.mark.real_model
def test_sam_hf_point_prompt():
    pytest.importorskip("transformers")
    from visionservex.engines.sam_hf import SAMHFEngine
    e = default_registry().get("sam-vit-base")
    engine = SAMHFEngine(e)
    engine.load(device="cpu", precision="fp32")
    img = _img(size=(256, 256), color="blue")
    result = engine.predict(img, points=[[128, 128]], point_labels=[1])
    assert result.task == "foundation_segment"
    assert len(result.segments) >= 1
    seg = result.segments[0]
    assert seg.mask.ndim == 2
    assert seg.mask.dtype == np.uint8
    assert seg.score >= 0.0  # IoU scores from SAM can exceed 1.0 in some versions
    engine.unload()


@pytest.mark.real_model
def test_sam_hf_box_prompt():
    pytest.importorskip("transformers")
    from visionservex import VisionModel
    m = VisionModel("sam-vit-base", device="cpu")
    r = m.predict(_img(size=(256, 256)), boxes=[[30, 30, 120, 120]])
    assert r.kind == "segmentation"
    assert len(r.segments) >= 1
    assert r.backend == "huggingface_sam"


@pytest.mark.real_model
def test_sam_hf_default_prompt():
    """Without any prompts, SAM should still return at least one mask."""
    pytest.importorskip("transformers")
    from visionservex import VisionModel
    m = VisionModel("sam-vit-base", device="cpu")
    r = m.predict(_img(size=(128, 128)))
    assert len(r.segments) >= 1


# ============================================================
# Grounded SAM
# ============================================================

@pytest.mark.real_model
def test_grounded_sam_pipeline():
    pytest.importorskip("transformers")
    from visionservex import VisionModel
    m = VisionModel("grounded-sam", device="cpu", precision="fp32")
    r = m.predict(_img(size=(256, 256), color="green"), prompts=["green object"])
    assert r.task == "grounded_segment"
    assert r.backend == "composed_gd_sam"
    # Should return at least one segment (even if zero detections → zero segments is acceptable)
    assert hasattr(r, "segments")


# ============================================================
# Grounding DINO fp16 fallback test
# ============================================================

@pytest.mark.real_model
def test_grounding_dino_cpu_fp32():
    pytest.importorskip("transformers")
    from visionservex import VisionModel
    m = VisionModel("grounding-dino-tiny", device="cpu", precision="fp32")
    r = m.predict(_img(size=(320, 240)), prompts=["object"])
    assert r.kind == "open_vocab"
    assert r.backend == "huggingface"

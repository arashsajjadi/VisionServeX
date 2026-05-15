# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for D-FINE, SAM2 (HF), and OneFormer backends (Phase H)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from visionservex.registry import default_registry


def _img(size=(128, 128), color="gray") -> Image.Image:
    return Image.new("RGB", size, color=color)


# ============================================================
# Registry metadata — D-FINE
# ============================================================


def test_dfine_n_registry():
    e = default_registry().get("dfine-n")
    assert e.task == "detect"
    assert e.engine == "dfine"
    assert e.implementation_status == "wired"
    assert e.status == "beta"
    assert e.hf_repo_id == "ustc-community/dfine-nano-coco"
    assert e.auto_download is True
    assert e.install_extra == "hf"


def test_dfine_s_registry():
    e = default_registry().get("dfine-s")
    assert e.hf_repo_id == "ustc-community/dfine-small-obj2coco"
    assert e.implementation_status == "wired"


def test_dfine_family_all_wired():
    for mid in ("dfine-n", "dfine-s", "dfine-m", "dfine-l", "dfine-x"):
        e = default_registry().get(mid)
        assert e.implementation_status == "wired", f"{mid} should be wired"
        assert e.engine == "dfine"
        assert e.hf_repo_id is not None


# ============================================================
# Registry metadata — SAM2 HF
# ============================================================


def test_sam2_hiera_tiny_registry():
    e = default_registry().get("sam2-hiera-tiny")
    assert e.task == "foundation_segment"
    assert e.engine == "sam2_hf"
    assert e.implementation_status == "wired"
    assert e.hf_repo_id == "facebook/sam2-hiera-tiny"
    assert e.install_extra == "hf"


def test_sam2_all_wired():
    for mid in ("sam2-hiera-tiny", "sam2-hiera-small", "sam2-hiera-base-plus", "sam2-hiera-large"):
        e = default_registry().get(mid)
        assert e.implementation_status == "wired", f"{mid} should be wired"
        assert e.engine == "sam2_hf"


# ============================================================
# Registry metadata — OneFormer
# ============================================================


def test_oneformer_swin_registry():
    e = default_registry().get("oneformer-swin-large")
    assert e.task == "segment"
    assert e.engine == "oneformer"
    assert e.implementation_status == "wired"
    assert e.hf_repo_id == "shi-labs/oneformer_coco_swin_large"
    assert e.install_extra == "hf"


def test_oneformer_all_wired():
    for mid in ("oneformer-swin-large", "oneformer-dinat-large", "oneformer-convnext-large"):
        e = default_registry().get(mid)
        assert e.implementation_status == "wired", f"{mid} should be wired"
        assert e.engine == "oneformer"


# ============================================================
# D-FINE engine real inference
# ============================================================


@pytest.mark.real_model
def test_dfine_n_real_inference():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("dfine-n", device="cpu")
    r = m.predict(_img(size=(640, 480)))
    assert r.kind == "detection"
    assert r.task == "detect"
    assert r.backend == "huggingface_dfine"
    assert r.model_id == "dfine-n"
    # Score and label types
    for det in r.detections:
        assert isinstance(det.label, str)
        assert 0.0 <= det.score <= 1.0
    m.unload()


@pytest.mark.real_model
def test_dfine_s_real_inference():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("dfine-s", device="cpu")
    r = m.predict(_img(size=(640, 480)))
    assert r.kind == "detection"
    assert r.backend == "huggingface_dfine"
    m.unload()


@pytest.mark.real_model
def test_dfine_result_schema():
    pytest.importorskip("transformers")
    import json

    from visionservex import VisionModel

    m = VisionModel("dfine-n", device="cpu")
    r = m.predict(_img(size=(320, 240)))
    d = r.to_dict()
    assert "detections" in d
    assert "kind" in d and d["kind"] == "detection"
    assert "backend" in d
    # JSON-serialisable
    json.dumps(d, default=str)
    m.unload()


# ============================================================
# SAM2 HF engine real inference
# ============================================================


@pytest.mark.real_model
def test_sam2_hiera_tiny_point_prompt():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("sam2-hiera-tiny", device="cpu")
    r = m.predict(_img(size=(256, 256)), points=[[128, 128]], point_labels=[1])
    assert r.kind == "segmentation"
    assert r.backend == "huggingface_sam2"
    assert len(r.segments) >= 1
    seg = r.segments[0]
    assert seg.mask.ndim == 2
    assert seg.mask.dtype == np.uint8
    assert seg.score >= 0.0
    m.unload()


@pytest.mark.real_model
def test_sam2_hiera_tiny_box_prompt():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("sam2-hiera-tiny", device="cpu")
    r = m.predict(_img(size=(256, 256)), boxes=[[30, 30, 220, 220]])
    assert r.kind == "segmentation"
    assert len(r.segments) >= 1
    m.unload()


@pytest.mark.real_model
def test_sam2_hiera_default_prompt():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("sam2-hiera-tiny", device="cpu")
    r = m.predict(_img(size=(128, 128)))
    assert len(r.segments) >= 1
    m.unload()


# ============================================================
# OneFormer engine real inference
# ============================================================


@pytest.mark.real_model
def test_oneformer_swin_semantic():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("oneformer-swin-large", device="cpu")
    r = m.predict(_img(size=(256, 256), color="blue"), task="semantic")
    assert r.kind == "segmentation"
    assert r.backend == "huggingface_oneformer"
    assert len(r.segments) >= 1
    for seg in r.segments:
        assert isinstance(seg.label, str)
        assert seg.mask.ndim == 2
    m.unload()


@pytest.mark.real_model
def test_oneformer_swin_panoptic():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("oneformer-swin-large", device="cpu")
    r = m.predict(_img(size=(256, 256)), task="panoptic")
    assert r.kind == "segmentation"
    assert r.metadata.get("oneformer_task") == "panoptic"
    m.unload()


@pytest.mark.real_model
def test_oneformer_metadata_task_field():
    pytest.importorskip("transformers")
    from visionservex import VisionModel

    m = VisionModel("oneformer-swin-large", device="cpu")
    r = m.predict(_img(size=(128, 128)), task="semantic")
    assert r.metadata["oneformer_task"] == "semantic"
    m.unload()

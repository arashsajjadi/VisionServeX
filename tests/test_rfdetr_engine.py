# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for RF-DETR engine and related backends."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from visionservex.engines.rfdetr import RFDETREngine, _sv_to_detections, _sv_to_segments
from visionservex.registry import default_registry
from visionservex.config import reload_settings


def _img(size=(64, 64), color="red") -> Image.Image:
    return Image.new("RGB", size, color=color)


# ---- registry metadata ----

def test_rfdetr_nano_in_registry():
    reg = default_registry()
    e = reg.get("rfdetr-nano")
    assert e.task == "detect"
    assert e.family == "rfdetr"
    assert e.engine == "rfdetr"
    assert e.download_type == "package_managed"
    assert e.implementation_status == "wired"
    assert e.status == "beta"
    assert e.auto_download is True


def test_rfdetr_seg_nano_in_registry():
    reg = default_registry()
    e = reg.get("rfdetr-seg-nano")
    assert e.task == "segment"
    assert e.implementation_status == "wired"


def test_rfdetr_models_have_required_fields():
    reg = default_registry()
    for model_id in ("rfdetr-nano", "rfdetr-small", "rfdetr-base", "rfdetr-seg-nano", "rfdetr-seg-small"):
        e = reg.get(model_id)
        assert e.license == "Apache-2.0"
        assert e.upstream_url
        assert e.install_extra == "rfdetr"
        assert e.requires_optional_extra is True


# ---- sv helper functions ----

class _FakeDetections:
    """Minimal simulation of supervision.Detections."""
    def __init__(self, xyxy, conf, class_ids, class_names=None, mask=None):
        self.xyxy = np.array(xyxy)
        self.confidence = np.array(conf)
        self.class_id = np.array(class_ids)
        self.mask = np.array(mask) if mask is not None else None
        self.data = {"class_name": np.array(class_names, dtype=object)} if class_names else {}
        self.metadata = {}

    def __len__(self):
        return len(self.xyxy)


def test_sv_to_detections_basic():
    fake = _FakeDetections(
        xyxy=[[10, 20, 100, 200]],
        conf=[0.9],
        class_ids=[0],
        class_names=["person"],
    )
    dets = _sv_to_detections(fake, ["person", "car"])
    assert len(dets) == 1
    assert dets[0].label == "person"
    assert abs(dets[0].score - 0.9) < 0.001
    assert dets[0].box.x1 == 10.0
    assert dets[0].box.y2 == 200.0


def test_sv_to_detections_empty():
    fake = _FakeDetections(xyxy=[], conf=[], class_ids=[])
    dets = _sv_to_detections(fake, [])
    assert dets == []


def test_sv_to_segments_with_mask():
    h, w = 64, 64
    mask = np.zeros((h, w), dtype=bool)
    mask[10:40, 10:40] = True
    fake = _FakeDetections(
        xyxy=[[10, 10, 40, 40]],
        conf=[0.8],
        class_ids=[1],
        class_names=["car"],
        mask=[mask],
    )
    segs = _sv_to_segments(fake, ["person", "car"])
    assert len(segs) == 1
    assert segs[0].label == "car"
    assert segs[0].mask.shape == (h, w)
    assert segs[0].mask.dtype == np.uint8


def test_sv_to_segments_no_mask():
    fake = _FakeDetections(
        xyxy=[[0, 0, 10, 10]],
        conf=[0.7],
        class_ids=[0],
    )
    segs = _sv_to_segments(fake, ["person"])
    assert len(segs) == 1


# ---- stub behavior (without rfdetr installed) ----

def test_rfdetr_engine_missing_dep_without_mock_fallback(monkeypatch, tmp_path):
    """Without rfdetr installed, the engine must raise MissingDependencyError."""
    import sys
    monkeypatch.setitem(sys.modules, "rfdetr", None)  # block import
    monkeypatch.setenv("VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK", "false")
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    from visionservex.engines.base import MissingDependencyError
    e = default_registry().get("rfdetr-nano")
    engine = RFDETREngine(e)
    with pytest.raises(MissingDependencyError):
        engine.load(device="cpu", precision="fp32")


def test_rfdetr_engine_mock_fallback_when_allowed(monkeypatch, tmp_path):
    """With mock_fallback=true, a missing rfdetr gives mock output with warning."""
    import sys
    monkeypatch.setitem(sys.modules, "rfdetr", None)  # block import
    monkeypatch.setenv("VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK", "true")
    monkeypatch.setenv("VISIONSERVEX_CACHE__CACHE_DIR", str(tmp_path))
    reload_settings()
    e = default_registry().get("rfdetr-nano")
    engine = RFDETREngine(e)
    engine.load(device="cpu", precision="fp32")
    assert engine._loaded
    result = engine.predict(_img())
    # Fallback: result should be a detection result (mock output) or have warning
    assert result.kind in {"detection", "classification", "segmentation", "pose", "obb", "open_vocab"}


# ---- real model smoke test (requires rfdetr installed) ----

@pytest.mark.real_model
def test_rfdetr_nano_real_inference():
    """Requires rfdetr package with downloaded weights."""
    pytest.importorskip("rfdetr")
    e = default_registry().get("rfdetr-nano")
    engine = RFDETREngine(e)
    engine.load(device="cpu", precision="fp32")
    result = engine.predict(_img(size=(384, 384)))
    assert result.task == "detect"
    assert result.backend == "rfdetr_package"
    engine.unload()


@pytest.mark.real_model
def test_rfdetr_seg_nano_real_inference():
    """Requires rfdetr package with downloaded seg weights."""
    pytest.importorskip("rfdetr")
    e = default_registry().get("rfdetr-seg-nano")
    engine = RFDETREngine(e)
    engine.load(device="cpu", precision="fp32")
    result = engine.predict(_img(size=(312, 312)))
    assert result.task == "segment"
    assert result.backend == "rfdetr_package"
    engine.unload()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Tests for v1.5.0: VRAM lifecycle manager, GPU CLI commands, mask AP evaluator,
process-isolated benchmark, benchmark-segmentation command."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from visionservex.cli.main import app

runner = CliRunner()


# ============================================================
# Phase 1 — GPU lifecycle manager
# ============================================================


def test_get_gpu_memory_state_no_crash():
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state

    state = get_gpu_memory_state("test")
    assert state.label == "test"
    assert isinstance(state.allocated_mb, float)
    assert isinstance(state.reserved_mb, float)
    assert state.allocated_mb >= 0.0
    assert state.reserved_mb >= 0.0


def test_get_gpu_memory_state_to_dict():
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state

    state = get_gpu_memory_state("snapshot")
    d = state.to_dict()
    assert "label" in d
    assert "allocated_mb" in d
    assert "reserved_mb" in d
    assert "cuda_available" in d


def test_memory_state_growth():
    from visionservex.runtime.gpu_lifecycle import MemoryState

    a = MemoryState(label="a", allocated_mb=100.0, reserved_mb=200.0)
    b = MemoryState(label="b", allocated_mb=150.0, reserved_mb=300.0)
    assert abs(b.growth_vs(a) - 50.0) < 1e-5
    assert abs(b.reserved_growth_vs(a) - 100.0) < 1e-5


def test_assert_memory_returned_to_baseline_ok():
    from visionservex.runtime.gpu_lifecycle import MemoryState, assert_memory_returned_to_baseline

    baseline = MemoryState(label="b", allocated_mb=100.0)
    current = MemoryState(label="c", allocated_mb=110.0)
    result = assert_memory_returned_to_baseline(baseline, current, max_growth_mb=256.0)
    assert result["status"] == "ok"
    assert result["allocated_growth_mb"] == pytest.approx(10.0)


def test_assert_memory_returned_to_baseline_warning():
    from visionservex.runtime.gpu_lifecycle import MemoryState, assert_memory_returned_to_baseline

    baseline = MemoryState(label="b", allocated_mb=100.0)
    current = MemoryState(label="c", allocated_mb=800.0)
    result = assert_memory_returned_to_baseline(baseline, current, max_growth_mb=256.0)
    assert result["status"] == "warning"
    assert "VRAM" in result["message"]


def test_clear_torch_cuda_cache_no_crash():
    from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache

    # Must not raise even without CUDA
    clear_torch_cuda_cache()


def test_force_gc_no_crash():
    from visionservex.runtime.gpu_lifecycle import force_gc

    force_gc()


def test_cleanup_gpu_after_model_no_crash():
    from visionservex.runtime.gpu_lifecycle import cleanup_gpu_after_model

    cleanup_gpu_after_model(None)  # model=None is valid


def test_cleanup_gpu_after_model_returns_state():
    from visionservex.runtime.gpu_lifecycle import MemoryState, cleanup_gpu_after_model

    state = cleanup_gpu_after_model(None)
    assert isinstance(state, MemoryState)


def test_get_process_gpu_memory_dict():
    from visionservex.runtime.gpu_lifecycle import get_process_gpu_memory

    result = get_process_gpu_memory()
    assert "available" in result


def test_recommend_process_restart_low_growth():
    from visionservex.runtime.gpu_lifecycle import MemoryState, recommend_process_restart_if_needed

    baseline = MemoryState(label="b", allocated_mb=100.0)
    current = MemoryState(label="c", allocated_mb=200.0)
    msg = recommend_process_restart_if_needed(baseline, current, threshold_mb=2048.0)
    assert msg is None


def test_recommend_process_restart_high_growth():
    from visionservex.runtime.gpu_lifecycle import MemoryState, recommend_process_restart_if_needed

    baseline = MemoryState(label="b", allocated_mb=100.0)
    current = MemoryState(label="c", allocated_mb=5000.0)
    msg = recommend_process_restart_if_needed(baseline, current, threshold_mb=1024.0)
    assert msg is not None
    assert "VRAM" in msg or "process" in msg.lower()


# ============================================================
# VisionModel GPU-safe unload
# ============================================================


def test_visionmodel_unload_does_not_crash():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    model._ensure_loaded()
    model.unload()
    assert not model.loaded


def test_visionmodel_close_alias():
    from visionservex import VisionModel

    model = VisionModel("mock-detect")
    model._ensure_loaded()
    model.close()
    assert not model.loaded


def test_visionmodel_context_manager_cleanup():
    from PIL import Image

    from visionservex import VisionModel

    img = Image.new("RGB", (320, 240))
    with VisionModel("mock-detect") as model:
        model.predict(img)
    assert not model.loaded


def test_visionmodel_predict_unload_after():
    from PIL import Image

    from visionservex import VisionModel

    img = Image.new("RGB", (320, 240))
    model = VisionModel("mock-detect")
    model.predict(img, unload_after=True)
    assert not model.loaded


# ============================================================
# GPU CLI commands
# ============================================================


def test_gpu_cleanup_cache_json():
    result = runner.invoke(app, ["gpu", "cleanup-cache", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "freed_reserved_mb" in data
    assert "cuda_available" in data


def test_gpu_explain_memory_json():
    result = runner.invoke(app, ["gpu", "explain-memory", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "cuda_available" in data
    assert "allocated_mb" in data


def test_gpu_memory_test_json():
    result = runner.invoke(app, ["gpu", "memory-test", "mock-detect", "--runs", "2", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "model_id" in data
    assert "status" in data
    assert "total_growth_mb" in data


def test_gpu_memory_test_suite_json():
    result = runner.invoke(
        app, ["gpu", "memory-test-suite", "--models", "mock-detect,mock-classify", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "results" in data
    assert data["models_tested"] == 2


def test_gpu_unload_all_json():
    result = runner.invoke(app, ["gpu", "unload-all", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "freed_reserved_mb" in data


def test_gpu_memory_test_human():
    result = runner.invoke(app, ["gpu", "memory-test", "mock-detect", "--runs", "2"])
    assert result.exit_code == 0


# ============================================================
# Phase 6 — Mask AP evaluator
# ============================================================


def test_mask_iou_perfect():
    from visionservex.runtime.segmentation_eval import _mask_iou

    mask = np.ones((10, 10), dtype=np.uint8)
    iou = _mask_iou(mask, mask)
    assert abs(iou - 1.0) < 1e-5


def test_mask_iou_no_overlap():
    from visionservex.runtime.segmentation_eval import _mask_iou

    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[:5, :5] = 1
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m2[5:, 5:] = 1
    iou = _mask_iou(m1, m2)
    assert abs(iou) < 1e-5


def test_mask_iou_partial():
    from visionservex.runtime.segmentation_eval import _mask_iou

    m1 = np.zeros((10, 10), dtype=np.uint8)
    m1[:6, :6] = 1
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m2[4:, 4:] = 1
    iou = _mask_iou(m1, m2)
    assert 0.0 < iou < 1.0


def test_mask_evaluator_perfect_match():
    from visionservex.runtime.segmentation_eval import MaskDetectionEvaluator

    mask = np.ones((10, 10), dtype=np.uint8)
    ev = MaskDetectionEvaluator()
    ev.add_image(
        pred_masks=[mask],
        pred_boxes=[[0, 0, 10, 10]],
        pred_scores=[0.9],
        pred_classes=["cat"],
        gt_masks=[mask],
        gt_boxes=[[0, 0, 10, 10]],
        gt_classes=["cat"],
    )
    metrics = ev.compute_metrics(iou_threshold=0.50)
    assert metrics["mask_ap50"] > 0.95


def test_mask_evaluator_no_predictions():
    from visionservex.runtime.segmentation_eval import MaskDetectionEvaluator

    ev = MaskDetectionEvaluator()
    mask = np.ones((10, 10), dtype=np.uint8)
    ev.add_image([], [], [], [], [mask], [[0, 0, 10, 10]], ["cat"])
    metrics = ev.compute_metrics()
    assert metrics["mask_ap50"] == 0.0


def test_mask_evaluator_map50_95():
    from visionservex.runtime.segmentation_eval import MaskDetectionEvaluator

    mask = np.ones((10, 10), dtype=np.uint8)
    ev = MaskDetectionEvaluator()
    ev.add_image([mask], [[0, 0, 10, 10]], [0.9], ["cat"], [mask], [[0, 0, 10, 10]], ["cat"])
    result = ev.compute_map50_95()
    assert "mask_map50_95" in result
    assert result["mask_map50_95"] <= result["mask_ap50"] + 0.01


def test_polygon_to_mask():
    from visionservex.runtime.segmentation_eval import _polygon_to_mask

    # A rectangle polygon
    polygon = [10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0]
    mask = _polygon_to_mask(polygon, height=100, width=100)
    assert mask.shape == (100, 100)
    assert mask[30, 30] == 1  # center should be inside
    assert mask[0, 0] == 0  # corner outside


def test_load_coco_segmentation_empty_dir(tmp_path):
    import json

    from visionservex.runtime.segmentation_eval import load_coco_segmentation_json

    ann = {
        "images": [],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(ann))
    samples, class_names = load_coco_segmentation_json(tmp_path, ann_file)
    assert samples == []
    assert class_names == ["cat"]


def test_load_coco_segmentation_one_image(tmp_path):
    import json

    from PIL import Image

    from visionservex.runtime.segmentation_eval import load_coco_segmentation_json

    img = Image.new("RGB", (100, 100))
    img.save(str(tmp_path / "img1.jpg"))

    ann = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 40, 40],
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            }
        ],
    }
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(ann))

    samples, _class_names = load_coco_segmentation_json(tmp_path, ann_file)
    assert len(samples) == 1
    assert samples[0].gt_classes == ["person"]
    assert len(samples[0].gt_masks) == 1
    assert samples[0].gt_masks[0] is not None
    assert samples[0].gt_masks[0].shape == (100, 100)


# ============================================================
# Benchmark-segmentation CLI
# ============================================================


def test_benchmark_segmentation_synthetic_json():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-segmentation",
            "--models",
            "mock-segment",
            "--max-images",
            "2",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "benchmark_type" in data
    assert "synthetic" in data["benchmark_type"]


def test_benchmark_segmentation_invalid_model_skipped():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-segmentation",
            "--models",
            "mock-detect",
            "--max-images",
            "2",
            "--json",
        ],
    )
    # mock-detect is not a segment model → should be skipped, exit non-zero (no valid models)
    assert result.exit_code != 0


def test_benchmark_segmentation_coco_json_missing_path():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-segmentation",
            "--models",
            "mock-segment",
            "--dataset",
            "coco-json:/nonexistent/images:/nonexistent/ann.json",
            "--max-images",
            "2",
        ],
    )
    assert result.exit_code != 0


def test_benchmark_segmentation_coco_json_with_data(tmp_path):
    import json

    from PIL import Image

    img = Image.new("RGB", (100, 100))
    img.save(str(tmp_path / "img1.jpg"))
    ann = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "thing"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 40, 40],
                "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
            }
        ],
    }
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(ann))

    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-segmentation",
            "--models",
            "mock-segment",
            "--dataset",
            f"coco-json:{tmp_path}:{ann_file}",
            "--max-images",
            "1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "mask_ap" in data["benchmark_type"]
    assert "models" in data
    assert len(data["models"]) == 1


# ============================================================
# Benchmark-competitiveness with unload
# ============================================================


def test_benchmark_competitiveness_unload_json():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--max-images",
            "2",
            "--unload-between-models",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "models" in data


def test_benchmark_competitiveness_no_unload_json():
    result = runner.invoke(
        app,
        [
            "benchmark",
            "benchmark-competitiveness",
            "--models",
            "mock-detect",
            "--max-images",
            "2",
            "--no-unload",
            "--json",
        ],
    )
    assert result.exit_code == 0


# ============================================================
# Version check
# ============================================================


def test_version_is_at_least_150():
    from visionservex import __version__

    major, minor, _ = (int(x) for x in __version__.split("."))
    assert (major, minor) >= (1, 5), f"Expected at least 1.5.x, got {__version__}"


def test_version_at_least_140():
    from visionservex import __version__

    major, minor, _ = (int(x) for x in __version__.split("."))
    assert (major, minor) >= (1, 4)

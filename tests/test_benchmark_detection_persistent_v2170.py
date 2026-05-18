# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Phase 2/3 (v2.17.0): persistent benchmark + GPU enforcement + GPU sampler.

These tests use ``mock-detect`` so they run on CPU without downloading
weights. They prove the v2.17 schema contract:

- load_count == 1 for N images
- timing breakdown (preprocess / inference / postprocess / evaluation /
  total p50/p95) present in JSON
- --require-gpu fails with GPU_REQUIRED_NOT_USED if device falls back
- --sample-gpu populates the gpu_utilization block (or records a
  graceful warning when nvidia-smi is missing)
- evaluation_scope follows the diagnostic_6 / full_N rules
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from visionservex.runtime.evaluation import DatasetSample
from visionservex.runtime.persistent_benchmark import (
    GpuUtilizationSampler,
    build_v217_row,
    run_persistent_detection_benchmark,
)


def _make_yolo_fixture(tmp_path: Path, *, n_images: int = 5) -> Path:
    """Build a tiny YOLO-format dataset under tmp_path/dataset/."""
    root = tmp_path / "dataset"
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        img = Image.new("RGB", (320, 240), (50 + i * 10, 80, 200))
        img.save(images / f"img_{i:03d}.jpg")
        # One GT box per image, in YOLO normalized [class cx cy w h]
        (labels / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (root / "data.yaml").write_text("names:\n  0: person\n")
    return root


# ---------------------------------------------------------------------------
# build_v217_row schema sanity
# ---------------------------------------------------------------------------


def test_build_v217_row_full_scope_evaluation_scope() -> None:
    row = build_v217_row(
        model_id="mock-detect",
        canonical_model_id="mock-detect",
        is_alias=False,
        n_images_requested=20,
        n_images_evaluated=20,
        device_requested="cuda",
        device_actual="cuda",
        gpu_name="NVIDIA GeForce RTX 5080",
        gpu_profile="desktop_16gb_fast",
        load_count=1,
        load_time_ms=120.0,
        preprocess_ms=[1.0] * 20,
        inference_ms=[5.0] * 20,
        postprocess_ms=[0.5] * 20,
        evaluation_ms=[0.2] * 20,
        total_latency_ms=[6.7] * 20,
        n_raw_predictions=400,
        n_normalized_predictions=400,
        n_invalid_predictions=0,
        n_dropped_predictions=0,
        no_detection_image_count=0,
        ap50=0.55,
        ap75=0.40,
        map50_95=0.35,
        class_agnostic_ap50=0.60,
        precision50=0.6,
        recall50=0.5,
        f1_50=0.55,
        gpu_utilization=None,
        warnings_=[],
        errors=[],
        status="ok",
        code="OK",
    )
    assert row["evaluation_scope"] == "full_20"
    assert row["load_count"] == 1
    assert row["total_latency_ms_p50"] == 6.7
    assert row["images_per_second"] > 0
    assert row["canonical_model_id"] == "mock-detect"


def test_build_v217_row_diagnostic_6_when_partial() -> None:
    row = build_v217_row(
        model_id="dfine-s-o365-coco",
        canonical_model_id="dfine-s-o365-coco",
        is_alias=False,
        n_images_requested=100,
        n_images_evaluated=6,
        device_requested="cuda",
        device_actual="cuda",
        gpu_name="x",
        gpu_profile="desktop_16gb_fast",
        load_count=1,
        load_time_ms=300,
        preprocess_ms=[],
        inference_ms=[],
        postprocess_ms=[],
        evaluation_ms=[],
        total_latency_ms=[],
        n_raw_predictions=0,
        n_normalized_predictions=0,
        n_invalid_predictions=0,
        n_dropped_predictions=0,
        no_detection_image_count=6,
        ap50=0.0,
        ap75=0.0,
        map50_95=0.0,
        class_agnostic_ap50=0.0,
        precision50=0.0,
        recall50=0.0,
        f1_50=0.0,
        gpu_utilization=None,
        warnings_=[],
        errors=[],
        status="ok",
        code="OK",
    )
    assert row["evaluation_scope"] == "diagnostic_6"


# ---------------------------------------------------------------------------
# GPU sampler
# ---------------------------------------------------------------------------


def test_gpu_sampler_never_raises_when_nvidia_smi_missing(monkeypatch) -> None:
    """If nvidia-smi is not on PATH, sampler must skip with a warning."""
    import visionservex.runtime.persistent_benchmark as pb_mod

    monkeypatch.setattr(pb_mod.shutil, "which", lambda _x: None)
    sampler = GpuUtilizationSampler(interval=0.05)
    sampler.start()
    sampler.stop()
    s = sampler.summary()
    assert s["samples"] == 0
    assert s["utilization_mean"] == 0.0
    assert any("nvidia-smi" in w for w in s["warnings"])


def test_gpu_sampler_summary_fields_present() -> None:
    sampler = GpuUtilizationSampler(interval=0.1)
    sampler._samples.append(
        type(
            "S",
            (),
            {
                "timestamp": 0,
                "gpu_utilization_pct": 30.0,
                "vram_used_mb": 500,
                "vram_total_mb": 16000,
            },
        )()
    )
    sampler._samples.append(
        type(
            "S",
            (),
            {
                "timestamp": 0.1,
                "gpu_utilization_pct": 80.0,
                "vram_used_mb": 2000,
                "vram_total_mb": 16000,
            },
        )()
    )
    s = sampler.summary()
    assert s["samples"] == 2
    assert s["utilization_mean"] == 55.0
    assert s["vram_used_peak_gb"] == round(2000 / 1024.0, 3)


# ---------------------------------------------------------------------------
# Persistent benchmark with mock-detect (CPU)
# ---------------------------------------------------------------------------


def test_persistent_benchmark_mock_detect_load_count_one(tmp_path: Path) -> None:
    """mock-detect must load exactly once for N images."""
    root = _make_yolo_fixture(tmp_path, n_images=5)
    samples: list[DatasetSample] = []
    for i in range(5):
        samples.append(
            DatasetSample(
                image_path=str(root / "images" / f"img_{i:03d}.jpg"),
                gt_boxes=[[64.0, 48.0, 256.0, 192.0]],
                gt_classes=["person"],
            )
        )

    row = run_persistent_detection_benchmark(
        model_id="mock-detect",
        samples=samples,
        device_requested="cpu",
        require_gpu=False,
        sample_gpu=False,
    )
    assert row["status"] == "ok", row
    assert row["load_count"] == 1, row
    assert row["n_images_requested"] == 5
    assert row["n_images_evaluated"] == 5
    assert row["evaluation_scope"] == "full_5"
    # Timing fields present
    for field in (
        "load_time_ms",
        "preprocess_ms_p50",
        "inference_ms_p50",
        "postprocess_ms_p50",
        "evaluation_ms_p50",
        "total_latency_ms_p50",
        "total_latency_ms_p95",
        "images_per_second",
    ):
        assert field in row, field
    # AP fields present (may be 0 for mock against random GT, that's fine)
    for field in ("ap50", "map50_95", "class_agnostic_ap50", "precision50", "recall50", "f1_50"):
        assert field in row


def test_persistent_benchmark_require_gpu_blocks_cpu(tmp_path: Path) -> None:
    """--require-gpu + CPU fallback must produce GPU_REQUIRED_NOT_USED."""
    root = _make_yolo_fixture(tmp_path, n_images=2)
    samples = [
        DatasetSample(
            image_path=str(root / "images" / "img_000.jpg"),
            gt_boxes=[[10.0, 10.0, 50.0, 50.0]],
            gt_classes=["person"],
        )
    ]
    # Force CPU but require GPU — must fail with structured code.
    row = run_persistent_detection_benchmark(
        model_id="mock-detect",
        samples=samples,
        device_requested="cpu",
        require_gpu=True,
        sample_gpu=False,
    )
    assert row["status"] == "failed"
    assert row["code"] == "GPU_REQUIRED_NOT_USED"
    assert row["device_actual"].startswith("cpu")
    assert row["n_images_evaluated"] == 0
    assert row["evaluation_scope"] == "failed"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _vsx_cmd() -> list[str]:
    binary = shutil.which("visionservex")
    return [binary] if binary else [sys.executable, "-m", "visionservex"]


def test_benchmark_detection_help_lists_v217_flags() -> None:
    res = subprocess.run(
        [*_vsx_cmd(), "benchmark-detection", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0
    assert "--require-gpu" in res.stdout
    assert "--sample-gpu" in res.stdout
    assert "--models" in res.stdout
    assert "--include-mocks" in res.stdout
    assert "--include-aliases" in res.stdout


def test_benchmark_detection_rejects_only_mocks(tmp_path: Path) -> None:
    """All-mock candidate list must fail at entry with ALL_MODELS_REJECTED."""
    root = _make_yolo_fixture(tmp_path, n_images=2)
    out = tmp_path / "result.json"
    res = subprocess.run(
        [
            *_vsx_cmd(),
            "benchmark-detection",
            "--models",
            "mock-detect,mock-open-vocab",
            "--dataset",
            f"yolo:{root}",
            "--max-images",
            "2",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert res.returncode == 2, (res.stdout, res.stderr)
    payload = json.loads(out.read_text())
    assert payload["code"] == "ALL_MODELS_REJECTED"
    assert any(s["reason"] == "MOCK_MODEL" for s in payload["skipped"])


def test_benchmark_detection_cli_persistent_path_writes_schema(tmp_path: Path) -> None:
    root = _make_yolo_fixture(tmp_path, n_images=5)
    out = tmp_path / "result.json"
    res = subprocess.run(
        [
            *_vsx_cmd(),
            "benchmark-detection",
            "--models",
            "mock-detect",
            "--include-mocks",
            "--dataset",
            f"yolo:{root}",
            "--max-images",
            "5",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    payload = json.loads(out.read_text())
    assert payload["benchmark_type"] == "persistent_detection_v2170"
    assert payload["models"], payload
    row = payload["models"][0]
    assert row["status"] == "ok"
    assert row["load_count"] == 1
    assert row["n_images_evaluated"] == 5
    assert row["evaluation_scope"] == "full_5"
    assert "total_latency_ms_p50" in row

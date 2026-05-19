# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.31.0: COCO RLE conversion for RF-DETR-Seg masks."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

_PYCOCOTOOLS = importlib.util.find_spec("pycocotools") is not None
pytestmark = pytest.mark.skipif(not _PYCOCOTOOLS, reason="pycocotools not installed")


def test_binary_mask_to_rle_roundtrip() -> None:
    """Encode a binary mask to COCO RLE and verify area is correct."""
    from pycocotools import mask as mask_utils

    h, w = 480, 640
    m = np.zeros((h, w), dtype=np.uint8)
    m[100:200, 50:150] = 1  # 100x100 = 10000 px
    binary = np.asfortranarray(m)
    rle = mask_utils.encode(binary)
    area = float(mask_utils.area(rle))
    assert abs(area - 10000.0) < 1.0, f"area mismatch: {area}"
    # Decode back
    decoded = mask_utils.decode(rle)
    assert np.array_equal(decoded, m)


def test_rle_encode_decode_for_json_serialization() -> None:
    """bytes counts decode to str for JSON, re-encode for pycocotools."""
    from pycocotools import mask as mask_utils

    h, w = 256, 256
    m = np.zeros((h, w), dtype=np.uint8)
    m[40:80, 40:80] = 1
    rle = mask_utils.encode(np.asfortranarray(m))
    # JSON serialization: decode bytes → str
    rle_str = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}
    assert isinstance(rle_str["counts"], str)
    # Re-encode for pycocotools internal use: str → bytes
    rle_bytes = {"size": rle_str["size"], "counts": rle_str["counts"].encode("utf-8")}
    area = float(mask_utils.area(rle_bytes))
    assert area > 0.0, "area must be positive after re-encode"


def test_build_class_id_to_coco_cat() -> None:
    """Class-id to COCO category mapping matches COCO 80-class ordering."""
    from visionservex.runtime.rfdetr_seg_benchmark import _build_class_id_to_coco_cat

    fake_ann = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 2, "name": "bicycle"},
            {"id": 3, "name": "car"},
        ]
    }
    m = _build_class_id_to_coco_cat(fake_ann)
    assert m[0] == 1  # person
    assert m[1] == 2  # bicycle
    assert m[2] == 3  # car
    assert 3 not in m  # only 3 categories


def test_coco_class_id_mapping_with_gaps() -> None:
    """COCO has gaps in category IDs (e.g. no id=11,25). Mapping must be index-based."""
    from visionservex.runtime.rfdetr_seg_benchmark import _build_class_id_to_coco_cat

    # Mimics real COCO category ordering with gaps
    cats = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "bicycle"},
        {"id": 3, "name": "car"},
        {"id": 4, "name": "motorcycle"},
        {"id": 5, "name": "airplane"},
        {"id": 7, "name": "bus"},
    ]  # no id 6
    m = _build_class_id_to_coco_cat({"categories": cats})
    assert m[0] == 1
    assert m[5] == 7  # index 5 maps to category_id 7 (gap at 6)


def test_rfdetr_seg_benchmark_returns_structured_blocker_on_missing_dataset() -> None:
    """Missing annotation file must return expected_blocker, not crash."""
    from visionservex.runtime.rfdetr_seg_benchmark import run_rfdetr_seg_benchmark

    result = run_rfdetr_seg_benchmark(
        ann_file="/nonexistent/path/to/coco.json",
        images_dir="/nonexistent/images",
        model_id="rfdetr-seg-small",
        device="cpu",
    )
    assert result.status == "expected_blocker"
    assert result.code in (
        "COCO_INSTANCE_DATASET_REQUIRED",
        "RFDETR_SEG_PACKAGE_REQUIRED",
        "PYCOCOTOOLS_REQUIRED",
    )


def test_rfdetr_seg_benchmark_no_nan_in_result() -> None:
    """Result fields must never be NaN — use None instead."""
    from visionservex.runtime.rfdetr_seg_benchmark import RFDETRSegBenchmarkResult

    r = RFDETRSegBenchmarkResult(model_id="x", status="ok", code="OK")
    d = r.to_dict()
    for k, v in d.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"field {k} is NaN"


def test_benchmark_segmentation_cli_exists() -> None:
    """benchmark-segmentation CLI must be callable and accept new v2.31 flags."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "visionservex", "benchmark-segmentation", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(repo),
    )
    assert proc.returncode == 0, f"--help failed: {proc.stderr[:200]}"
    assert "--max-images" in proc.stdout
    assert "--threshold" in proc.stdout

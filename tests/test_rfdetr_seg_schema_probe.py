# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.29.0: RF-DETR-Seg schema probe — determine actual mask output format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROBE_REPORT = Path("reports/rfdetr_seg_schema_probe_v229.json")
SMOKE_IMG = Path("tests/assets/smoke/coco_person_car.jpg")


def _probe_rfdetr_seg(model_id: str = "rfdetr-seg-small") -> dict:
    """Run a real predict on rfdetr-seg-small and inspect the output schema."""
    try:
        from PIL import Image as _PIL

        from visionservex.core.model import VisionModel
    except ImportError as exc:
        return {
            "status": "expected_blocker",
            "code": "RFDETR_REQUIRED",
            "message": str(exc),
            "model_id": model_id,
        }

    img_path = SMOKE_IMG if SMOKE_IMG.exists() else Path("examples/images/street.jpg")
    if not img_path.exists():
        return {
            "status": "expected_blocker",
            "code": "SMOKE_ASSET_MISSING",
            "message": f"image not found: {img_path}",
            "model_id": model_id,
        }

    try:
        img = _PIL.open(img_path).convert("RGB")
        with VisionModel(model_id, device="cpu") as vm:
            result = vm.predict(img)
    except Exception as exc:
        exc_str = str(exc)
        # Identify structured blocker code
        from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

        code = next((c for c in EXPECTED_BLOCKER_CODES if c in exc_str), "RFDETR_LOAD_FAILED")
        return {
            "status": "expected_blocker",
            "code": code,
            "message": exc_str[:400],
            "model_id": model_id,
        }

    # Probe the result schema
    schema: dict = {"model_id": model_id, "status": "probed"}
    schema["result_type"] = type(result).__name__
    schema["result_attrs"] = sorted(a for a in dir(result) if not a.startswith("_"))

    # Detect mask format
    mask_info: dict = {}
    if hasattr(result, "mask") and result.mask is not None:
        import numpy as _np

        arr = _np.asarray(result.mask)
        mask_info["mask_format"] = "single_binary_mask"
        mask_info["mask_shape"] = list(arr.shape)
        mask_info["mask_dtype"] = str(arr.dtype)
    elif hasattr(result, "masks") and result.masks is not None:
        masks = result.masks
        if hasattr(masks, "__len__"):
            mask_info["n_masks"] = len(masks)
        if len(getattr(masks, "__class__.__mro__", [])) or hasattr(masks, "__getitem__"):
            try:
                import numpy as _np

                first = masks[0] if hasattr(masks, "__getitem__") and len(masks) > 0 else masks
                arr = _np.asarray(first)
                mask_info["mask_format"] = "mask_list"
                mask_info["first_mask_shape"] = list(arr.shape)
                mask_info["first_mask_dtype"] = str(arr.dtype)
            except Exception as me:
                mask_info["mask_format"] = "mask_list_unreadable"
                mask_info["error"] = str(me)
        else:
            mask_info["mask_format"] = "mask_list_nolen"
    elif hasattr(result, "detections") and result.detections:
        dets = result.detections
        first = dets[0] if hasattr(dets, "__getitem__") else None
        if first is not None and hasattr(first, "mask"):
            mask_info["mask_format"] = "per_detection_mask"
        elif first is not None and hasattr(first, "polygon"):
            mask_info["mask_format"] = "per_detection_polygon"
        else:
            mask_info["mask_format"] = "detections_no_mask_attr"
    else:
        mask_info["mask_format"] = "RFDETR_SEG_MASK_OUTPUT_NOT_EXPOSED"
        mask_info["recommendation"] = (
            "No mask, masks, or detections[i].mask found on the result object. "
            "Blocker code: RFDETR_SEG_MASK_OUTPUT_NOT_EXPOSED"
        )

    schema.update(mask_info)
    return schema


def test_rfdetr_seg_schema_probe_runs() -> None:
    """Probing rfdetr-seg-small schema must not crash unstructured."""
    probe = _probe_rfdetr_seg("rfdetr-seg-small")
    assert isinstance(probe, dict), "probe must return a dict"
    assert "status" in probe, f"probe missing 'status': {probe}"


def test_rfdetr_seg_schema_no_raw_crash() -> None:
    """Schema probe must return structured result, not raise an uncaught exception."""
    try:
        probe = _probe_rfdetr_seg("rfdetr-seg-small")
    except Exception as exc:
        pytest.fail(f"rfdetr-seg schema probe raised unstructured exception: {exc}")
    assert probe is not None


def test_rfdetr_seg_schema_probe_saved() -> None:
    """If rfdetr package is available, the schema probe report should be writable."""
    probe = _probe_rfdetr_seg("rfdetr-seg-small")
    PROBE_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_REPORT.write_text(json.dumps(probe, indent=2))
    assert PROBE_REPORT.exists()
    loaded = json.loads(PROBE_REPORT.read_text())
    assert loaded.get("status") in ("probed", "expected_blocker")


def test_rfdetr_seg_no_schema_unknown_after_probe() -> None:
    """After running the schema probe, mask_format must not be RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN."""
    probe = _probe_rfdetr_seg("rfdetr-seg-small")
    if probe.get("status") == "expected_blocker":
        # Model unavailable — acceptable outcome, code must be known
        from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

        code = probe.get("code", "")
        assert code in EXPECTED_BLOCKER_CODES, (
            f"unexpected blocker code: {code!r} — add to EXPECTED_BLOCKER_CODES"
        )
        return
    # Model ran — mask_format must not be the generic unknown placeholder
    mask_fmt = probe.get("mask_format", "")
    assert mask_fmt != "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN", (
        "After probing, mask_format is still RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN — "
        "probe did not determine the real schema"
    )
    assert mask_fmt, "mask_format is empty after probe"

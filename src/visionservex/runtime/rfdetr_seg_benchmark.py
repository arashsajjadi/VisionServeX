# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.31.0: RF-DETR-Seg COCO mask-AP benchmark runner.

Runs rfdetr-seg-* on a COCO-format annotation set, converts each
``SegmentationResult.segments[i].mask`` (HxW uint8) to COCO RLE via
pycocotools, then evaluates mask AP using pycocotools COCOeval.

Usage:
    from visionservex.runtime.rfdetr_seg_benchmark import run_rfdetr_seg_benchmark
    result = run_rfdetr_seg_benchmark(
        ann_file="/path/to/instances_val2017_400.json",
        images_dir="/path/to/images",
        model_id="rfdetr-seg-small",
        device="cuda",
        threshold=0.3,
        max_images=400,
        draw_dir=Path("visuals/rfdetr_seg"),
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RFDETRSegBenchmarkResult:
    model_id: str
    status: str  # ok | expected_blocker
    code: str

    n_images: int = 0
    n_predictions: int = 0
    invalid_mask_count: int = 0

    mask_mAP50_95: float | None = None
    mask_AP50: float | None = None
    mask_AP75: float | None = None
    mask_APs: float | None = None
    mask_APm: float | None = None
    mask_APl: float | None = None

    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    fps: float | None = None
    total_runtime_s: float | None = None

    device: str = "cuda"
    threshold: float = 0.3
    ann_file: str = ""
    images_dir: str = ""
    draw_dir: str = ""
    failures: list[dict[str, Any]] = field(default_factory=list)
    version: str = "v2.31.0"

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _build_class_id_to_coco_cat(ann_data: dict[str, Any]) -> dict[int, int]:
    """Map 0-indexed class_id (sorted by COCO cat id) → COCO category_id.

    RF-DETR-Seg is trained on COCO 80 classes (0-indexed, sorted by COCO
    category_id). COCO category IDs are 1-90 with gaps; this map translates
    the 0-based index the model emits into the actual COCO category_id.
    """
    cats = sorted(ann_data.get("categories", []), key=lambda c: c["id"])
    return {i: c["id"] for i, c in enumerate(cats)}


def run_rfdetr_seg_benchmark(
    *,
    ann_file: str | Path,
    images_dir: str | Path,
    model_id: str = "rfdetr-seg-small",
    device: str = "cuda",
    threshold: float = 0.3,
    max_images: int = 400,
    draw_dir: Path | None = None,
) -> RFDETRSegBenchmarkResult:
    """Run RF-DETR-Seg benchmark against a COCO annotation file.

    Returns a :class:`RFDETRSegBenchmarkResult` with exact mask AP metrics
    or a structured expected_blocker if any dependency is unavailable.
    """
    ann_file = Path(ann_file)
    images_dir = Path(images_dir)

    # ── Dependency checks ────────────────────────────────────────────────
    try:
        import numpy as np
    except ImportError:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="NUMPY_REQUIRED",
        )

    try:
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="PYCOCOTOOLS_REQUIRED",
        )

    try:
        from PIL import Image as PILImage
    except ImportError:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="PILLOW_REQUIRED",
        )

    try:
        from visionservex.core.model import VisionModel
    except ImportError:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="RFDETR_SEG_PACKAGE_REQUIRED",
        )

    # ── Dataset validation ───────────────────────────────────────────────
    if not ann_file.exists():
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="COCO_INSTANCE_DATASET_REQUIRED",
            ann_file=str(ann_file),
        )
    if not images_dir.exists():
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="COCO_INSTANCE_DATASET_REQUIRED",
            images_dir=str(images_dir),
        )

    # ── Load annotations ─────────────────────────────────────────────────
    import json

    ann_data = json.loads(ann_file.read_text())
    class_id_to_coco_cat = _build_class_id_to_coco_cat(ann_data)
    coco_gt = COCO(str(ann_file))

    all_images = sorted(ann_data.get("images", []), key=lambda x: x["id"])
    images_to_run = all_images[:max_images]

    if draw_dir is not None:
        draw_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────────
    coco_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []
    n_predictions = 0
    invalid_masks = 0
    t_run_start = time.monotonic()

    try:
        with VisionModel(model_id, device=device) as vm:
            for img_meta in images_to_run:
                img_id = img_meta["id"]
                fname = img_meta.get("file_name", f"{img_id:012d}.jpg")
                img_path = images_dir / fname
                if not img_path.exists():
                    # Try by numeric name
                    img_path = images_dir / f"{img_id:012d}.jpg"
                if not img_path.exists():
                    failures.append({"image_id": img_id, "reason": "image_file_not_found"})
                    continue

                try:
                    pil_img = PILImage.open(img_path).convert("RGB")
                    h_orig, w_orig = pil_img.height, pil_img.width

                    t0 = time.monotonic()
                    result = vm.predict(pil_img, threshold=threshold)
                    lat_ms = (time.monotonic() - t0) * 1000.0
                    latencies.append(lat_ms)

                    segs = getattr(result, "segments", [])
                    for seg in segs:
                        try:
                            raw_mask = np.asarray(seg.mask)
                            if raw_mask.ndim != 2:
                                invalid_masks += 1
                                continue
                            if raw_mask.shape[0] != h_orig or raw_mask.shape[1] != w_orig:
                                invalid_masks += 1
                                failures.append(
                                    {
                                        "image_id": img_id,
                                        "reason": "mask_shape_mismatch",
                                        "mask_shape": list(raw_mask.shape),
                                        "image_shape": [h_orig, w_orig],
                                    }
                                )
                                continue
                            if raw_mask.max() == 0:
                                invalid_masks += 1
                                continue

                            binary_mask = np.asfortranarray(raw_mask.astype(np.uint8))
                            rle = mask_utils.encode(binary_mask)

                            class_id_0 = int(getattr(seg, "class_id", -1))
                            coco_cat_id = class_id_to_coco_cat.get(class_id_0)
                            if coco_cat_id is None:
                                invalid_masks += 1
                                continue

                            score = float(getattr(seg, "score", 0.0))
                            # rle["counts"] is bytes from mask_utils.encode.
                            # area() and toBbox() accept encoded RLE directly.
                            # frPyObjects is NOT needed here — that function is
                            # for converting uncompressed/polygon inputs to RLE.
                            area = float(mask_utils.area(rle))
                            bbox = list(map(float, mask_utils.toBbox([rle])[0]))
                            # For JSON-compatible storage: counts as UTF-8 string.
                            coco_results.append(
                                {
                                    "image_id": img_id,
                                    "category_id": coco_cat_id,
                                    "segmentation": {
                                        "size": rle["size"],
                                        "counts": rle["counts"].decode("utf-8"),
                                    },
                                    "score": score,
                                    "bbox": bbox,
                                    "area": area,
                                }
                            )
                            n_predictions += 1

                        except Exception as seg_exc:
                            invalid_masks += 1
                            failures.append(
                                {
                                    "image_id": img_id,
                                    "reason": "rle_conversion_error",
                                    "error": str(seg_exc)[:200],
                                }
                            )

                    # Draw overlay
                    if draw_dir is not None and segs:
                        try:
                            from PIL import ImageDraw

                            draw_img = pil_img.copy()
                            dw = ImageDraw.Draw(draw_img, "RGBA")
                            for seg in segs[:20]:
                                # Overlay mask as semi-transparent fill
                                raw_mask = np.asarray(seg.mask)
                                if raw_mask.ndim == 2 and raw_mask.max() > 0:
                                    coords = list(zip(*np.where(raw_mask > 0), strict=False))
                                    if coords:
                                        ys = [c[0] for c in coords]
                                        xs = [c[1] for c in coords]
                                        dw.rectangle(
                                            [min(xs), min(ys), max(xs), max(ys)],
                                            outline=(255, 100, 0),
                                            width=2,
                                        )
                            draw_img.save(draw_dir / f"{img_id:012d}_seg.jpg")
                        except Exception:
                            pass  # draw failure is non-fatal

                except Exception as inf_exc:
                    failures.append(
                        {
                            "image_id": img_id,
                            "reason": "inference_error",
                            "error": str(inf_exc)[:200],
                        }
                    )

    except Exception as model_exc:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="RFDETR_SEG_CHECKPOINT_REQUIRED",
            failures=[{"reason": "model_load_error", "error": str(model_exc)[:400]}],
            ann_file=str(ann_file),
            images_dir=str(images_dir),
            device=device,
        )

    total_runtime_s = time.monotonic() - t_run_start

    # ── COCO evaluation ──────────────────────────────────────────────────
    if not coco_results:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="RFDETR_SEG_NO_PREDICTIONS",
            n_images=len(images_to_run),
            n_predictions=0,
            invalid_mask_count=invalid_masks,
            failures=failures[:100],
            ann_file=str(ann_file),
            images_dir=str(images_dir),
            device=device,
        )

    try:
        # Need to re-encode RLE counts to bytes for COCOeval
        coco_dt_data: list[dict[str, Any]] = []
        for r in coco_results:
            entry = dict(r)
            seg = entry["segmentation"]
            seg_bytes = {
                "size": seg["size"],
                "counts": seg["counts"].encode("utf-8"),
            }
            entry["segmentation"] = seg_bytes
            coco_dt_data.append(entry)

        import contextlib
        import io

        coco_dt = coco_gt.loadRes(coco_dt_data)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats
        # stats[0] = mAP@[.5:.95], stats[1] = AP50, stats[2] = AP75
        # stats[3] = APs (small), stats[4] = APm (medium), stats[5] = APl (large)

        def _safe_float(v: float) -> float | None:
            return float(v) if v >= 0 else None

        mask_mAP = _safe_float(stats[0])
        mask_AP50 = _safe_float(stats[1])
        mask_AP75 = _safe_float(stats[2])
        mask_APs = _safe_float(stats[3])
        mask_APm = _safe_float(stats[4])
        mask_APl = _safe_float(stats[5])

    except Exception as eval_exc:
        return RFDETRSegBenchmarkResult(
            model_id=model_id,
            status="expected_blocker",
            code="RFDETR_SEG_COCO_RLE_CONVERSION_FAILED",
            n_images=len(images_to_run),
            n_predictions=n_predictions,
            invalid_mask_count=invalid_masks,
            failures=[*failures[:50], {"reason": "eval_error", "error": str(eval_exc)[:400]}],
            ann_file=str(ann_file),
            images_dir=str(images_dir),
            device=device,
        )

    # ── Latency stats ────────────────────────────────────────────────────
    lat_p50: float | None = None
    lat_p95: float | None = None
    fps: float | None = None
    if latencies:
        import numpy as np

        lat_p50 = float(np.percentile(latencies, 50))
        lat_p95 = float(np.percentile(latencies, 95))
        fps = 1000.0 / lat_p50 if lat_p50 > 0 else None

    return RFDETRSegBenchmarkResult(
        model_id=model_id,
        status="ok",
        code="OK",
        n_images=len(images_to_run),
        n_predictions=n_predictions,
        invalid_mask_count=invalid_masks,
        mask_mAP50_95=mask_mAP,
        mask_AP50=mask_AP50,
        mask_AP75=mask_AP75,
        mask_APs=mask_APs,
        mask_APm=mask_APm,
        mask_APl=mask_APl,
        latency_ms_p50=lat_p50,
        latency_ms_p95=lat_p95,
        fps=fps,
        total_runtime_s=total_runtime_s,
        device=device,
        threshold=threshold,
        ann_file=str(ann_file),
        images_dir=str(images_dir),
        draw_dir=str(draw_dir) if draw_dir else "",
        failures=failures[:100],
        version="v2.31.0",
    )

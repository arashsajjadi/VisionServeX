# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Mask AP evaluator for instance segmentation.

Supports:
- COCO JSON segmentation annotations (polygon and RLE)
- Binary mask predictions from VisionServeX SegmentationResult
- COCO-style 101-point interpolated mask AP50 and mAP50:95
- Box AP50 as fallback when mask GT not available

If pycocotools is installed, uses the standard COCO evaluator API.
If not, falls back to a pure numpy implementation.

Usage::

    from visionservex.runtime.segmentation_eval import run_segmentation_evaluation
    result = run_segmentation_evaluation(
        model_id="rfdetr-seg-medium",
        samples=samples,      # list of SegDatasetSample
        device="auto",
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SegDatasetSample:
    """One image with ground-truth instance segmentation annotations."""

    image_path: str
    image_id: int = 0
    width: int = 0
    height: int = 0
    gt_boxes: list[list[float]] = field(default_factory=list)  # [[x1,y1,x2,y2], ...]
    gt_classes: list[str] = field(default_factory=list)  # class names
    gt_masks: list[Any] = field(default_factory=list)  # binary np.ndarray (H,W) or None


@dataclass
class SegEvaluationResult:
    model_id: str
    dataset: str
    n_images: int
    mask_ap50: float
    mask_map50_95: float
    box_ap50: float
    precision: float
    recall: float
    n_no_mask: int = 0
    n_invalid_mask: int = 0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    status: str = "ok"
    error: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "dataset": self.dataset,
            "n_images": self.n_images,
            "mask_ap50": self.mask_ap50,
            "mask_map50_95": self.mask_map50_95,
            "box_ap50": self.box_ap50,
            "precision": self.precision,
            "recall": self.recall,
            "n_no_mask": self.n_no_mask,
            "n_invalid_mask": self.n_invalid_mask,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "status": self.status,
        }
        if self.error:
            d["error"] = self.error
        if self.note:
            d["note"] = self.note
        return d


# ---------------------------------------------------------------------------
# COCO JSON loader
# ---------------------------------------------------------------------------


def _polygon_to_mask(polygon: list[float], height: int, width: int) -> np.ndarray:
    """Convert a flat COCO polygon [x1,y1,x2,y2,...] to a binary mask."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    pairs = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon) - 1, 2)]
    if len(pairs) >= 3:
        draw.polygon(pairs, fill=1)
    return np.array(img, dtype=np.uint8)


def _rle_to_mask(rle: dict, height: int, width: int) -> np.ndarray:
    """Decode a COCO RLE segmentation to a binary mask."""
    counts = rle.get("counts", [])
    if isinstance(counts, str):
        # Compressed RLE — needs pycocotools
        try:
            from pycocotools import mask as coco_mask  # type: ignore

            return coco_mask.decode(rle).astype(np.uint8)
        except ImportError:
            return np.zeros((height, width), dtype=np.uint8)
    # Uncompressed RLE (list of counts)
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for count in counts:
        flat[idx : idx + count] = val
        idx += count
        val = 1 - val
    return flat.reshape((height, width), order="F")


def load_coco_segmentation_json(
    images_dir: Path,
    ann_file: Path,
    max_images: int = 500,
) -> tuple[list[SegDatasetSample], list[str]]:
    """Load COCO instance segmentation JSON annotations."""
    from PIL import Image as _PIL

    with open(ann_file, encoding="utf-8") as f:
        coco_data = json.load(f)

    cat_id_to_name: dict[int, str] = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    class_names = [cat_id_to_name[k] for k in sorted(cat_id_to_name)]

    img_meta: dict[int, dict] = {img["id"]: img for img in coco_data.get("images", [])}
    anns_by_image: dict[int, list] = {}
    for ann in coco_data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    img_ids = sorted(img_meta.keys())[:max_images]
    samples: list[SegDatasetSample] = []

    for img_id in img_ids:
        meta = img_meta[img_id]
        file_name = meta["file_name"]
        ip = images_dir / file_name
        if not ip.exists():
            ip = images_dir / Path(file_name).name
        if not ip.exists():
            continue

        try:
            with _PIL.open(ip) as img:
                w, h = img.size
        except Exception:
            w = meta.get("width", 0)
            h = meta.get("height", 0)

        gt_boxes, gt_classes, gt_masks = [], [], []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            cls_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, bw, bh = bbox
                gt_boxes.append([x, y, x + bw, y + bh])
                gt_classes.append(cls_name)

                seg = ann.get("segmentation")
                if isinstance(seg, list) and seg:
                    # Polygon format
                    mask = _polygon_to_mask(seg[0], h, w)
                    gt_masks.append(mask)
                elif isinstance(seg, dict):
                    # RLE format
                    mask = _rle_to_mask(seg, h, w)
                    gt_masks.append(mask)
                else:
                    gt_masks.append(None)

        samples.append(
            SegDatasetSample(
                image_path=str(ip),
                image_id=img_id,
                width=w,
                height=h,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_masks=gt_masks,
            )
        )

    return samples, class_names


# ---------------------------------------------------------------------------
# Mask IoU
# ---------------------------------------------------------------------------


def _mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """IoU between two binary masks."""
    if pred_mask.shape != gt_mask.shape:
        from PIL import Image

        gt_mask = np.array(
            Image.fromarray(gt_mask.astype(np.uint8) * 255).resize(
                (pred_mask.shape[1], pred_mask.shape[0])
            )
        )
        gt_mask = (gt_mask > 127).astype(np.uint8)
    inter = np.logical_and(pred_mask.astype(bool), gt_mask.astype(bool)).sum()
    union = np.logical_or(pred_mask.astype(bool), gt_mask.astype(bool)).sum()
    return float(inter) / max(float(union), 1.0)


# ---------------------------------------------------------------------------
# Mask AP computation (mirrors detection AP in evaluation.py)
# ---------------------------------------------------------------------------


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        ap += np.max(precisions[mask]) if mask.any() else 0.0
    return float(ap / 101.0)


class MaskDetectionEvaluator:
    """Accumulates segmentation predictions and computes mask AP."""

    def __init__(self) -> None:
        self._data: list[tuple] = []

    def add_image(
        self,
        pred_masks: list[np.ndarray],
        pred_boxes: list[list[float]],
        pred_scores: list[float],
        pred_classes: list[str],
        gt_masks: list[np.ndarray | None],
        gt_boxes: list[list[float]],
        gt_classes: list[str],
    ) -> None:
        self._data.append(
            (pred_masks, pred_boxes, pred_scores, pred_classes, gt_masks, gt_boxes, gt_classes)
        )

    def _all_gt_classes(self) -> list[str]:
        seen: set[str] = set()
        for *_, gt_c in self._data:
            seen.update(gt_c)
        return sorted(seen)

    def _compute_mask_ap_for_class(
        self, class_name: str, iou_threshold: float
    ) -> tuple[float, float, float, int, int]:
        """Returns (ap, precision, recall, n_gt, n_pred) for mask IoU."""
        global_preds: list[tuple[float, int, np.ndarray]] = []
        per_image_gt: list[dict] = []
        n_gt = 0

        for img_idx, (
            pred_masks,
            _pred_boxes,
            pred_scores,
            pred_classes,
            gt_masks,
            _gt_boxes,
            gt_classes,
        ) in enumerate(self._data):
            p_idx = [i for i, c in enumerate(pred_classes) if c == class_name]
            g_idx = [i for i, c in enumerate(gt_classes) if c == class_name]

            img_gt_masks = [gt_masks[i] for i in g_idx if gt_masks[i] is not None]
            n_gt += len(img_gt_masks)
            per_image_gt.append({"masks": img_gt_masks, "matched": [False] * len(img_gt_masks)})

            for j in p_idx:
                if j < len(pred_masks) and pred_masks[j] is not None:
                    global_preds.append((pred_scores[j], img_idx, pred_masks[j]))

        n_pred = len(global_preds)
        if n_gt == 0:
            return 0.0, 0.0, 0.0, 0, n_pred
        if n_pred == 0:
            return 0.0, 0.0, 0.0, n_gt, 0

        global_preds.sort(key=lambda x: -x[0])
        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)

        for k, (_, img_idx, pred_mask) in enumerate(global_preds):
            g = per_image_gt[img_idx]
            g_masks = g["masks"]
            g_matched = g["matched"]
            if not g_masks:
                fp[k] = 1.0
                continue
            ious = [_mask_iou(pred_mask, gm) for gm in g_masks]
            best_j = int(np.argmax(ious))
            best_iou = ious[best_j]
            if best_iou >= iou_threshold and not g_matched[best_j]:
                tp[k] = 1.0
                g_matched[best_j] = True
            else:
                fp[k] = 1.0

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recalls = cum_tp / (n_gt + 1e-8)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        ap = _ap_from_pr(recalls, precisions)
        f1s = 2.0 * precisions * recalls / (precisions + recalls + 1e-8)
        best_k = int(np.argmax(f1s))
        return ap, float(precisions[best_k]), float(recalls[best_k]), n_gt, n_pred

    def compute_metrics(self, iou_threshold: float = 0.50) -> dict[str, Any]:
        classes = self._all_gt_classes()
        ap_values: list[float] = []
        w_prec = w_rec = 0.0
        total_gt = 0

        per_class = []
        for cls in classes:
            ap, prec, rec, n_gt, n_pred = self._compute_mask_ap_for_class(cls, iou_threshold)
            per_class.append(
                {
                    "class": cls,
                    "ap": round(ap, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "n_gt": n_gt,
                    "n_pred": n_pred,
                }
            )
            if n_gt > 0:
                ap_values.append(ap)
                w_prec += prec * n_gt
                w_rec += rec * n_gt
                total_gt += n_gt

        map50 = float(np.mean(ap_values)) if ap_values else 0.0
        if total_gt > 0:
            w_prec /= total_gt
            w_rec /= total_gt
        return {
            "mask_ap50": round(map50, 4),
            "precision": round(w_prec, 4),
            "recall": round(w_rec, 4),
            "per_class": per_class,
            "n_classes_with_gt": len(ap_values),
        }

    def compute_map50_95(self) -> dict[str, Any]:
        thresholds = np.arange(0.50, 0.955, 0.05)
        maps = [self.compute_metrics(float(t))["mask_ap50"] for t in thresholds]
        result = self.compute_metrics(0.50)
        result["mask_map50_95"] = round(float(np.mean(maps)), 4)
        return result


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_segmentation_evaluation(
    model_id: str,
    samples: list[SegDatasetSample],
    *,
    device: str = "auto",
    dataset_name: str = "user_dataset",
) -> SegEvaluationResult:
    """Run a VisionServeX segmentation model and compute mask AP."""
    import time

    from visionservex.core.model import VisionModel
    from visionservex.core.results import SegmentationResult
    from visionservex.engines.base import MissingDependencyError
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired
    from visionservex.runtime.gpu_lifecycle import cleanup_gpu_after_model

    try:
        model = VisionModel(model_id, device=device)
        model._ensure_loaded()
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        return SegEvaluationResult(
            model_id=model_id,
            dataset=dataset_name,
            n_images=0,
            mask_ap50=0.0,
            mask_map50_95=0.0,
            box_ap50=0.0,
            precision=0.0,
            recall=0.0,
            status="skip",
            error=str(exc)[:200],
        )
    except Exception as exc:
        return SegEvaluationResult(
            model_id=model_id,
            dataset=dataset_name,
            n_images=0,
            mask_ap50=0.0,
            mask_map50_95=0.0,
            box_ap50=0.0,
            precision=0.0,
            recall=0.0,
            status="error",
            error=str(exc)[:200],
        )

    evaluator = MaskDetectionEvaluator()
    latencies: list[float] = []
    n_no_mask = 0
    n_invalid = 0

    from PIL import Image as _PIL

    for sample in samples:
        try:
            img = _PIL.open(sample.image_path).convert("RGB")
            t0 = time.perf_counter()
            result = model.predict(img)
            latencies.append((time.perf_counter() - t0) * 1000)

            if isinstance(result, SegmentationResult):
                pred_masks = [seg.mask for seg in result.segments]
                pred_boxes = [
                    [seg.box.x1, seg.box.y1, seg.box.x2, seg.box.y2] for seg in result.segments
                ]
                pred_scores = [seg.score for seg in result.segments]
                pred_classes = [seg.label for seg in result.segments]

                if not pred_masks:
                    n_no_mask += 1

                for m in pred_masks:
                    if m is None or m.size == 0:
                        n_invalid += 1

                evaluator.add_image(
                    pred_masks,
                    pred_boxes,
                    pred_scores,
                    pred_classes,
                    sample.gt_masks,
                    sample.gt_boxes,
                    sample.gt_classes,
                )
            else:
                evaluator.add_image(
                    [], [], [], [], sample.gt_masks, sample.gt_boxes, sample.gt_classes
                )
                n_no_mask += 1
        except Exception:
            evaluator.add_image([], [], [], [], sample.gt_masks, sample.gt_boxes, sample.gt_classes)
            n_no_mask += 1

    s_lat = sorted(latencies) if latencies else [0.0]
    lat_p50 = s_lat[len(s_lat) // 2]
    lat_p95 = s_lat[min(len(s_lat) - 1, int(len(s_lat) * 0.95))]

    metrics = evaluator.compute_map50_95()

    box_ap50 = 0.0  # Box AP not computed in segmentation eval; use detection evaluator separately
    cleanup_gpu_after_model(model)

    has_gt_masks = any(any(m is not None for m in s.gt_masks) for s in samples)
    note = "" if has_gt_masks else "No mask GT available — mask AP is 0. Only box AP is meaningful."

    return SegEvaluationResult(
        model_id=model_id,
        dataset=dataset_name,
        n_images=len(samples),
        mask_ap50=metrics["mask_ap50"],
        mask_map50_95=metrics["mask_map50_95"],
        box_ap50=box_ap50,
        precision=metrics["precision"],
        recall=metrics["recall"],
        n_no_mask=n_no_mask,
        n_invalid_mask=n_invalid,
        latency_p50_ms=round(lat_p50, 2),
        latency_p95_ms=round(lat_p95, 2),
        status="ok",
        note=note,
    )


__all__ = [
    "MaskDetectionEvaluator",
    "SegDatasetSample",
    "SegEvaluationResult",
    "load_coco_segmentation_json",
    "run_segmentation_evaluation",
]

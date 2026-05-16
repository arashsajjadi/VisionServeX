# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Real AP/mAP detection evaluation engine.

Implements COCO-style 101-point interpolated AP computation.
Supports:
  - YOLO-format datasets (images/ + labels/ + data.yaml)
  - COCO JSON annotation format
  - Class-aware and class-agnostic matching
  - AP50, mAP50:95, precision, recall, F1 per class and aggregated
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

COCO80_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass
class DatasetSample:
    """One image + its ground-truth annotations."""

    image_path: str
    gt_boxes: list[list[float]]  # [[x1, y1, x2, y2], ...] absolute pixels
    gt_classes: list[str]  # class name strings, parallel with gt_boxes
    width: int = 0
    height: int = 0


@dataclass
class PerClassMetric:
    class_name: str
    ap50: float
    precision: float
    recall: float
    f1: float
    n_gt: int
    n_pred: int


@dataclass
class EvaluationResult:
    model_id: str
    dataset: str
    n_images: int
    ap50: float
    map50_95: float
    precision: float
    recall: float
    f1: float
    per_class: list[PerClassMetric] = field(default_factory=list)
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    n_no_detection: int = 0
    n_invalid_boxes: int = 0
    n_classes_with_gt: int = 0
    device: str = "cpu"
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "dataset": self.dataset,
            "n_images": self.n_images,
            "ap50": self.ap50,
            "map50_95": self.map50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "n_no_detection": self.n_no_detection,
            "n_invalid_boxes": self.n_invalid_boxes,
            "n_classes_with_gt": self.n_classes_with_gt,
            "device": self.device,
            "status": self.status,
            "per_class": [
                {
                    "class": m.class_name,
                    "ap50": m.ap50,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    "n_gt": m.n_gt,
                    "n_pred": m.n_pred,
                }
                for m in sorted(self.per_class, key=lambda x: -x.ap50)[:20]
            ],
        }
        if self.error:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# Core IoU and AP primitives
# ---------------------------------------------------------------------------


def _box_iou(box: list[float], gt_arr: np.ndarray) -> np.ndarray:
    """IoU between one box [x1,y1,x2,y2] and an array of GT boxes [N,4]."""
    x1 = np.maximum(box[0], gt_arr[:, 0])
    y1 = np.maximum(box[1], gt_arr[:, 1])
    x2 = np.minimum(box[2], gt_arr[:, 2])
    y2 = np.minimum(box[3], gt_arr[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_gt = np.maximum(0.0, (gt_arr[:, 2] - gt_arr[:, 0])) * np.maximum(
        0.0, (gt_arr[:, 3] - gt_arr[:, 1])
    )
    union = area_box + area_gt - inter
    return inter / np.maximum(union, 1e-8)


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        mask = recalls >= t
        ap += np.max(precisions[mask]) if mask.any() else 0.0
    return float(ap / 101.0)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DetectionEvaluator:
    """Accumulates per-image predictions and GTs; computes AP metrics."""

    def __init__(self) -> None:
        # Each entry: (pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
        self._data: list[tuple] = []

    def add_image(
        self,
        pred_boxes: list[list[float]],
        pred_scores: list[float],
        pred_classes: list[str],
        gt_boxes: list[list[float]],
        gt_classes: list[str],
    ) -> None:
        self._data.append((pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes))

    def _all_gt_classes(self) -> list[str]:
        seen: set[str] = set()
        for _, _, _, _, gt_cls in self._data:
            seen.update(gt_cls)
        return sorted(seen)

    def _compute_ap_for_class(
        self,
        class_name: str,
        iou_threshold: float,
    ) -> tuple[float, float, float, int, int]:
        """Returns (ap, precision, recall, n_gt, n_pred) at given IoU threshold."""
        # Collect all predictions for this class across all images
        global_preds: list[tuple[float, int, list[float]]] = []
        per_image_gt: list[dict] = []
        n_gt = 0

        for img_idx, (pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes) in enumerate(
            self._data
        ):
            cls_pred_idx = [i for i, c in enumerate(pred_classes) if c == class_name]
            cls_gt_idx = [i for i, c in enumerate(gt_classes) if c == class_name]

            img_gt_boxes = [gt_boxes[i] for i in cls_gt_idx]
            n_gt += len(img_gt_boxes)
            per_image_gt.append({"boxes": img_gt_boxes, "matched": [False] * len(img_gt_boxes)})

            for j in cls_pred_idx:
                global_preds.append((pred_scores[j], img_idx, pred_boxes[j]))

        n_pred = len(global_preds)
        if n_gt == 0:
            return 0.0, 0.0, 0.0, 0, n_pred
        if n_pred == 0:
            return 0.0, 0.0, 0.0, n_gt, 0

        # Sort all predictions by score descending
        global_preds.sort(key=lambda x: -x[0])

        tp = np.zeros(n_pred, dtype=np.float32)
        fp = np.zeros(n_pred, dtype=np.float32)

        for k, (_, img_idx, pb) in enumerate(global_preds):
            g = per_image_gt[img_idx]
            g_boxes = g["boxes"]
            g_matched = g["matched"]

            if not g_boxes:
                fp[k] = 1.0
                continue

            gt_arr = np.array(g_boxes, dtype=np.float32)
            ious = _box_iou(pb, gt_arr)
            best_j = int(np.argmax(ious))
            best_iou = float(ious[best_j])

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

        # Best F1 threshold
        f1s = 2.0 * precisions * recalls / (precisions + recalls + 1e-8)
        best_k = int(np.argmax(f1s))
        return ap, float(precisions[best_k]), float(recalls[best_k]), n_gt, n_pred

    def compute_metrics(self, iou_threshold: float = 0.50) -> dict[str, Any]:
        classes = self._all_gt_classes()
        per_class_results = []
        ap_values = []

        for cls in classes:
            ap, prec, rec, n_gt, n_pred = self._compute_ap_for_class(cls, iou_threshold)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
            per_class_results.append(
                {
                    "class": cls,
                    "ap": round(ap, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "n_gt": n_gt,
                    "n_pred": n_pred,
                }
            )
            if n_gt > 0:
                ap_values.append(ap)

        map50 = float(np.mean(ap_values)) if ap_values else 0.0

        total_gt = sum(d["n_gt"] for d in per_class_results)
        if total_gt > 0:
            w_prec = sum(d["precision"] * d["n_gt"] for d in per_class_results) / total_gt
            w_rec = sum(d["recall"] * d["n_gt"] for d in per_class_results) / total_gt
        else:
            w_prec = w_rec = 0.0
        w_f1 = 2.0 * w_prec * w_rec / (w_prec + w_rec + 1e-8)

        return {
            "map50": round(map50, 4),
            "precision": round(w_prec, 4),
            "recall": round(w_rec, 4),
            "f1": round(w_f1, 4),
            "per_class": per_class_results,
            "n_classes_with_gt": len(ap_values),
        }

    def compute_map50_95(self) -> dict[str, Any]:
        thresholds = np.arange(0.50, 0.955, 0.05)
        maps_per_threshold = []
        for t in thresholds:
            m = self.compute_metrics(iou_threshold=float(t))
            maps_per_threshold.append(m["map50"])
        result = self.compute_metrics(iou_threshold=0.50)
        result["map50_95"] = round(float(np.mean(maps_per_threshold)), 4)
        return result


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _read_class_names_from_yaml(yaml_path: Path) -> list[str]:
    """Read class names from a data.yaml / dataset.yaml file."""
    import yaml  # already in base deps

    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        if isinstance(names, list):
            return [str(n) for n in names]
        if isinstance(names, dict):
            # {0: 'person', 1: 'bicycle', ...}
            return [str(names[k]) for k in sorted(names)]
    except Exception:
        pass
    return COCO80_CLASSES


def load_yolo_format(
    dataset_dir: Path,
    max_images: int = 500,
) -> tuple[list[DatasetSample], list[str]]:
    """Load a YOLO-format dataset.

    Returns (samples, class_names).

    Directory layout::

        dataset_dir/
          images/      (or images/train/, images/val/)
          labels/      (*.txt: class_id cx cy w h normalized)
          data.yaml    (optional, contains names:)
    """
    from PIL import Image as _PIL  # already a base dep

    # Find images directory
    img_dir = dataset_dir / "images"
    if not img_dir.exists():
        for sub in ("train", "val", "test"):
            if (dataset_dir / "images" / sub).exists():
                img_dir = dataset_dir / "images" / sub
                break

    if not img_dir.exists():
        img_dir = dataset_dir

    label_dir = dataset_dir / "labels"
    if not label_dir.exists():
        label_dir = img_dir.parent.parent / "labels" / img_dir.name

    # Class names
    class_names: list[str] = COCO80_CLASSES
    for yaml_name in ("data.yaml", "dataset.yaml", "coco128.yaml"):
        yp = dataset_dir / yaml_name
        if yp.exists():
            class_names = _read_class_names_from_yaml(yp)
            break

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in img_exts)
    img_paths = img_paths[:max_images]

    samples: list[DatasetSample] = []
    for ip in img_paths:
        label_path = label_dir / (ip.stem + ".txt")
        if not label_path.exists():
            # Try sibling labels/ structure
            label_path = ip.parent.parent.parent / "labels" / ip.parent.name / (ip.stem + ".txt")

        try:
            with _PIL.open(ip) as img:
                w, h = img.size
        except Exception:
            continue

        gt_boxes: list[list[float]] = []
        gt_classes: list[str] = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1]) * w
                    cy = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2
                    cls_name = (
                        class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                    )
                    gt_boxes.append([x1, y1, x2, y2])
                    gt_classes.append(cls_name)
                except (ValueError, IndexError):
                    continue

        samples.append(
            DatasetSample(
                image_path=str(ip),
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                width=w,
                height=h,
            )
        )

    return samples, class_names


def load_coco_json(
    images_dir: Path,
    ann_file: Path,
    max_images: int = 500,
) -> tuple[list[DatasetSample], list[str]]:
    """Load a COCO JSON format dataset.

    Returns (samples, class_names).
    """
    from PIL import Image as _PIL

    with open(ann_file, encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build category mapping: category_id → name
    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco_data.get("categories", [])
    }
    class_names = [cat_id_to_name[k] for k in sorted(cat_id_to_name)]

    # Build image mapping: image_id → {file_name, width, height}
    img_meta: dict[int, dict] = {img["id"]: img for img in coco_data.get("images", [])}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    img_ids = sorted(img_meta.keys())[:max_images]
    samples: list[DatasetSample] = []

    for img_id in img_ids:
        meta = img_meta[img_id]
        file_name = meta["file_name"]
        # Try direct path first, then nested
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

        gt_boxes: list[list[float]] = []
        gt_classes: list[str] = []
        for ann in anns_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            cls_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
            bbox = ann.get("bbox", [])  # COCO: [x, y, w, h]
            if len(bbox) == 4:
                x, y, bw, bh = bbox
                gt_boxes.append([x, y, x + bw, y + bh])
                gt_classes.append(cls_name)

        samples.append(
            DatasetSample(
                image_path=str(ip),
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                width=w,
                height=h,
            )
        )

    return samples, class_names


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_model_on_dataset(
    model_id: str,
    samples: list[DatasetSample],
    *,
    device: str = "auto",
    dataset_name: str = "user_dataset",
) -> EvaluationResult:
    """Run a VisionServeX model on a dataset and compute AP metrics.

    Uses a low confidence threshold (0.01) to get the full PR curve for AP
    computation.
    """
    import time

    from visionservex.core.model import VisionModel
    from visionservex.core.results import DetectionResult
    from visionservex.engines.base import MissingDependencyError
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired

    try:
        model = VisionModel(model_id, device=device)
        model._ensure_loaded()
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        return EvaluationResult(
            model_id=model_id,
            dataset=dataset_name,
            n_images=0,
            ap50=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status="skip",
            error=str(exc)[:200],
        )
    except Exception as exc:
        return EvaluationResult(
            model_id=model_id,
            dataset=dataset_name,
            n_images=0,
            ap50=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status="error",
            error=str(exc)[:200],
        )

    evaluator = DetectionEvaluator()
    latencies: list[float] = []
    n_no_det = 0
    n_invalid = 0

    from PIL import Image as _PIL

    for sample in samples:
        try:
            img = _PIL.open(sample.image_path).convert("RGB")
            t0 = time.perf_counter()
            result = model.predict(img, threshold=0.01)
            lat_ms = (time.perf_counter() - t0) * 1000
            latencies.append(lat_ms)

            if isinstance(result, DetectionResult):
                pred_boxes = [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in result.detections]
                pred_scores = [d.score for d in result.detections]
                pred_classes = [d.label for d in result.detections]

                if not pred_boxes:
                    n_no_det += 1

                w, h = img.size
                for pb in pred_boxes:
                    if (
                        pb[0] < 0
                        or pb[1] < 0
                        or pb[2] > w
                        or pb[3] > h
                        or pb[0] >= pb[2]
                        or pb[1] >= pb[3]
                    ):
                        n_invalid += 1

                evaluator.add_image(
                    pred_boxes, pred_scores, pred_classes, sample.gt_boxes, sample.gt_classes
                )
            else:
                # Non-detection result
                evaluator.add_image([], [], [], sample.gt_boxes, sample.gt_classes)
                n_no_det += 1
        except Exception:
            evaluator.add_image([], [], [], sample.gt_boxes, sample.gt_classes)
            n_no_det += 1

    if not latencies:
        lat_p50 = lat_p95 = 0.0
    else:
        s_lat = sorted(latencies)
        lat_p50 = s_lat[len(s_lat) // 2]
        lat_p95 = s_lat[min(len(s_lat) - 1, int(len(s_lat) * 0.95))]

    metrics = evaluator.compute_map50_95()

    per_class = [
        PerClassMetric(
            class_name=d["class"],
            ap50=d["ap"],
            precision=d["precision"],
            recall=d["recall"],
            f1=d["f1"],
            n_gt=d["n_gt"],
            n_pred=d["n_pred"],
        )
        for d in metrics["per_class"]
    ]

    actual_device = model.device if hasattr(model, "device") else device

    return EvaluationResult(
        model_id=model_id,
        dataset=dataset_name,
        n_images=len(samples),
        ap50=metrics["map50"],
        map50_95=metrics["map50_95"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        per_class=per_class,
        latency_p50_ms=round(lat_p50, 2),
        latency_p95_ms=round(lat_p95, 2),
        n_no_detection=n_no_det,
        n_invalid_boxes=n_invalid,
        n_classes_with_gt=metrics["n_classes_with_gt"],
        device=str(actual_device),
        status="ok",
    )


def generate_honest_conclusion(results: list[EvaluationResult]) -> str:
    """Generate a brutally honest conclusion from multiple evaluation results."""
    ok = [r for r in results if r.status == "ok"]
    if not ok:
        return "No models ran successfully. Check dependencies with `visionservex doctor`."

    has_gt = any(r.n_classes_with_gt > 0 for r in ok)

    if not has_gt:
        fastest = min(ok, key=lambda r: r.latency_p50_ms if r.latency_p50_ms > 0 else 9999)
        return (
            f"No ground-truth annotations found — running latency benchmark only. "
            f"Fastest model: {fastest.model_id} at {fastest.latency_p50_ms:.1f} ms P50. "
            "To compute AP50/mAP, provide a dataset with annotations. "
            "Latency rankings do not imply accuracy rankings."
        )

    # AP-based conclusion
    best_ap50 = max(ok, key=lambda r: r.ap50)
    best_map = max(ok, key=lambda r: r.map50_95)

    parts = [
        f"Best AP50: {best_ap50.model_id} at AP50={best_ap50.ap50:.3f}.",
        f"Best mAP50:95: {best_map.model_id} at mAP50:95={best_map.map50_95:.3f}.",
    ]

    # Check if any model clearly dominates
    sorted_by_ap = sorted(ok, key=lambda r: -r.ap50)
    if len(sorted_by_ap) >= 2:
        diff = sorted_by_ap[0].ap50 - sorted_by_ap[1].ap50
        if diff > 0.02:
            parts.append(
                f"{sorted_by_ap[0].model_id} leads {sorted_by_ap[1].model_id} "
                f"by {diff:.3f} AP50 on this dataset."
            )
        else:
            parts.append(
                f"AP50 difference between top models is {diff:.3f} — within typical "
                "run-to-run variance on small datasets."
            )

    # Warn about small datasets
    n_imgs = ok[0].n_images
    if n_imgs < 100:
        parts.append(
            f"WARNING: Only {n_imgs} images evaluated. AP estimates from small datasets "
            "have high variance. Use ≥500 images for reliable results."
        )

    parts.append(
        "NOTE: AP is computed at IoU=0.50 for AP50, swept 0.50-0.95 for mAP50:95. "
        "These are detection-only metrics. Do not mix with segmentation/classification AP."
    )

    return " ".join(parts)


__all__ = [
    "COCO80_CLASSES",
    "DatasetSample",
    "DetectionEvaluator",
    "EvaluationResult",
    "PerClassMetric",
    "generate_honest_conclusion",
    "load_coco_json",
    "load_yolo_format",
    "run_model_on_dataset",
]

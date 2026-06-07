#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.49.0 LibreYOLO COCO benchmark script.

Uses cached weight files from ~/.cache/visionservex/libreyolo/ and runs
400-image COCO val2017 benchmark with pycocotools.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

COCO_ANN = "/home/arash/datasets/coco_val2017_400_vsx/annotations.json"
COCO_IMG_DIR = "/home/arash/datasets/coco_val2017_400_vsx/images"
WEIGHT_CACHE = Path.home() / ".cache" / "visionservex" / "libreyolo"

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/libreyolo_coco_bench.json"
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MODELS_ARG = sys.argv[3].split(",") if len(sys.argv) > 3 else None

# Map visionservex model_id → (family_class_import, size, weight_stem)
# Weight file: ~/.cache/visionservex/libreyolo/{FILENAME_PREFIX}{size}.pt
MODEL_MAP: dict[str, tuple[str, str, str]] = {}
for _size in ("n", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-dfine-{_size}"] = ("LibreDFINE", _size, f"LibreDFINE{_size}")
for _size in ("n", "t", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-yolox-{_size}"] = ("LibreYOLOX", _size, f"LibreYOLOX{_size}")
for _size in ("n", "s", "m", "l"):
    MODEL_MAP[f"libreyolo-rfdetr-{_size}"] = ("LibreYOLORFDETR", _size, f"LibreRFDETR{_size}")
for _size in ("r18", "r34", "r50", "r50m", "r101", "l", "x"):
    MODEL_MAP[f"libreyolo-rtdetr-{_size}"] = ("LibreYOLORTDETR", _size, f"LibreRTDETR{_size}")
for _size in ("t", "s", "m", "c"):
    MODEL_MAP[f"libreyolo-yolov9-{_size}"] = ("LibreYOLO9", _size, f"LibreYOLO9{_size}")
# Segmentation variants
for _size in ("n", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-dfine-{_size}-seg"] = (
        "LibreDFINE",
        f"{_size}-seg",
        f"LibreDFINE{_size}-seg",
    )
for _size in ("n", "s", "m", "l"):
    MODEL_MAP[f"libreyolo-rfdetr-{_size}-seg"] = (
        "LibreYOLORFDETR",
        f"{_size}-seg",
        f"LibreRFDETR{_size}-seg",
    )
for _size in ("r18", "r34", "r50", "r50m", "r101"):
    MODEL_MAP[f"libreyolo-rtdetr-{_size}-seg"] = (
        "LibreYOLORTDETR",
        f"{_size}-seg",
        f"LibreRTDETR{_size}-seg",
    )
for _size in ("n", "t", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-yolox-{_size}-seg"] = (
        "LibreYOLOX",
        f"{_size}-seg",
        f"LibreYOLOX{_size}-seg",
    )

DETECT_MODELS = [k for k in MODEL_MAP if not k.endswith("-seg") and "yolov9" not in k]
models_to_run = MODELS_ARG if MODELS_ARG else DETECT_MODELS


def _get_family_cls(cls_name: str):
    """Dynamically import the correct model class."""
    from libreyolo.models.dfine.model import LibreDFINE
    from libreyolo.models.rtdetr.model import LibreYOLORTDETR
    from libreyolo.models.yolox.model import LibreYOLOX

    mapping = {
        "LibreDFINE": LibreDFINE,
        "LibreYOLOX": LibreYOLOX,
        "LibreYOLORTDETR": LibreYOLORTDETR,
    }
    if cls_name == "LibreYOLORFDETR":
        try:
            from libreyolo.models.rfdetr.model import LibreYOLORFDETR

            mapping["LibreYOLORFDETR"] = LibreYOLORFDETR
        except ImportError:
            return None
    if cls_name == "LibreYOLO9":
        try:
            from libreyolo.models.yolo9.model import LibreYOLO9

            mapping["LibreYOLO9"] = LibreYOLO9
        except ImportError:
            return None
    return mapping.get(cls_name)


def _load_coco():
    from pycocotools.coco import COCO

    coco_gt = COCO(COCO_ANN)
    img_infos = list(coco_gt.imgs.values())[:MAX_IMAGES]
    img_paths, img_ids = [], []
    for img in img_infos:
        p = Path(COCO_IMG_DIR) / img["file_name"]
        if p.exists():
            img_paths.append(str(p))
            img_ids.append(img["id"])
    return coco_gt, img_ids, img_paths


def _coco_cat_map(coco_gt) -> dict[int, int]:
    """Map 0-indexed model class → COCO category_id."""
    return dict(enumerate(sorted(coco_gt.cats.keys())))


def _run_model(model_id: str, coco_gt, img_ids, img_paths, cat_map) -> dict:
    result = {
        "model_id": model_id,
        "status": "failed",
        "code": "",
        "mAP50_95": None,
        "AP50": None,
        "AP75": None,
        "latency_ms_p50": None,
        "fps": None,
        "n_images": len(img_paths),
        "error": None,
    }

    if model_id not in MODEL_MAP:
        result["code"] = "UNKNOWN_MODEL_ID"
        result["error"] = f"No mapping for {model_id}"
        return result

    cls_name, size, weight_stem = MODEL_MAP[model_id]
    weight_path = WEIGHT_CACHE / f"{weight_stem}.pt"

    if not weight_path.exists():
        result["code"] = "CHECKPOINT_REQUIRED"
        result["error"] = f"Weight not cached: {weight_path}"
        print(f"  [{model_id}] MISSING: {weight_path}")
        return result

    cls = _get_family_cls(cls_name)
    if cls is None:
        result["code"] = "MODEL_CLASS_UNAVAILABLE"
        result["error"] = f"Cannot import {cls_name}"
        print(f"  [{model_id}] CLASS UNAVAILABLE: {cls_name}")
        return result

    try:
        model = cls(model_path=str(weight_path), size=size.replace("-seg", ""), device="cuda")
        print(f"  [{model_id}] loaded from {weight_stem}.pt")
    except Exception as e:
        result["code"] = "MODEL_LOAD_FAILED"
        result["error"] = str(e)[:300]
        print(f"  [{model_id}] LOAD FAILED: {str(e)[:80]}")
        return result

    preds = []
    latencies = []
    size.endswith("-seg")

    for img_id, img_path in zip(img_ids, img_paths, strict=False):
        try:
            t0 = time.perf_counter()
            out = model(source=img_path, save=False, verbose=False)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            if out is None:
                continue
            boxes = getattr(out, "boxes", None)
            if boxes is None or boxes.xyxy.shape[0] == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls, score in zip(xyxy, cls_ids, confs, strict=False):
                coco_cat = cat_map.get(int(cls), int(cls) + 1)
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score),
                    }
                )
        except Exception:
            continue

    try:
        del model
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    if not preds:
        result["code"] = "NO_PREDICTIONS"
        result["error"] = "No valid predictions produced"
        print(f"  [{model_id}] NO PREDICTIONS")
        return result

    from pycocotools.cocoeval import COCOeval

    try:
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = img_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        lats = sorted(latencies)
        p50 = lats[len(lats) // 2] if lats else None
        fps = 1000.0 / (sum(latencies) / len(latencies)) if latencies else None
        result.update(
            {
                "status": "ok",
                "code": "OK",
                "mAP50_95": round(float(ev.stats[0]), 4),
                "AP50": round(float(ev.stats[1]), 4),
                "AP75": round(float(ev.stats[2]), 4),
                "latency_ms_p50": round(p50, 1) if p50 else None,
                "fps": round(fps, 1) if fps else None,
            }
        )
        print(
            f"  [{model_id}] mAP50:95={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} lat_p50={p50:.0f}ms fps={fps:.0f}"
        )
    except Exception as e:
        result["code"] = "EVAL_FAILED"
        result["error"] = str(e)[:300]
        print(f"  [{model_id}] EVAL FAILED: {e}")

    return result


def main():
    print(f"LibreYOLO COCO benchmark: {len(models_to_run)} models, {MAX_IMAGES} images")
    coco_gt, img_ids, img_paths = _load_coco()
    cat_map = _coco_cat_map(coco_gt)
    print(f"COCO GT: {len(img_ids)} images, {len(cat_map)} categories")

    results = []
    for i, mid in enumerate(models_to_run):
        print(f"\n[{i + 1}/{len(models_to_run)}] {mid}")
        r = _run_model(mid, coco_gt, img_ids, img_paths, cat_map)
        results.append(r)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(
            {
                "benchmark_type": "libreyolo_coco_detection",
                "dataset": "yolo:" + COCO_IMG_DIR,
                "n_images": len(img_paths),
                "models": results,
            },
            f,
            indent=2,
        )

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSummary: {len(ok)}/{len(results)} succeeded")
    if ok:
        print(f"Best mAP50:95: {max(r['mAP50_95'] for r in ok):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

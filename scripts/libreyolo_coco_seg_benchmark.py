#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.50.0 LibreYOLO COCO instance segmentation benchmark.

Runs LibreYOLO models with -seg suffix that produce masks against the
400-image COCO val2017 subset using pycocotools mask AP computation.

Only models that actually emit masks (probed at load time) are
benchmarked. Models without mask output get model_capability_mismatch.
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

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/libreyolo_coco_seg.json"
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MODELS_ARG = sys.argv[3].split(",") if len(sys.argv) > 3 else None

MODEL_MAP: dict[str, tuple[str, str, str]] = {}
for sz in ("n", "s", "m", "l"):
    MODEL_MAP[f"libreyolo-rfdetr-{sz}-seg"] = ("LibreYOLORFDETR", sz, f"LibreRFDETR{sz}-seg")
# Other seg variants are listed but probed and either fall back or
# get model_capability_mismatch at runtime.
for sz in ("r18", "r34", "r50", "r50m", "r101"):
    MODEL_MAP[f"libreyolo-rtdetr-{sz}-seg"] = ("LibreYOLORTDETR", sz, f"LibreRTDETR{sz}-seg")
for sz in ("n", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-dfine-{sz}-seg"] = ("LibreDFINE", sz, f"LibreDFINE{sz}-seg")
for sz in ("n", "t", "s", "m", "l", "x"):
    MODEL_MAP[f"libreyolo-yolox-{sz}-seg"] = ("LibreYOLOX", sz, f"LibreYOLOX{sz}-seg")


def _get_cls(name: str):
    from libreyolo.models.dfine.model import LibreDFINE
    from libreyolo.models.rtdetr.model import LibreYOLORTDETR
    from libreyolo.models.yolox.model import LibreYOLOX

    mapping = {
        "LibreDFINE": LibreDFINE,
        "LibreYOLOX": LibreYOLOX,
        "LibreYOLORTDETR": LibreYOLORTDETR,
    }
    if name == "LibreYOLORFDETR":
        try:
            from libreyolo.models.rfdetr.model import LibreYOLORFDETR

            return LibreYOLORFDETR
        except ImportError:
            return None
    return mapping.get(name)


def _coco_cat_map(coco_gt):
    return {i: cat_id for i, cat_id in enumerate(sorted(coco_gt.cats.keys()))}


def _encode_mask(mask_t):
    import numpy as np
    from pycocotools import mask as mask_utils

    arr = (mask_t.cpu().numpy() > 0.5).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(arr))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _run_model(model_id: str, coco_gt, img_ids, img_paths, cat_map):
    result = {
        "model_id": model_id,
        "status": "failed",
        "code": "",
        "mask_mAP50_95": None,
        "mask_AP50": None,
        "mask_AP75": None,
        "mean_iou": None,
        "latency_ms_p50": None,
        "fps": None,
        "n_images": len(img_paths),
        "observed_schema": {},
        "error": None,
    }
    if model_id not in MODEL_MAP:
        result["code"] = "UNKNOWN_MODEL_ID"
        return result
    cls_name, size, stem = MODEL_MAP[model_id]
    wpath = WEIGHT_CACHE / f"{stem}.pt"
    if not wpath.exists():
        result["code"] = "CHECKPOINT_NOT_PUBLISHED"
        result["error"] = f"Weight not in cache; not published on HF: {stem}.pt"
        print(f"  [{model_id}] NOT PUBLISHED")
        return result
    cls = _get_cls(cls_name)
    if cls is None:
        result["code"] = "MODEL_CLASS_UNAVAILABLE"
        return result
    try:
        model = cls(model_path=str(wpath), size=size, device="cuda")
    except Exception as e:
        result["code"] = "MODEL_LOAD_FAILED"
        result["error"] = str(e)[:300]
        print(f"  [{model_id}] LOAD FAILED: {str(e)[:80]}")
        return result

    # Probe first image to detect mask capability
    probe = model(source=img_paths[0], save=False, verbose=False)
    has_masks = probe.masks is not None and probe.masks.data.numel() > 0
    result["observed_schema"] = {
        "has_boxes": probe.boxes is not None,
        "has_masks": bool(has_masks),
        "n_classes": len(probe.names) if hasattr(probe, "names") and probe.names else 0,
    }
    if not has_masks:
        result["code"] = "LIBREYOLO_SEG_MASK_OUTPUT_NOT_AVAILABLE"
        result["error"] = (
            "Weight stem has -seg suffix but model.predict().masks is None. "
            "This model only emits detection boxes, no segmentation masks."
        )
        result["status"] = "capability_mismatch"
        print(f"  [{model_id}] CAPABILITY_MISMATCH: no masks output despite -seg name")
        try:
            del model
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return result

    preds = []
    latencies = []
    for img_id, img_path in zip(img_ids, img_paths):
        try:
            t0 = time.perf_counter()
            out = model(source=img_path, save=False, verbose=False)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            if out.masks is None or out.boxes is None or out.boxes.xyxy.shape[0] == 0:
                continue
            masks = out.masks.data
            cls_ids = out.boxes.cls.cpu().numpy().astype(int)
            confs = out.boxes.conf.cpu().numpy()
            for i in range(masks.shape[0]):
                rle = _encode_mask(masks[i])
                coco_cat = cat_map.get(int(cls_ids[i]), int(cls_ids[i]) + 1)
                preds.append(
                    {
                        "image_id": img_id,
                        "category_id": coco_cat,
                        "segmentation": rle,
                        "score": float(confs[i]),
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
        return result

    from pycocotools.cocoeval import COCOeval

    try:
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="segm")
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
                "mask_mAP50_95": round(float(ev.stats[0]), 4),
                "mask_AP50": round(float(ev.stats[1]), 4),
                "mask_AP75": round(float(ev.stats[2]), 4),
                "latency_ms_p50": round(p50, 1) if p50 else None,
                "fps": round(fps, 1) if fps else None,
            }
        )
        print(
            f"  [{model_id}] mask_mAP50:95={ev.stats[0]:.4f} mask_AP50={ev.stats[1]:.4f} lat={p50:.0f}ms"
        )
    except Exception as e:
        result["code"] = "EVAL_FAILED"
        result["error"] = str(e)[:300]
    return result


def main():
    from pycocotools.coco import COCO

    models_to_run = MODELS_ARG if MODELS_ARG else list(MODEL_MAP.keys())
    print(f"LibreYOLO seg benchmark: {len(models_to_run)} models, {MAX_IMAGES} images")
    coco_gt = COCO(COCO_ANN)
    img_infos = list(coco_gt.imgs.values())[:MAX_IMAGES]
    img_paths, img_ids = [], []
    for img in img_infos:
        p = Path(COCO_IMG_DIR) / img["file_name"]
        if p.exists():
            img_paths.append(str(p))
            img_ids.append(img["id"])
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
                "benchmark_type": "libreyolo_coco_instance_segmentation",
                "dataset": "coco-instance-seg:" + COCO_ANN,
                "n_images": len(img_paths),
                "models": results,
            },
            f,
            indent=2,
        )
    ok = sum(1 for r in results if r["status"] == "ok")
    mismatch = sum(1 for r in results if r["status"] == "capability_mismatch")
    failed = len(results) - ok - mismatch
    print(f"\nSummary: {ok} benchmark_passed, {mismatch} capability_mismatch, {failed} other failed")
    if ok:
        best = max((r["mask_mAP50_95"] for r in results if r["status"] == "ok"), default=0)
        print(f"Best mask mAP50:95: {best:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

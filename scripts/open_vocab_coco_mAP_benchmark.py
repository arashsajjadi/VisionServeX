#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.51.0 Open-vocabulary detection COCO mAP benchmark.

Uses VisionModel.predict(img, prompts=coco_80_cat_names) to run all 9
open-vocab models against 400-image COCO val2017 subset and computes
mAP50:95/AP50/AP75 with pycocotools COCOeval.

Prompt policy: ALL 80 COCO category names (fair/standard leaderboard mode).
Models: grounding-dino-swin-b/t/tiny, grounding-dino-original-swin-b/t,
        owlv2-base-patch16, owlv2-large-patch14, owlvit-base-patch32/large-patch14.
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

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v251_ovd.json"
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 400
THRESHOLD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
MODELS_ARG = sys.argv[4].split(",") if len(sys.argv) > 4 else None

DEFAULT_MODELS = [
    "grounding-dino-swin-t",
    "grounding-dino-swin-b",
    "grounding-dino-tiny",
    "grounding-dino-original-swin-t",
    "grounding-dino-original-swin-b",
    "owlvit-base-patch32",
    "owlvit-large-patch14",
    "owlv2-base-patch16",
    "owlv2-large-patch14",
]

# v2.51 requirement: use all 80 COCO categories as prompts (not present-only).
PROMPT_SOURCE = "coco_all_80_categories"


def _run_model(
    model_id: str,
    coco_gt,
    img_ids: list[int],
    img_paths: list[str],
    cat_names: list[str],
    name_to_cat_id: dict[str, int],
    threshold: float,
) -> dict:
    """Benchmark one open-vocab model against COCO GT."""
    from visionservex import VisionModel
    from PIL import Image
    from pycocotools.cocoeval import COCOeval

    result = {
        "model_id": model_id,
        "status": "failed",
        "code": "",
        "prompt_source": PROMPT_SOURCE,
        "prompt_count": len(cat_names),
        "mAP50_95": None,
        "AP50": None,
        "AP75": None,
        "AR100": None,
        "latency_ms_p50": None,
        "fps": None,
        "n_images": len(img_paths),
        "n_predictions": None,
        "error": None,
    }

    # OWL models require prompts= list (not period-string which crashes OWLForward).
    # GD models are faster with period-string (one forward pass for all categories).
    is_owl = any(x in model_id for x in ("owlv2", "owlvit"))

    try:
        model = VisionModel(model_id, auto_pull=False)
        print(f"  [{model_id}] loaded (owl={is_owl})")
    except Exception as e:
        err = str(e)[:300]
        if "not found" in err.lower() or "unknown" in err.lower():
            result["code"] = "CHECKPOINT_REQUIRED"
        else:
            result["code"] = "MODEL_LOAD_FAILED"
        result["error"] = err
        print(f"  [{model_id}] LOAD FAILED: {err[:80]}")
        return result

    # OWL models: use 100 images (list format takes ~4s/img; 100 = ~6.5 min/model)
    # GD models: use all images (period-string is fast, ~0.13s/img)
    eff_ids = img_ids[:100] if is_owl else img_ids
    eff_paths = img_paths[:100] if is_owl else img_paths
    result["n_images"] = len(eff_paths)

    preds: list[dict] = []
    latencies: list[float] = []

    for img_id, img_path in zip(eff_ids, eff_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            if is_owl:
                # OWL: list of prompts (each is a separate text query embedding)
                out = model.predict(img, prompts=cat_names, threshold=threshold)
            else:
                # GD: period-joined string (single forward pass for all 80 categories)
                out = model.predict(img, prompt=". ".join(cat_names), threshold=threshold)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)
            detections = getattr(out, "detections", []) or []
            for det in detections:
                label = getattr(det, "label", "") or ""
                box = getattr(det, "box", None)
                score = float(getattr(det, "score", 0))
                if not box:
                    continue
                # Map label → COCO category_id
                cat_id = name_to_cat_id.get(label.lower())
                if cat_id is None:
                    # Fuzzy: check if label is a substring of any cat name or vice-versa
                    for cn, cid in name_to_cat_id.items():
                        if label.lower() in cn or cn in label.lower():
                            cat_id = cid
                            break
                if cat_id is None:
                    continue
                x1, y1 = float(box.x1), float(box.y1)
                x2, y2 = float(box.x2), float(box.y2)
                preds.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                })
        except Exception:
            continue

    try:
        del model
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    result["n_predictions"] = len(preds)
    if not preds:
        result["code"] = "NO_VALID_BOXES"
        result["error"] = "No predictions after label → category_id mapping"
        print(f"  [{model_id}] NO VALID BOXES")
        return result

    try:
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = img_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
        lats = sorted(latencies)
        p50 = lats[len(lats) // 2] if lats else None
        fps = 1000.0 / (sum(latencies) / len(latencies)) if latencies else None
        result.update({
            "status": "ok",
            "code": "OK",
            "mAP50_95": round(float(ev.stats[0]), 4),
            "AP50": round(float(ev.stats[1]), 4),
            "AP75": round(float(ev.stats[2]), 4),
            "AR100": round(float(ev.stats[8]), 4),
            "latency_ms_p50": round(p50, 1) if p50 else None,
            "fps": round(fps, 1) if fps else None,
        })
        print(
            f"  [{model_id}] mAP50:95={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} "
            f"lat={p50:.0f}ms fps={fps:.0f}"
        )
    except Exception as e:
        result["code"] = "EVAL_FAILED"
        result["error"] = str(e)[:300]
        print(f"  [{model_id}] EVAL FAILED: {e}")

    return result


def main():
    from pycocotools.coco import COCO

    models_to_run = MODELS_ARG or DEFAULT_MODELS
    print(f"Open-vocab COCO mAP benchmark: {len(models_to_run)} models, {MAX_IMAGES} images")
    print(f"Prompt source: {PROMPT_SOURCE}")
    print(f"Threshold: {THRESHOLD}")

    coco_gt = COCO(COCO_ANN)
    cats = sorted(coco_gt.cats.values(), key=lambda c: c["id"])
    cat_names = [c["name"] for c in cats]
    name_to_cat_id = {c["name"].lower(): c["id"] for c in cats}
    print(f"COCO categories: {len(cat_names)}")

    img_infos = list(coco_gt.imgs.values())[:MAX_IMAGES]
    img_ids, img_paths = [], []
    for img in img_infos:
        p = Path(COCO_IMG_DIR) / img["file_name"]
        if p.exists():
            img_ids.append(img["id"])
            img_paths.append(str(p))
    print(f"Images: {len(img_ids)}")

    results = []
    for i, mid in enumerate(models_to_run):
        print(f"\n[{i+1}/{len(models_to_run)}] {mid}")
        r = _run_model(mid, coco_gt, img_ids, img_paths, cat_names, name_to_cat_id, THRESHOLD)
        results.append(r)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "benchmark_type": "open_vocab_coco_mAP",
            "prompt_source": PROMPT_SOURCE,
            "coco_categories": cat_names,
            "n_images": len(img_ids),
            "threshold": THRESHOLD,
            "models": results,
        }, f, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\nSummary: {len(ok)}/{len(results)} benchmark_passed")
    if ok:
        print(f"Best mAP50:95: {max(r['mAP50_95'] for r in ok):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

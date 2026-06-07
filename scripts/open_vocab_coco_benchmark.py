#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v2.50.0 Open-vocab detection benchmark on COCO with per-image category prompts.

For each image, uses the 80 COCO category names as text prompts and evaluates
predicted boxes against COCO GT boxes using pycocotools.
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

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "/tmp/v250_ovd.json"
MAX_IMAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MODELS_ARG = sys.argv[3].split(",") if len(sys.argv) > 3 else None

DEFAULT_MODELS = [
    "owlvit-base-patch32",
    "owlvit-large-patch14",
    "owlv2-base-patch16",
    "owlv2-large-patch14",
]


def main():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(COCO_ANN)
    cats = sorted(coco_gt.cats.values(), key=lambda c: c["id"])
    cat_names = [c["name"] for c in cats]
    {c["id"]: i for i, c in enumerate(cats)}
    idx_to_cat_id = {i: c["id"] for i, c in enumerate(cats)}

    img_infos = list(coco_gt.imgs.values())[:MAX_IMAGES]
    img_records = [
        (img["id"], str(Path(COCO_IMG_DIR) / img["file_name"]))
        for img in img_infos
        if (Path(COCO_IMG_DIR) / img["file_name"]).exists()
    ]
    print(f"Images: {len(img_records)}, prompts: {len(cat_names)} COCO categories")

    models_to_run = MODELS_ARG or DEFAULT_MODELS
    results = []

    for model_id in models_to_run:
        print(f"\n[{model_id}]")
        result = {
            "model_id": model_id,
            "status": "failed",
            "code": "",
            "mAP50_95": None,
            "AP50": None,
            "AP75": None,
            "prompt_category_coverage": None,
            "latency_ms_p50": None,
            "fps": None,
            "n_images": len(img_records),
        }
        try:
            pass
        except Exception as e:
            result["code"] = "IMPORT_FAILED"
            result["error"] = str(e)[:300]
            results.append(result)
            continue

        try:
            preds = []
            latencies = []
            # For each image, run all prompts and collect predictions
            from visionservex.runtime.predict import predict_open_vocab_detection

            for img_id, img_path in img_records:
                t0 = time.perf_counter()
                try:
                    out = predict_open_vocab_detection(
                        model_id=model_id,
                        image_path=img_path,
                        prompts=cat_names,
                        threshold=0.05,
                    )
                except Exception:
                    continue
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                # Output format: list of detections {box, score, label_index}
                for det in out.get("detections", []):
                    label_idx = det.get("label_index")
                    if label_idx is None:
                        continue
                    cat_id = idx_to_cat_id.get(int(label_idx))
                    if cat_id is None:
                        continue
                    box = det.get("xyxy") or det.get("box")
                    if not box or len(box) != 4:
                        continue
                    x1, y1, x2, y2 = box
                    preds.append(
                        {
                            "image_id": img_id,
                            "category_id": cat_id,
                            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            "score": float(det.get("score", 0.5)),
                        }
                    )

            if not preds:
                result["code"] = "NO_PREDICTIONS"
                results.append(result)
                continue

            coco_dt = coco_gt.loadRes(preds)
            ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
            ev.params.imgIds = [r[0] for r in img_records]
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
                    "prompt_category_coverage": round(
                        len({d["category_id"] for d in preds}) / 80, 3
                    ),
                }
            )
            print(
                f"  mAP50:95={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} coverage={result['prompt_category_coverage']}"
            )
        except Exception as e:
            result["code"] = "BENCHMARK_FAILED"
            result["error"] = str(e)[:300]
            print(f"  FAILED: {e}")
        results.append(result)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"benchmark_type": "open_vocab_coco", "models": results}, f, indent=2)
    print(f"\nWritten: {OUT_JSON}")
    print(f"OK: {sum(1 for r in results if r['status'] == 'ok')}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

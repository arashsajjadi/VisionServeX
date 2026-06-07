#!/usr/bin/env python3
"""V3 classic smart-refine benchmark — REAL promptable segmentation on COCO.

Honest design: real COCO val2017 images + real GT instance masks. The user prompt
(box / positive+negative points / polygon) is *derived from the GT mask* (the
standard protocol for promptable-segmentation evaluation, e.g. SAM). The metric
(IoU / boundary-IoU vs GT) is therefore a real measurement of each weight-free
classic refiner — not a synthetic toy. CPU-only.

Usage:
    python scripts/v3_classic_smart_refine_benchmark.py \
        --dataset /home/arash/datasets/coco_val2017_400_vsx \
        --limit 40 --max-side 256 \
        --out  notebook/_runs/<RUN_ID>/reports/v3_classic_smart_refine_benchmark.json \
        --csv  notebook/_runs/<RUN_ID>/reports/v3_classic_smart_refine_benchmark.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from visionservex.smart_annotation import Prompt, list_methods, refine  # noqa: E402
from visionservex.smart_annotation.metrics import (  # noqa: E402
    boundary_iou,
    iou,
    summarize,
)


def _derive_prompt(gt: np.ndarray, rng: np.random.Generator) -> Prompt:
    """Derive a realistic user prompt from a GT mask."""
    import cv2
    from scipy import ndimage

    ys, xs = np.where(gt)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    box = (x1, y1, x2, y2)
    # positive points: from eroded interior so they are safely inside
    interior = ndimage.binary_erosion(gt, iterations=2)
    pys, pxs = np.where(interior if interior.any() else gt)
    k = min(4, len(pys))
    idx = rng.choice(len(pys), size=k, replace=False)
    pos = [[int(pxs[i]), int(pys[i])] for i in idx]
    # negative points: just outside the box, inside the image
    h, w = gt.shape
    neg = []
    for _ in range(4):
        nx = int(np.clip(rng.integers(max(0, x1 - 20), min(w, x2 + 20)), 0, w - 1))
        ny = int(np.clip(rng.integers(max(0, y1 - 20), min(h, y2 + 20)), 0, h - 1))
        if not gt[ny, nx]:
            neg.append([nx, ny])
    # polygon / polyline: subsampled GT contour
    contours, _ = cv2.findContours(gt.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    poly = None
    if contours:
        cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
        step = max(1, len(cnt) // 16)
        poly = [[int(x), int(y)] for x, y in cnt[::step]]
    return Prompt(
        box=box, positive_points=pos, negative_points=neg or None, polygon=poly, polyline=poly
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/arash/datasets/coco_val2017_400_vsx")
    ap.add_argument("--limit", type=int, default=40, help="number of instances")
    ap.add_argument("--max-side", type=int, default=256)
    ap.add_argument("--min-area-frac", type=float, default=0.01)
    ap.add_argument("--methods", default=",".join(list_methods()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    from pycocotools import mask as mask_utils

    ds = Path(args.dataset)
    coco = json.loads((ds / "annotations.json").read_text())
    imgs = {im["id"]: im for im in coco["images"]}
    img_dir = ds / "images"
    rng = np.random.default_rng(args.seed)

    # pick reasonably-sized instances, spread across images
    anns = [a for a in coco["annotations"] if a.get("segmentation") and not a.get("iscrowd")]
    rng.shuffle(anns)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    per_method: dict[str, dict[str, list]] = {
        m: {"iou": [], "biou": [], "lat": []} for m in methods
    }
    n_used = 0
    errors: dict[str, int] = dict.fromkeys(methods, 0)

    for a in anns:
        if n_used >= args.limit:
            break
        im = imgs[a["image_id"]]
        path = img_dir / im["file_name"]
        if not path.exists():
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        H, W = im["height"], im["width"]
        # GT mask
        seg = a["segmentation"]
        if isinstance(seg, list):  # polygon
            rle = mask_utils.frPyObjects(seg, H, W)
            rle = mask_utils.merge(rle)
        else:
            rle = seg if isinstance(seg, dict) else mask_utils.frPyObjects(seg, H, W)
        gt_full = mask_utils.decode(rle).astype("uint8")
        if gt_full.ndim == 3:
            gt_full = gt_full[..., 0]
        if gt_full.sum() < args.min_area_frac * H * W:
            continue
        # resize to keep CPU light
        scale = min(1.0, args.max_side / max(H, W))
        if scale < 1.0:
            nh, nw = round(H * scale), round(W * scale)
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
            gt = (cv2.resize(gt_full, (nw, nh), interpolation=cv2.INTER_NEAREST) > 0).astype(
                "uint8"
            )
        else:
            gt = gt_full
        if gt.sum() < 50:
            continue
        prompt = _derive_prompt(gt, rng)
        n_used += 1
        for m in methods:
            try:
                r = refine(image, prompt, method=m)
                per_method[m]["iou"].append(iou(r.mask, gt))
                per_method[m]["biou"].append(boundary_iou(r.mask, gt))
                per_method[m]["lat"].append(r.latency_ms)
            except Exception:
                errors[m] += 1

    rows = []
    for m in methods:
        s = summarize(per_method[m]["iou"], per_method[m]["biou"], per_method[m]["lat"])
        s["method"] = m
        s["errors"] = errors[m]
        rows.append(s)
    rows.sort(key=lambda r: r["mean_iou"], reverse=True)

    report = {
        "benchmark": "v3_classic_smart_refine",
        "dataset": f"{ds.name}:promptable-derived-from-GT",
        "dataset_kind": "real_images_real_gt_masks_prompts_derived_from_gt",
        "instances_evaluated": n_used,
        "max_side": args.max_side,
        "device": "cpu",
        "prompt_types": ["box", "point", "polygon"],
        "metrics": [
            "mean_iou",
            "boundary_iou",
            "success_rate_at_iou_50",
            "success_rate_at_iou_75",
            "latency_ms",
        ],
        "results": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    import csv as _csv

    with open(args.csv, "w", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "n",
                "mean_iou",
                "boundary_iou",
                "success_rate_at_iou_50",
                "success_rate_at_iou_75",
                "latency_ms_mean",
                "latency_ms_p50",
                "cpu_only_ok",
                "errors",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print(f"instances={n_used}  methods={len(methods)}")
    for r in rows:
        print(
            f"  {r['method']:30s} IoU={r['mean_iou']:.3f} bIoU={r['boundary_iou']:.3f} "
            f"SR@50={r['success_rate_at_iou_50']:.2f} lat={r['latency_ms_mean']:.1f}ms err={r['errors']}"
        )
    print("wrote:", args.out, "|", args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

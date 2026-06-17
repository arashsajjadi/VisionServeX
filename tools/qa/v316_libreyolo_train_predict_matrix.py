#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.16 QA: LibreYOLO train -> reload -> EVAL vs PREDICT mismatch diagnostic.

Trains each LibreYOLO detector on a small *learnable* synthetic YOLO dataset
(distinct class colours at random positions), reloads the checkpoint through the
public VisionServeX API, then compares the libreyolo validator's eval mAP against
what ``predict()`` actually returns. Surfaces:

    eval mAP50 > 0  but  predict boxes == 0   ->  TRAINED_CHECKPOINT_EVAL_PREDICT_MISMATCH
    predict returns a flood of low-conf boxes ->  PREDICT_POSTPROCESS_NMS_MISSING (inspect counts)

Default device CPU (resource-safety). Usage:
    python tools/qa/v316_libreyolo_train_predict_matrix.py --epochs 8 --device cpu \
        --models libreyolo-yolox-s,libreyolo-yolov9-s,libreyolo-rtdetr-r50,libreyolo-dfine-n \
        --out docs/qa/v316_libreyolo_reliability/train_predict_matrix.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import traceback
from pathlib import Path
from typing import Any


def _rng(seed: int):
    # deterministic LCG (Math.random/np.random are fine here; keep it explicit)
    state = {"s": seed}

    def rnd() -> float:
        state["s"] = (1103515245 * state["s"] + 12345) & 0x7FFFFFFF
        return state["s"] / 0x7FFFFFFF

    return rnd


def make_dataset(
    root: Path, *, nc: int = 2, n_train: int = 24, n_val: int = 8, imgsz: int = 320
) -> Path:
    from PIL import Image, ImageDraw

    rnd = _rng(1234)
    colors = [(210, 60, 60), (60, 60, 210), (60, 200, 80)][:nc]
    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cls = i % nc
            bg = (20 + int(rnd() * 40), 30 + int(rnd() * 40), 40 + int(rnd() * 40))
            img = Image.new("RGB", (imgsz, imgsz), bg)
            d = ImageDraw.Draw(img)
            bw = int(imgsz * (0.22 + rnd() * 0.12))
            bh = int(imgsz * (0.22 + rnd() * 0.12))
            x0 = int(rnd() * (imgsz - bw))
            y0 = int(rnd() * (imgsz - bh))
            d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=colors[cls])
            img.save(root / "images" / split / f"{split}_{i:03d}.jpg", quality=92)
            cx = (x0 + bw / 2) / imgsz
            cy = (y0 + bh / 2) / imgsz
            (root / "labels" / split / f"{split}_{i:03d}.txt").write_text(
                f"{cls} {cx:.6f} {cy:.6f} {bw / imgsz:.6f} {bh / imgsz:.6f}\n"
            )
    yaml = root / "data.yaml"
    yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: {nc}\n"
        f"names: {[f'obj{i}' for i in range(nc)]}\n"
    )
    return yaml


def _box_stats(dets) -> dict:
    if not dets:
        return {"n": 0}
    scores = [float(d.score) for d in dets]
    areas = [float((d.box.x2 - d.box.x1) * (d.box.y2 - d.box.y1)) for d in dets]
    return {
        "n": len(dets),
        "score_min": round(min(scores), 4),
        "score_max": round(max(scores), 4),
        "area_min": round(min(areas), 1),
        "area_max": round(max(areas), 1),
        "labels": sorted({d.label for d in dets}),
    }


def test_one(
    model_id: str, yaml: Path, val_img: Path, *, epochs: int, device: str, imgsz: int, project: Path
) -> dict:
    from PIL import Image

    from visionservex.core.model import VisionModel

    row: dict[str, Any] = {"model_id": model_id, "final_verdict": "NOT_RUN", "error": None}
    try:
        # train
        res = VisionModel(model_id).train(
            str(yaml),
            epochs=epochs,
            device=None if device in ("auto", "") else device,
            imgsz=imgsz,
            batch=4,
            project=str(project),
            exist_ok=True,
        )
        row["train_status"] = res.get("status")
        row["eval_mAP50_train"] = (res.get("metrics") or {}).get("best_mAP50")
        ckpt = res.get("best_checkpoint") or res.get("last_checkpoint")
        row["checkpoint"] = ckpt
        row["checkpoint_exists"] = bool(ckpt and Path(ckpt).is_file())
        if not row["checkpoint_exists"]:
            row["final_verdict"] = "NO_CHECKPOINT"
            return row

        # reload via public API
        trained = VisionModel.from_checkpoint(ckpt, model_id=model_id, device=device)

        # eval mAP via libreyolo's own validator on the reloaded model
        eval_map = None
        with contextlib.suppress(Exception):
            vres = trained.engine._model.val(data=str(yaml), verbose=False)
            eval_map = float(vres.get("metrics/mAP50", vres.get("mAP50", 0.0)))
        row["eval_mAP50_reload"] = eval_map

        # predict at two thresholds on a val image
        img = Image.open(val_img).convert("RGB")
        for thr in (0.25, 0.05):
            pred = trained.predict(img, threshold=thr)
            dets = getattr(pred, "detections", None) or []
            row[f"predict_thr_{thr}"] = _box_stats(dets)

        trained.unload()

        n25 = row["predict_thr_0.25"]["n"]
        n05 = row["predict_thr_0.05"]["n"]
        emap = row.get("eval_mAP50_reload") or row.get("eval_mAP50_train") or 0.0
        if emap and emap > 0.05 and n05 == 0:
            row["final_verdict"] = "TRAINED_CHECKPOINT_EVAL_PREDICT_MISMATCH"
        elif n05 > 50:
            row["final_verdict"] = "PREDICT_POSSIBLE_RAW_FLOOD"
        elif n25 >= 1:
            row["final_verdict"] = "PREDICT_OK"
        else:
            row["final_verdict"] = "PREDICT_ZERO_BOXES_LOW_TRAIN"
    except Exception:
        row["final_verdict"] = "CRASHED"
        row["error"] = traceback.format_exc(limit=6)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument(
        "--models",
        default="libreyolo-yolox-s,libreyolo-yolov9-s,libreyolo-rtdetr-r50,libreyolo-dfine-n",
    )
    ap.add_argument("--workdir", default="/home/arash/.cache/vsx_tmp/v316_qa")
    ap.add_argument("--out", default="docs/qa/v316_libreyolo_reliability/train_predict_matrix.json")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    yaml = make_dataset(work / "ds", imgsz=args.imgsz)
    val_img = work / "ds" / "images" / "val" / "val_000.jpg"

    rows = []
    for mid in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== {mid} (epochs={args.epochs}) =====", flush=True)
        row = test_one(
            mid,
            yaml,
            val_img,
            epochs=args.epochs,
            device=args.device,
            imgsz=args.imgsz,
            project=work / "runs",
        )
        print(
            f"  verdict={row['final_verdict']} eval_mAP={row.get('eval_mAP50_reload')} "
            f"n@0.25={row.get('predict_thr_0.25', {}).get('n')} n@0.05={row.get('predict_thr_0.05', {}).get('n')}",
            flush=True,
        )
        if row.get("error"):
            print(row["error"], flush=True)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"epochs": args.epochs, "device": args.device, "rows": rows}, indent=2, default=str
        )
    )
    md = [
        "# v3.16 LibreYOLO train→reload→eval-vs-predict matrix",
        "",
        f"epochs={args.epochs} device={args.device}",
        "",
        "| Model | verdict | eval mAP50 | predict@0.25 | predict@0.05 |",
        "|---|---|--:|--:|--:|",
    ]
    for r in rows:
        md.append(
            f"| {r['model_id']} | {r['final_verdict']} | {r.get('eval_mAP50_reload')} | "
            f"{r.get('predict_thr_0.25', {}).get('n')} | {r.get('predict_thr_0.05', {}).get('n')} |"
        )
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\nSUMMARY:")
    for r in rows:
        print(f"  {r['model_id']:24s} {r['final_verdict']}")


if __name__ == "__main__":
    main()

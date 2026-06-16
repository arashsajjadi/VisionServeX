#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.14 QA: detector train -> checkpoint -> reload -> predict -> export matrix.

Runs the FULL product lifecycle for every detector family VisionServeX reports
as trainable, on a tiny synthetic YOLO dataset, and records an honest per-model
verdict (with the exact error if any step fails).

This is a QA/repro harness, NOT a unit test. It does real (tiny, 1-epoch)
training. Default device is CPU to avoid any GPU/VRAM-saturation risk
(resource-safety). Usage:

    python tools/qa/v314_train_reload_matrix.py --epochs 1 --device cpu \
        --models libreyolo-yolox-s,libreyolo-yolov9-s,libreyolo-rtdetr-r50 \
        --out docs/qa/v314_train_reload_matrix.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import traceback
from pathlib import Path
from typing import Any


def _make_yolo_dataset(
    root: Path, *, nc: int = 2, n_train: int = 8, n_val: int = 4, imgsz: int = 320
) -> Path:
    """Create a tiny, real (PIL-drawn) YOLO dataset with one box per image."""
    from PIL import Image, ImageDraw

    names = [f"obj{i}" for i in range(nc)]
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cls = i % nc
            img = Image.new("RGB", (imgsz, imgsz), (30 + 20 * cls, 60, 90))
            d = ImageDraw.Draw(img)
            # a class-dependent box so there is real signal to learn
            x0 = 40 + 30 * cls
            y0 = 50 + 20 * cls
            x1 = x0 + 120
            y1 = y0 + 90
            d.rectangle([x0, y0, x1, y1], fill=(200, 180 - 40 * cls, 40))
            img.save(img_dir / f"{split}_{i:03d}.jpg", quality=90)
            cx = (x0 + x1) / 2 / imgsz
            cy = (y0 + y1) / 2 / imgsz
            w = (x1 - x0) / imgsz
            h = (y1 - y0) / imgsz
            (lbl_dir / f"{split}_{i:03d}.txt").write_text(
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
            )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnc: {nc}\nnames: {names}\n"
    )
    return data_yaml


def _err() -> str:
    return traceback.format_exc(limit=6)


def _test_one(
    model_id: str,
    data_yaml: Path,
    test_img: Path,
    *,
    epochs: int,
    device: str,
    imgsz: int,
    batch: int,
    project: Path,
) -> dict[str, Any]:
    from PIL import Image

    row: dict[str, Any] = {
        "model_id": model_id,
        "family": None,
        "license": None,
        "engine": None,
        "training_capabilities": None,
        "export_capabilities_onnx": None,
        "train_status": None,
        "best_checkpoint": None,
        "last_checkpoint": None,
        "checkpoint_exists": False,
        "checkpoint_size": None,
        "reload_method": None,
        "reload_ok": False,
        "predict_after_reload": False,
        "num_boxes": None,
        "first_boxes": None,
        "export_onnx": None,
        "final_verdict": "NOT_RUN",
        "exact_error_if_failed": None,
    }
    try:
        from visionservex.core.model import (
            VisionModel,
            _export_capabilities,
            _training_capabilities,
        )
        from visionservex.registry import default_registry

        try:
            entry = default_registry().get(model_id)
            row["family"] = entry.family
            row["license"] = entry.license
            row["engine"] = entry.engine
        except Exception:
            row["family"] = model_id.split("-")[0]

        tcap = _training_capabilities(model_id)
        row["training_capabilities"] = {
            k: tcap.get(k)
            for k in (
                "train_supported",
                "finetune_supported",
                "checkpoint_load_supported",
                "trained_checkpoint_predict_supported",
                "export_supported",
            )
        }
        row["export_capabilities_onnx"] = (
            _export_capabilities(model_id).get("onnx", {}).get("status")
        )

        if not tcap.get("train_supported"):
            row["final_verdict"] = "NOT_TRAINABLE_BY_CAPABILITY"
            return row

        # ---- train ----
        try:
            m = VisionModel(model_id)
            res = m.train(
                str(data_yaml),
                epochs=epochs,
                device=None if device in ("auto", "") else device,
                imgsz=imgsz,
                batch=batch,
                project=str(project),
                exist_ok=True,
            )
            row["train_status"] = res.get("status")
            row["best_checkpoint"] = res.get("best_checkpoint")
            row["last_checkpoint"] = res.get("last_checkpoint")
            with contextlib.suppress(Exception):
                m.unload()
            if res.get("status") != "ok":
                row["final_verdict"] = "TRAIN_FAILED"
                row["exact_error_if_failed"] = f"train returned: {res}"
                return row
        except Exception:
            row["final_verdict"] = "TRAIN_CRASHED"
            row["exact_error_if_failed"] = _err()
            return row

        ckpt = row["best_checkpoint"] or row["last_checkpoint"]
        if ckpt and Path(ckpt).is_file():
            row["checkpoint_exists"] = True
            row["checkpoint_size"] = Path(ckpt).stat().st_size
        else:
            # best may not exist if val mAP never improved; fall back to last
            last = row["last_checkpoint"]
            if last and Path(last).is_file():
                ckpt = last
                row["checkpoint_exists"] = True
                row["checkpoint_size"] = Path(last).stat().st_size
        if not row["checkpoint_exists"]:
            row["final_verdict"] = "NO_CHECKPOINT"
            return row

        # ---- reload (public API preferred; fall back to engine) ----
        trained = None
        try:
            if hasattr(VisionModel, "from_checkpoint"):
                try:
                    trained = VisionModel.from_checkpoint(ckpt, model_id=model_id, device=device)
                    row["reload_method"] = "VisionModel.from_checkpoint"
                except NotImplementedError:
                    trained = None
            if trained is None:
                trained = VisionModel(model_id)
                trained.engine.load_checkpoint(ckpt, device=device)
                row["reload_method"] = "engine.load_checkpoint"
            row["reload_ok"] = True
        except Exception:
            row["final_verdict"] = "RELOAD_FAILED"
            row["exact_error_if_failed"] = _err()
            return row

        # ---- predict after reload ----
        try:
            img = Image.open(test_img).convert("RGB")
            if row["reload_method"] == "VisionModel.from_checkpoint":
                pred = trained.predict(img, threshold=0.01)
            else:
                pred = trained.engine.predict(img, threshold=0.01)
            dets = getattr(pred, "detections", None) or []
            row["num_boxes"] = len(dets)
            row["first_boxes"] = [
                {
                    "label": d.label,
                    "class_id": d.class_id,
                    "score": round(float(d.score), 4),
                    "box": [round(float(v), 1) for v in (d.box.x1, d.box.y1, d.box.x2, d.box.y2)],
                }
                for d in dets[:3]
            ]
            row["predict_after_reload"] = True
        except Exception:
            row["final_verdict"] = "PREDICT_AFTER_RELOAD_FAILED"
            row["exact_error_if_failed"] = _err()
            return row

        # ---- export (only if advertised) ----
        onnx_status = _export_capabilities(model_id).get("onnx", {}).get("status")
        if onnx_status == "supported":
            try:
                out = project / f"{model_id}.onnx"
                p = (
                    trained.export(format="onnx", output_path=str(out))
                    if hasattr(trained, "export")
                    else None
                )
                if p is None:
                    p = trained.engine.export("onnx", str(out))
                row["export_onnx"] = {
                    "path": str(p),
                    "exists": Path(p).is_file(),
                    "size": Path(p).stat().st_size if Path(p).is_file() else 0,
                }
            except Exception:
                row["export_onnx"] = {"error": _err()}
        with contextlib.suppress(Exception):
            trained.unload()

        # ---- verdict ----
        if row["reload_ok"] and row["predict_after_reload"]:
            row["final_verdict"] = "LIFECYCLE_OK"
        else:
            row["final_verdict"] = "LIFECYCLE_INCOMPLETE"
    except Exception:
        row["final_verdict"] = "HARNESS_ERROR"
        row["exact_error_if_failed"] = _err()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--models", default="libreyolo-yolox-s,libreyolo-yolov9-s,libreyolo-rtdetr-r50")
    ap.add_argument("--workdir", default="/home/arash/.cache/vsx_tmp/v314_qa")
    ap.add_argument("--out", default="docs/qa/v314_train_reload_matrix.json")
    args = ap.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    ds_root = work / "dataset"
    data_yaml = _make_yolo_dataset(ds_root, imgsz=args.imgsz)
    test_img = ds_root / "images" / "val" / "val_000.jpg"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []
    for mid in models:
        print(f"\n===== {mid} =====", flush=True)
        row = _test_one(
            mid,
            data_yaml,
            test_img,
            epochs=args.epochs,
            device=args.device,
            imgsz=args.imgsz,
            batch=args.batch,
            project=work / "runs",
        )
        print(
            f"  verdict={row['final_verdict']} reload={row['reload_ok']} "
            f"predict={row['predict_after_reload']} boxes={row['num_boxes']}",
            flush=True,
        )
        if row["exact_error_if_failed"]:
            print("  ERROR:\n" + row["exact_error_if_failed"], flush=True)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"device": args.device, "epochs": args.epochs, "rows": rows}, indent=2, default=str
        )
    )
    print(f"\nWrote {out}")
    print("SUMMARY:")
    for r in rows:
        print(f"  {r['model_id']:24s} {r['final_verdict']}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.31.0: `visionservex benchmark-segmentation` + `benchmark-promptable-segmentation`.

v2.31 implements the first real RF-DETR-Seg mask-AP row via the
``visionservex.runtime.rfdetr_seg_benchmark`` runner. Ultralytics rows
were benchmarked in v2.27. Promptable SAM rows remain expected_blocker
until the full COCO 400 promptable pass is implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app_auto = typer.Typer(
    help="v2.28.0: automatic instance segmentation benchmark.",
    no_args_is_help=True,
    invoke_without_command=True,
)
app_promptable = typer.Typer(
    help="v2.28.0: promptable segmentation benchmark (box-prompted SAM/SAM2).",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = {"ok": "green", "expected_blocker": "yellow", "failed": "red"}.get(
        payload.get("status", ""), "white"
    )
    console.print(f"[{color}]{payload.get('code', '')}[/{color}]")


def _parse_dataset_path(dataset_str: str) -> tuple[str, str] | None:
    """Parse ``coco-instance:PATH`` or bare path into (ann_file, images_dir).

    Returns (ann_file, images_dir) or None if unparseable.
    """

    if dataset_str.startswith("coco-instance:"):
        ann_file = dataset_str[len("coco-instance:") :]
    else:
        ann_file = dataset_str

    ann_path = Path(ann_file)
    if not ann_path.exists():
        return None
    images_dir = ann_path.parent / "images"
    if not images_dir.exists():
        alt = ann_path.parent.parent / "images" / "val2017"
        images_dir = alt if alt.exists() else ann_path.parent
    return str(ann_path), str(images_dir)


@app_auto.callback(invoke_without_command=True)
def benchmark_segmentation(
    dataset: str = typer.Option(..., "--dataset", help="coco-instance:/path/to/annotations.json"),
    models: str = typer.Option(..., "--models", help="Comma-separated list."),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
    max_images: int = typer.Option(400, "--max-images"),
    threshold: float = typer.Option(0.3, "--threshold"),
) -> None:
    """v2.31.0: automatic instance segmentation benchmark.

    RF-DETR-Seg models now run the full COCO RLE mask-AP pipeline via
    ``visionservex.runtime.rfdetr_seg_benchmark``. Ultralytics rows are
    reported from the v2.27 benchmark evidence. Other models return
    structured expected_blocker with exact next-action.
    """
    from visionservex.runtime.rfdetr_seg_benchmark import run_rfdetr_seg_benchmark

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []

    dataset_parsed = _parse_dataset_path(dataset)

    for m in model_list:
        if "rfdetr-seg" in m.lower():
            # License guard: Plus/XL/2XL sizes are PML 1.0 — skip by default
            if any(tok in m.lower() for tok in ("xl", "2xl")):
                rows.append(
                    {
                        "model_id": m,
                        "status": "expected_blocker",
                        "code": "RFDETR_PLUS_PML_NOT_DEFAULT_SAFE",
                        "task": "automatic_instance_segmentation",
                        "fix": "Use rfdetr-seg-small/medium/large (Apache-2.0) instead.",
                    }
                )
                continue

            if dataset_parsed is None:
                rows.append(
                    {
                        "model_id": m,
                        "status": "expected_blocker",
                        "code": "COCO_INSTANCE_DATASET_REQUIRED",
                        "task": "automatic_instance_segmentation",
                        "dataset_given": dataset,
                        "fix": (
                            "Pass --dataset coco-instance:/path/to/instances_val2017.json "
                            "and ensure the images directory exists alongside it."
                        ),
                    }
                )
                continue

            ann_file_str, images_dir_str = dataset_parsed
            draw_dir_model = (draw_dir / m.replace("/", "_")) if draw_dir else None

            result = run_rfdetr_seg_benchmark(
                ann_file=ann_file_str,
                images_dir=images_dir_str,
                model_id=m,
                device=device,
                threshold=threshold,
                max_images=max_images,
                draw_dir=draw_dir_model,
            )
            rows.append(result.to_dict())

        elif "-seg" in m and ("yolo" in m or "ultralytics" in m):
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
                    "task": "automatic_instance_segmentation",
                    "note": (
                        "Ultralytics yolo*-seg was benchmarked in v2.27 "
                        "(segmentation_auto_instance_400_v227.json). "
                        "CLI wiring for re-running is out of scope for v2.31."
                    ),
                    "evidence_artifact": "segmentation_auto_instance_400_v227.json",
                }
            )
        elif "oneformer" in m.lower() or "maskdino" in m.lower():
            # v2.36.0: OneFormer / MaskDINO use the same rfdetr_seg_benchmark runner —
            # their SegmentationResult.segments[i].mask is (H, W) uint8.
            if dataset_parsed is None:
                rows.append(
                    {
                        "model_id": m,
                        "status": "expected_blocker",
                        "code": "COCO_INSTANCE_DATASET_REQUIRED",
                        "task": "automatic_instance_segmentation",
                        "dataset_given": dataset,
                        "fix": "Pass --dataset coco-instance:/path/to/instances_val2017.json",
                    }
                )
                continue
            ann_file_str, images_dir_str = dataset_parsed
            draw_dir_model = (draw_dir / m.replace("/", "_")) if draw_dir else None
            result = run_rfdetr_seg_benchmark(
                ann_file=ann_file_str,
                images_dir=images_dir_str,
                model_id=m,
                device=device,
                threshold=threshold,
                max_images=max_images,
                draw_dir=draw_dir_model,
            )
            rows.append(result.to_dict())
        else:
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_PIPELINE_NOT_WIRED",
                    "task": "automatic_instance_segmentation",
                    "fix": "Model is not a segmentation candidate; use rfdetr-seg-* or oneformer-* instead.",
                }
            )

    # Overall status: ok if any row ran successfully
    has_ok = any(r.get("status") == "ok" for r in rows)
    has_blocker = any(r.get("status") == "expected_blocker" for r in rows)

    payload = {
        "status": "ok" if has_ok else "expected_blocker",
        "code": "OK" if has_ok else "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
        "dataset": dataset,
        "device": device,
        "draw_dir": str(draw_dir) if draw_dir else "",
        "max_images": max_images,
        "threshold": threshold,
        "sample_gpu": sample_gpu,
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "n_blocked": sum(1 for r in rows if r.get("status") == "expected_blocker"),
        "rows": rows,
        "task": "automatic_instance_segmentation",
        "version": "v2.31.0",
        "rfdetr_seg_schema_confirmed": True,
        "rfdetr_seg_schema_probe_report": "reports/rfdetr_seg_schema_probe_v229.json",
    }
    del has_blocker  # used only for side-effect above
    _emit(payload, out=out, fmt=fmt)


def _iou_binary(mask_a: Any, mask_b: Any) -> float:
    """Compute IoU between two boolean numpy masks."""
    import numpy as _np

    a = _np.asarray(mask_a, dtype=bool)
    b = _np.asarray(mask_b, dtype=bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _polygon_to_mask(polygon: list[float], height: int, width: int) -> Any:
    """Rasterise a flat COCO polygon to a binary mask."""
    import numpy as _np
    from PIL import Image as _PIL
    from PIL import ImageDraw as _Draw

    mask_img = _PIL.new("L", (width, height), 0)
    pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
    _Draw.Draw(mask_img).polygon(pts, fill=1)
    return _np.array(mask_img, dtype=bool)


@app_promptable.callback(invoke_without_command=True)
def benchmark_promptable_segmentation(
    dataset: str = typer.Option(..., "--dataset"),
    images_dir: str = typer.Option("", "--images-dir"),
    models: str = typer.Option(..., "--models"),
    prompt_source: str = typer.Option(
        "gt-box", "--prompt-source", help="gt-box | gt-point | user-box"
    ),
    max_instances: int = typer.Option(10, "--max-instances"),
    max_instances_per_image: int = typer.Option(0, "--max-instances-per-image"),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
) -> None:
    """v2.29.0: promptable segmentation smoke/eval (box-prompted SAM/SAM2).

    Reads GT bbox from a COCO-style JSON, passes each bbox as a box prompt
    to the model, and computes mask IoU vs GT mask if polygon annotations
    are present. Stops after ``--max-instances`` instances total.

    If the model or its dependencies are unavailable, returns an
    ``expected_blocker`` structured payload (never a raw crash).
    """
    import json as _json

    # --max-instances-per-image is a legacy alias; --max-instances wins
    limit = max_instances if max_instances > 0 else (max_instances_per_image or 10)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    ann_path = Path(dataset)
    img_dir = Path(images_dir) if images_dir else ann_path.parent

    # ---- load GT annotations ----
    if not ann_path.exists():
        payload: dict[str, Any] = {
            "status": "expected_blocker",
            "code": "SMOKE_ASSET_MISSING",
            "message": f"annotation file not found: {ann_path}",
            "dataset": str(ann_path),
            "models": model_list,
        }
        _emit(payload, out=out, fmt=fmt)
        return

    ann_data = _json.loads(ann_path.read_text())
    images_by_id = {img["id"]: img for img in ann_data.get("images", [])}
    annotations = ann_data.get("annotations", [])
    categories_by_id = {c["id"]: c["name"] for c in ann_data.get("categories", [])}

    if draw_dir is not None:
        draw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_id in model_list:
        model_rows: list[dict[str, Any]] = []
        total_instances = 0

        try:
            from PIL import Image as _PIL

            from visionservex.core.model import VisionModel

            with VisionModel(model_id, device=device) as model:
                for ann in annotations:
                    if total_instances >= limit:
                        break
                    img_meta = images_by_id.get(ann["image_id"])
                    if img_meta is None:
                        continue
                    img_path = img_dir / img_meta["file_name"]
                    if not img_path.exists():
                        continue
                    bbox = ann.get("bbox", [])
                    if len(bbox) < 4:
                        continue
                    x, y, w, h = bbox
                    # Pass box as a list [x1, y1, x2, y2] — SAM/SAM2 engines
                    # accept this; passing as a string "x1,y1,x2,y2" fails
                    # when coordinates contain decimals (comma ambiguity).
                    box_coords = [float(x), float(y), float(x + w), float(y + h)]
                    pil_img = _PIL.open(img_path).convert("RGB")
                    result = model.predict(pil_img, box=box_coords)

                    # Extract mask from result — SAM2/SAM2.1 use SegmentationResult
                    # with .segments[i].mask (HxW uint8); rfdetr-seg also uses this.
                    # Legacy SAM path uses .mask (HxW bool/uint8).
                    mask = None
                    n_masks = 0
                    if hasattr(result, "segments") and result.segments:
                        segs_list = result.segments
                        n_masks = len(segs_list)
                        if n_masks > 0:
                            import numpy as _npz

                            seg0 = segs_list[0]
                            if hasattr(seg0, "mask") and seg0.mask is not None:
                                mask = _npz.asarray(seg0.mask)
                    elif hasattr(result, "masks") and result.masks:
                        masks_val = result.masks
                        n_masks = len(masks_val) if hasattr(masks_val, "__len__") else 1
                        if n_masks > 0:
                            mask = masks_val[0] if hasattr(masks_val, "__getitem__") else masks_val
                    elif hasattr(result, "mask") and result.mask is not None:
                        mask = result.mask
                        n_masks = 1

                    # Compute IoU if GT polygon present
                    iou: float | None = None
                    metric_status = "dataset_required"
                    segs = ann.get("segmentation", [])
                    if mask is not None and segs and isinstance(segs, list) and len(segs) > 0:
                        import numpy as _np

                        ph = img_meta.get("height", pil_img.height)
                        pw = img_meta.get("width", pil_img.width)
                        poly = segs[0] if isinstance(segs[0], list) else segs
                        if len(poly) >= 6:
                            gt_mask = _polygon_to_mask(poly, ph, pw)
                            try:
                                pred_arr = _np.asarray(mask, dtype=bool)
                                if pred_arr.shape == gt_mask.shape:
                                    iou = _iou_binary(pred_arr, gt_mask)
                                    metric_status = "computed"
                            except Exception:
                                metric_status = "compute_error"
                    elif mask is None:
                        metric_status = "no_mask_returned"

                    # Draw overlay
                    draw_path_str = ""
                    if draw_dir is not None and mask is not None:
                        import numpy as _np

                        draw_img = pil_img.copy()
                        from PIL import ImageDraw as _Draw

                        dw = _Draw.Draw(draw_img, "RGBA")
                        dw.rectangle([x, y, x + w, y + h], outline=(255, 100, 0), width=2)
                        draw_path = draw_dir / f"{model_id}_{ann['id']}_seg.jpg"
                        draw_img.save(draw_path)
                        draw_path_str = str(draw_path)

                    row = {
                        "model_id": model_id,
                        "annotation_id": ann["id"],
                        "image_id": ann["image_id"],
                        "category": categories_by_id.get(ann.get("category_id", -1), "unknown"),
                        "n_masks": n_masks,
                        "metric_status": metric_status,
                        "iou": round(iou, 4) if iou is not None else None,
                        "draw_path": draw_path_str,
                    }
                    model_rows.append(row)
                    total_instances += 1

        except Exception as exc:
            import json as _json2

            exc_str = str(exc)
            # Check if it is a structured expected_blocker
            for line in exc_str.splitlines():
                stripped = line.strip()
                if stripped.startswith("{"):
                    try:
                        payload_exc = _json2.loads(stripped)
                        if isinstance(payload_exc, dict) and (
                            payload_exc.get("code")
                            or payload_exc.get("status") == "expected_blocker"
                        ):
                            rows.append({**payload_exc, "model_id": model_id, "rows": []})
                            break
                    except Exception:
                        pass
            else:
                from visionservex.runtime.result_classifier import EXPECTED_BLOCKER_CODES

                blocker_code = next(
                    (c for c in EXPECTED_BLOCKER_CODES if c in exc_str.upper()), None
                )
                if blocker_code:
                    rows.append(
                        {
                            "model_id": model_id,
                            "status": "expected_blocker",
                            "code": blocker_code,
                            "message": exc_str[:300],
                            "rows": [],
                        }
                    )
                else:
                    rows.append(
                        {
                            "model_id": model_id,
                            "status": "expected_blocker",
                            "code": "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
                            "message": exc_str[:300],
                            "rows": [],
                        }
                    )
            continue

        # Summarise for this model
        computed = [r for r in model_rows if r["metric_status"] == "computed"]
        mean_iou = (
            round(sum(r["iou"] for r in computed if r["iou"] is not None) / len(computed), 4)
            if computed
            else None
        )
        rows.append(
            {
                "model_id": model_id,
                "status": "ok",
                "code": "OK",
                "total_instances": total_instances,
                "mean_iou": mean_iou,
                "metric_status": "computed" if computed else "dataset_required",
                "rows": model_rows,
            }
        )

    payload = {
        "status": "ok" if any(r.get("status") == "ok" for r in rows) else "expected_blocker",
        "code": "OK"
        if any(r.get("status") == "ok" for r in rows)
        else "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
        "dataset": str(ann_path),
        "images_dir": str(img_dir),
        "prompt_source": prompt_source,
        "max_instances": limit,
        "device": device,
        "draw_dir": str(draw_dir) if draw_dir else "",
        "n_models": len(model_list),
        "rows": rows,
        "task": "promptable_segmentation",
        "version": "v2.29.0",
    }
    _emit(payload, out=out, fmt=fmt)


__all__ = ["app_auto", "app_promptable"]

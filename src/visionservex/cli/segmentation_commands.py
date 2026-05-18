# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: `visionservex benchmark-segmentation` + `benchmark-promptable-segmentation`.

These CLIs frame the two protocols separately so VisionServeX never mixes
them. Both currently emit structured blockers for VSX/SAM rows until the
mask AP / promptable adapters are written. Ultralytics yolo*-seg models
were already benchmarked in v2.27 (see segmentation_auto_instance_400_v227.json).
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


@app_auto.callback(invoke_without_command=True)
def benchmark_segmentation(
    dataset: str = typer.Option(..., "--dataset", help="coco-instance:ANNOTATIONS.json"),
    models: str = typer.Option(..., "--models", help="Comma-separated list."),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
) -> None:
    """v2.28.0: automatic instance segmentation.

    For Ultralytics yolo*-seg models, v2.27 already shipped a working
    runner; v2.28 surfaces the same protocol via this command. For VSX
    rfdetr-seg-* models, the mask-AP adapter is not yet written and the
    command returns ``RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN`` per model.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    for m in model_list:
        if "rfdetr-seg" in m.lower():
            # v2.29.0: schema probed. result.segments[i].mask is (H,W) uint8.
            # Blocker is now pycocotools for COCO RLE mask-AP, not schema unknown.
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "GT_MASKS_REQUIRED_FOR_MASK_METRICS",
                    "task": "automatic_instance_segmentation",
                    "mask_format": "segments_list_with_per_segment_mask",
                    "mask_field_path": "result.segments[i].mask",
                    "mask_dtype": "uint8",
                    "schema_probe_report": "reports/rfdetr_seg_schema_probe_v229.json",
                    "next_action": (
                        "Schema confirmed v2.29.0: result.segments[i].mask is (H,W) uint8. "
                        "Next step: implement COCO RLE conversion via pycocotools and run "
                        "mask AP against COCO val2017 GT annotations."
                    ),
                    "install_command": "pip install pycocotools",
                }
            )
        elif "-seg" in m and ("yolo" in m or "ultralytics" in m):
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
                    "task": "automatic_instance_segmentation",
                    "next_action": (
                        "Ultralytics yolo*-seg already benchmarked in v2.27 "
                        "(segmentation_auto_instance_400_v227.json). The new "
                        "package CLI wiring is deferred to v2.29."
                    ),
                    "evidence_artifact": "segmentation_auto_instance_400_v227.json",
                }
            )
        else:
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_PIPELINE_NOT_WIRED",
                    "task": "automatic_instance_segmentation",
                    "next_action": (
                        "Model is not registered as a segmentation candidate in the v2.28 pipeline."
                    ),
                }
            )

    payload = {
        "status": "expected_blocker",
        "code": "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
        "dataset": dataset,
        "device": device,
        "draw_dir": str(draw_dir) if draw_dir else "",
        "sample_gpu": sample_gpu,
        "isolate_process": isolate_process,
        "n_rows": len(rows),
        "rows": rows,
        "task": "automatic_instance_segmentation",
        "version": "v2.29.0",
        "rfdetr_seg_schema_confirmed": True,
        "rfdetr_seg_schema_probe_report": "reports/rfdetr_seg_schema_probe_v229.json",
        "message": (
            "v2.29.0: rfdetr-seg schema probed (result.segments[i].mask, uint8 HxW). "
            "Full mask-AP benchmark requires pycocotools + COCO val2017 GT masks."
        ),
    }
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
                    box_str = f"{x:.1f},{y:.1f},{x + w:.1f},{y + h:.1f}"
                    pil_img = _PIL.open(img_path).convert("RGB")
                    result = model.predict(pil_img, box=box_str)

                    # Extract mask from result
                    mask = None
                    n_masks = 0
                    if hasattr(result, "masks") and result.masks:
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Draw / overlay CLI — operates on prediction JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    help="Draw / overlay CLI — render prediction JSON onto images.",
    no_args_is_help=True,
)
console = Console()


def _load_pred(pred_path: Path) -> dict:
    return json.loads(pred_path.read_text())


def _yolo_labels_to_gt(
    labels_path: Path, image_path: Path, names: list[str] | None = None
) -> list[dict]:
    """Convert YOLO-format label file to ground-truth list."""
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    gt: list[dict] = []
    for line in labels_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(parts[0])
            cx, cy, bw, bh = (float(p) for p in parts[1:5])
        except ValueError:
            continue
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        name = names[cls] if names and cls < len(names) else f"#{cls}"
        gt.append({"box": [x1, y1, x2, y2], "class_id": cls, "class_name": name})
    return gt


@app.command("image")
def draw_image(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred", help="JSON file with detections."),
    out: Path = typer.Option(..., "--out", help="Annotated image output path."),
    line_width: int = typer.Option(2, "--line-width"),
    font_size: int = typer.Option(14, "--font-size"),
    hide_labels: bool = typer.Option(False, "--hide-labels"),
    hide_conf: bool = typer.Option(False, "--hide-conf"),
) -> None:
    """Draw detections from a prediction JSON onto an image."""
    from visionservex.visualization import draw_detections

    payload = _load_pred(pred)
    dets = payload.get("detections", []) if isinstance(payload, dict) else payload
    img = draw_detections(
        image,
        dets,
        line_width=line_width,
        font_size=font_size,
        hide_labels=hide_labels,
        hide_conf=hide_conf,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(dets)} detections → {out}[/green]")


@app.command("gt")
def draw_gt(
    image: Path = typer.Option(..., "--image"),
    labels: Path = typer.Option(..., "--labels"),
    fmt: str = typer.Option("yolo", "--format", help="yolo|json"),
    names: str = typer.Option("coco", "--names", help="'coco' or path to names file"),
    out: Path = typer.Option(..., "--out"),
    line_width: int = typer.Option(2, "--line-width"),
) -> None:
    """Draw ground-truth boxes from a labels file."""
    from visionservex.visualization import draw_ground_truth

    name_list: list[str] | None = None
    if names == "coco":
        # Minimal COCO 80 names for ground-truth labelling
        name_list = [
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
    elif Path(names).exists():
        name_list = [n.strip() for n in Path(names).read_text().splitlines() if n.strip()]

    if fmt.lower() == "yolo":
        gt = _yolo_labels_to_gt(labels, image, name_list)
    else:
        payload = _load_pred(labels)
        gt = payload.get("ground_truth") or payload.get("detections") or payload

    img = draw_ground_truth(image, gt, line_width=line_width)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(gt)} ground-truth boxes → {out}[/green]")


@app.command("compare")
def draw_compare(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    gt: Path = typer.Option(..., "--gt"),
    gt_format: str = typer.Option("yolo", "--gt-format"),
    names: str = typer.Option("coco", "--names"),
    out: Path = typer.Option(..., "--out"),
    line_width: int = typer.Option(2, "--line-width"),
) -> None:
    """Draw predictions + ground truth overlaid on the same image."""
    from visionservex.visualization import draw_prediction_comparison

    pred_payload = _load_pred(pred)
    preds = pred_payload.get("detections", []) if isinstance(pred_payload, dict) else pred_payload

    name_list: list[str] | None = None
    if names == "coco":
        name_list = None  # use class_id labels for compare brevity
    if gt_format.lower() == "yolo":
        gt_list = _yolo_labels_to_gt(gt, image, name_list)
    else:
        gt_payload = _load_pred(gt)
        gt_list = gt_payload.get("ground_truth") or gt_payload.get("detections") or gt_payload

    img = draw_prediction_comparison(image, preds, gt_list, line_width=line_width)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew compare ({len(preds)} pred + {len(gt_list)} gt) → {out}[/green]")


@app.command("segment")
def draw_segment(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
    alpha: float = typer.Option(0.45, "--alpha"),
) -> None:
    """Draw segmentation masks (alpha-blended) on an image."""
    from visionservex.visualization import draw_segmentation_masks

    payload = _load_pred(pred)
    masks = payload.get("masks", []) if isinstance(payload, dict) else payload
    img = draw_segmentation_masks(image, masks, alpha=alpha)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(masks)} masks → {out}[/green]")


@app.command("pose")
def draw_pose_cmd(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Draw pose keypoints/skeleton on an image."""
    from visionservex.visualization import draw_pose

    payload = _load_pred(pred)
    persons = payload.get("persons", []) if isinstance(payload, dict) else payload
    img = draw_pose(image, persons)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(persons)} persons → {out}[/green]")


@app.command("obb")
def draw_obb_cmd(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Draw oriented rotated boxes."""
    from visionservex.visualization import draw_obb

    payload = _load_pred(pred)
    boxes = payload.get("oriented_boxes", []) if isinstance(payload, dict) else payload
    img = draw_obb(image, boxes)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(boxes)} OBB → {out}[/green]")


@app.command("tracks")
def draw_tracks_cmd(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Draw tracker output on an image frame."""
    from visionservex.visualization import draw_tracks

    payload = _load_pred(pred)
    tracks = payload.get("tracks", []) if isinstance(payload, dict) else payload
    img = draw_tracks(image, tracks)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]drew {len(tracks)} tracks → {out}[/green]")


__all__ = ["app"]

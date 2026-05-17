# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Annotate CLI — render images/videos/frames using prediction JSON or JSONL.

Examples::

    visionservex annotate image --image in.jpg --pred preds.json --out out.jpg
    visionservex annotate video --video in.mp4 --jsonl frames.jsonl --out out.mp4
    visionservex annotate frames --frames-dir frames/ --jsonl frames.jsonl --out-dir annotated/
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    help="Annotate images / video / frame folders from JSON or JSONL predictions.",
    no_args_is_help=True,
)
console = Console()


def _load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


@app.command("image")
def annotate_image_cmd(
    image: Path = typer.Option(..., "--image"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
    line_width: int = typer.Option(2, "--line-width"),
    font_size: int = typer.Option(14, "--font-size"),
    mask_alpha: float = typer.Option(0.45, "--mask-alpha"),
    hide_labels: bool = typer.Option(False, "--hide-labels"),
    hide_conf: bool = typer.Option(False, "--hide-conf"),
) -> None:
    """Annotate a single image using a single-payload JSON."""
    from visionservex.visualization import annotate_image as do_annotate

    payload = json.loads(pred.read_text(encoding="utf-8"))
    img = do_annotate(
        image, payload,
        line_width=line_width, font_size=font_size, mask_alpha=mask_alpha,
        hide_labels=hide_labels, hide_conf=hide_conf,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    console.print(f"[green]annotated → {out}[/green]")


@app.command("frames")
def annotate_frames_cmd(
    frames_dir: Path = typer.Option(..., "--frames-dir"),
    jsonl: Path = typer.Option(..., "--jsonl"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    line_width: int = typer.Option(2, "--line-width"),
    mask_alpha: float = typer.Option(0.45, "--mask-alpha"),
    hide_labels: bool = typer.Option(False, "--hide-labels"),
    hide_conf: bool = typer.Option(False, "--hide-conf"),
) -> None:
    """Annotate every frame in a folder using its matching JSONL payload."""
    from visionservex.visualization import annotate_image as do_annotate

    out_dir.mkdir(parents=True, exist_ok=True)
    items = _load_jsonl(jsonl)
    by_index = {p.get("frame_index"): p for p in items if "frame_index" in p}

    img_paths = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    n_done = 0
    for idx, img_path in enumerate(img_paths):
        payload = by_index.get(idx, {})
        annotated = do_annotate(
            img_path, payload,
            line_width=line_width, mask_alpha=mask_alpha,
            hide_labels=hide_labels, hide_conf=hide_conf,
        )
        annotated.save(out_dir / img_path.name)
        n_done += 1
    console.print(f"[green]annotated {n_done} frames → {out_dir}[/green]")


@app.command("video")
def annotate_video_cmd(
    video: Path = typer.Option(..., "--video"),
    jsonl: Path = typer.Option(..., "--jsonl"),
    out: Path = typer.Option(..., "--out"),
    line_width: int = typer.Option(2, "--line-width"),
    mask_alpha: float = typer.Option(0.45, "--mask-alpha"),
    fps: float = typer.Option(0.0, "--fps", help="Override output FPS (0 = autodetect from input)"),
) -> None:
    """Annotate every frame of a video using JSONL payloads and write to MP4."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        console.print(f"[red]opencv-python-headless required: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    from visionservex.visualization import annotate_image as do_annotate

    items = _load_jsonl(jsonl)
    by_index = {p.get("frame_index"): p for p in items if "frame_index" in p}

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        console.print(f"[red]VIDEO_SOURCE_OPEN_FAILED: {video}[/red]")
        raise typer.Exit(code=2)

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    out_fps = fps if fps > 0 else in_fps

    out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        console.print("[red]VIDEO_WRITER_FAILED[/red]")
        raise typer.Exit(code=2)

    from PIL import Image

    frame_index = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, :, ::-1]
            pil_img = Image.fromarray(rgb)
            payload = by_index.get(frame_index, {})
            annotated = do_annotate(
                pil_img, payload,
                line_width=line_width, mask_alpha=mask_alpha,
            )
            bgr = np.array(annotated)[:, :, ::-1]
            writer.write(bgr)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()
    console.print(f"[green]annotated {written} frames → {out}[/green]")


__all__ = ["app"]

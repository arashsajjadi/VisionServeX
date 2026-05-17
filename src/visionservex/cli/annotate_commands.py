# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Annotate CLI — render images/videos/frames using prediction JSON/JSONL or live inference.

Examples::

    # Inference-then-annotate (notebook contract):
    visionservex annotate image --model dfine-s-o365-coco --image in.jpg --task detect --out out.jpg --json-out preds.json
    visionservex annotate video --model mock-detect --video in.mp4 --task detect --out out.mp4 --json-out preds.jsonl --max-frames 10

    # Post-annotation from pre-computed predictions:
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


def _run_inference(model_id: str, image_path: Path, task: str | None, prompt: str | None) -> dict:
    """Run inference using the VisionServeX predict pipeline and return the result dict."""
    from visionservex.core.runner import load_model

    kwargs: dict = {}
    if task:
        kwargs["task"] = task
    if prompt:
        kwargs["prompt"] = prompt

    model = load_model(model_id, auto_pull=False)
    result = model.predict(image_path, **kwargs)
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    return {"raw": str(result)}


@app.command("image")
def annotate_image_cmd(
    image: Path = typer.Option(..., "--image"),
    pred: Path | None = typer.Option(
        None, "--pred", help="Pre-computed prediction JSON (mutually exclusive with --model)."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Run inference with this model, then annotate."
    ),
    task: str | None = typer.Option(
        None, "--task", help="Task hint for inference (detect|open-vocab|segment)."
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt for open-vocab models."),
    out: Path = typer.Option(..., "--out"),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Save raw prediction JSON to this path."
    ),
    line_width: int = typer.Option(2, "--line-width"),
    font_size: int = typer.Option(14, "--font-size"),
    mask_alpha: float = typer.Option(0.45, "--mask-alpha"),
    hide_labels: bool = typer.Option(False, "--hide-labels"),
    hide_conf: bool = typer.Option(False, "--hide-conf"),
) -> None:
    """Annotate a single image using pre-computed JSON or live inference.

    Two modes:
      --pred preds.json   → annotate from existing predictions (original mode)
      --model MODEL_ID    → run inference then annotate (notebook mode)
    """
    if model is None and pred is None:
        console.print("[red]ERROR: provide either --model MODEL_ID or --pred PRED_JSON[/red]")
        raise typer.Exit(code=2)
    if model is not None and pred is not None:
        console.print("[red]ERROR: --model and --pred are mutually exclusive[/red]")
        raise typer.Exit(code=2)

    if model is not None:
        # Inference-then-annotate mode
        try:
            payload = _run_inference(model, image, task, prompt)
        except Exception as exc:
            err_msg = str(exc)
            structured = {
                "status": "expected_blocker",
                "code": "MODEL_LOAD_FAILED",
                "message": err_msg,
                "model_id": model,
            }
            if json_out:
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(json.dumps(structured, indent=2))
            console.print(f"[yellow]MODEL_LOAD_FAILED: {err_msg[:200]}[/yellow]")
            raise typer.Exit(code=1) from exc
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(payload, indent=2))
    else:
        payload = json.loads(pred.read_text(encoding="utf-8"))  # type: ignore[union-attr]
        if json_out:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(payload, indent=2))

    from visionservex.visualization import annotate_image as do_annotate

    try:
        img = do_annotate(
            image,
            payload,
            line_width=line_width,
            font_size=font_size,
            mask_alpha=mask_alpha,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        console.print(f"[green]annotated → {out}[/green]")
    except Exception as exc:
        console.print(f"[yellow]DRAW_FAILED: {exc}[/yellow]")
        raise typer.Exit(code=1) from exc


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
        p
        for p in frames_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    n_done = 0
    for idx, img_path in enumerate(img_paths):
        payload = by_index.get(idx, {})
        annotated = do_annotate(
            img_path,
            payload,
            line_width=line_width,
            mask_alpha=mask_alpha,
            hide_labels=hide_labels,
            hide_conf=hide_conf,
        )
        annotated.save(out_dir / img_path.name)
        n_done += 1
    console.print(f"[green]annotated {n_done} frames → {out_dir}[/green]")


@app.command("video")
def annotate_video_cmd(
    video: Path | None = typer.Option(None, "--video"),
    jsonl: Path | None = typer.Option(
        None, "--jsonl", help="Pre-computed JSONL (mutually exclusive with --model)."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Run inference per-frame with this model (notebook mode)."
    ),
    task: str | None = typer.Option(
        None, "--task", help="Task hint for inference (detect|open-vocab)."
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt for open-vocab models."),
    out: Path = typer.Option(..., "--out"),
    json_out: Path | None = typer.Option(
        None, "--json-out", help="Save per-frame JSONL predictions."
    ),
    max_frames: int | None = typer.Option(
        None, "--max-frames", help="Stop after this many frames."
    ),
    tracker: str | None = typer.Option(
        None, "--tracker", help="Tracker name (simple-iou, bytetrack). Informational only."
    ),
    line_width: int = typer.Option(2, "--line-width"),
    mask_alpha: float = typer.Option(0.45, "--mask-alpha"),
    fps: float = typer.Option(0.0, "--fps", help="Override output FPS (0 = autodetect from input)"),
) -> None:
    """Annotate every frame of a video using JSONL payloads or live inference, and write to MP4.

    Two modes:
      --jsonl frames.jsonl  → annotate from existing predictions (original mode)
      --model MODEL_ID      → run per-frame inference then annotate (notebook mode)
    """
    if video is None:
        console.print("[red]ERROR: --video is required[/red]")
        raise typer.Exit(code=2)
    if model is None and jsonl is None:
        console.print("[red]ERROR: provide either --model MODEL_ID or --jsonl PRED_JSONL[/red]")
        raise typer.Exit(code=2)
    if model is not None and jsonl is not None:
        console.print("[red]ERROR: --model and --jsonl are mutually exclusive[/red]")
        raise typer.Exit(code=2)

    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as exc:
        console.print(f"[red]opencv-python-headless required: {exc}[/red]")
        raise typer.Exit(code=2) from exc

    from visionservex.visualization import annotate_image as do_annotate

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

    # Resolve prediction source
    if jsonl is not None:
        items = _load_jsonl(jsonl)
        by_index: dict[int, dict] = {p.get("frame_index"): p for p in items if "frame_index" in p}
    else:
        by_index = {}

    # Load model if in inference mode
    infer_model = None
    if model is not None:
        try:
            from visionservex.core.runner import load_model as _load_model

            infer_model = _load_model(model, auto_pull=False)
        except Exception as exc:
            cap.release()
            writer.release()
            structured = {
                "status": "expected_blocker",
                "code": "MODEL_LOAD_FAILED",
                "message": str(exc),
                "model_id": model,
            }
            if json_out:
                json_out.parent.mkdir(parents=True, exist_ok=True)
                json_out.write_text(json.dumps(structured, indent=2))
            console.print(f"[yellow]MODEL_LOAD_FAILED: {exc}[/yellow]")
            raise typer.Exit(code=1) from exc

    jsonl_lines: list[str] = []
    frame_index = 0
    written = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, :, ::-1]
            pil_img = Image.fromarray(rgb)

            if infer_model is not None:
                infer_kwargs: dict = {}
                if task:
                    infer_kwargs["task"] = task
                if prompt:
                    infer_kwargs["prompt"] = prompt
                try:
                    result = infer_model.predict(pil_img, **infer_kwargs)
                    payload: dict = (
                        result.to_dict()
                        if hasattr(result, "to_dict")
                        else (result if isinstance(result, dict) else {"raw": str(result)})
                    )
                except Exception:
                    payload = {}
                payload["frame_index"] = frame_index
            else:
                payload = by_index.get(frame_index, {})

            if json_out:
                jsonl_lines.append(json.dumps(payload))

            try:
                annotated = do_annotate(
                    pil_img, payload, line_width=line_width, mask_alpha=mask_alpha
                )
                bgr = np.array(annotated)[:, :, ::-1]
            except Exception:
                bgr = frame
            writer.write(bgr)
            written += 1
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if json_out and jsonl_lines:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text("\n".join(jsonl_lines) + "\n")

    console.print(f"[green]annotated {written} frames → {out}[/green]")


__all__ = ["app"]

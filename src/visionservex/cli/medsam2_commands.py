# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Experimental MedSAM2 runtime CLI: `visionservex medical medsam2 ...`.

Drives the real in-process MedSAM2 adapter (research-only, non-commercial). Every
command emits machine-readable JSON and maps :class:`MedSAM2RuntimeError` to a
structured payload with a non-zero exit — never a stack trace. Heavy deps are
imported lazily inside the adapter, so this module stays import-light.

Commands: doctor | load | segment | batch | benchmark-smoke
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Experimental MedSAM2 runtime (research-only).", no_args_is_help=True)
console = Console()

_DISCLAIMER = "Research/education only — NOT for diagnosis. MedSAM2 weights are non-commercial."


def _print(payload: dict, json_: bool) -> None:
    if json_:
        print(json.dumps(payload, indent=2, default=str))
    else:
        console.print_json(json.dumps(payload, default=str))


def _parse_boxes(box: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for raw in box:
        parts = [float(x) for x in raw.split(",")]
        if len(parts) != 4:
            raise typer.BadParameter(f"box must be x1,y1,x2,y2 (got {raw!r})")
        out.append(parts)
    return out


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Report MedSAM2 runtime availability (no heavy import, no download)."""
    from visionservex.medical.medsam2_runtime import medsam2_doctor

    _print(medsam2_doctor(), json_)


@app.command("load")
def load(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="MedSAM2 .pt checkpoint."),
    config: str = typer.Option("", "--config", help="Optional Hydra config path."),
    device: str = typer.Option("cpu", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Load a MedSAM2 checkpoint and report structured load status."""
    from visionservex.medical.medsam2_runtime import MedSAM2RuntimeError, load_medsam2_runtime

    try:
        rt = load_medsam2_runtime(checkpoint, config=config or None, device=device)
    except MedSAM2RuntimeError as exc:
        _print(exc.to_dict(), json_)
        raise typer.Exit(3) from exc
    _print(rt.info(), json_)


@app.command("segment")
def segment(
    image: Path = typer.Argument(..., help="2D PNG/JPEG (or NIfTI middle slice)."),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    config: str = typer.Option("", "--config"),
    device: str = typer.Option("cpu", "--device"),
    box: list[str] = typer.Option([], "--box", help="x1,y1,x2,y2 (repeat for multiple)."),
    point: str = typer.Option("", "--point", help="x,y point prompt."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run real MedSAM2 2D segmentation; save mask PNG(s) + metadata JSON."""
    from visionservex.medical.medsam2_runtime import (
        MedSAM2RuntimeError,
        load_2d_input,
        load_medsam2_runtime,
        segment_2d,
    )

    try:
        boxes = _parse_boxes(box) or None
        points = [[float(v) for v in point.split(",")]] if point else None
        img = load_2d_input(image)
        rt = load_medsam2_runtime(checkpoint, config=config or None, device=device)
        result = segment_2d(rt, img, boxes=boxes, points=points)
    except MedSAM2RuntimeError as exc:
        _print(exc.to_dict(), json_)
        raise typer.Exit(3) from exc

    import numpy as np
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    masks_saved = []
    for i, seg in enumerate(result.segments):
        mp = out / f"{stem}_medsam2_mask_{i:03d}.png"
        if mp.exists() and not overwrite:
            raise typer.Exit(  # honest: never silently overwrite
                _err(f"output exists: {mp} (pass --overwrite)", "OUTPUT_EXISTS", json_)
            )
        Image.fromarray((np.asarray(seg.mask) * 255).astype(np.uint8)).save(mp)
        masks_saved.append(
            {
                "mask_path": str(mp),
                "score": seg.score,
                "box": [seg.box.x1, seg.box.y1, seg.box.x2, seg.box.y2],
            }
        )
    payload = {
        "status": "ok" if masks_saved else "no_masks",
        "model_id": "medsam2",
        "engine": "medsam2_runtime",
        "input": str(image),
        "input_mode": result.metadata.get("input_mode"),
        "device": device,
        "n_masks": len(masks_saved),
        "masks_saved": masks_saved,
        "commercial_safe": False,
        "research_only": True,
        "disclaimer": _DISCLAIMER,
    }
    meta = out / f"{stem}_medsam2.json"
    meta.write_text(json.dumps(payload, indent=2, default=str))
    payload["metadata_path"] = str(meta)
    _print(payload, json_)


@app.command("batch")
def batch(
    inputs: Path = typer.Argument(..., help="Text file: one image path per line."),
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    config: str = typer.Option("", "--config"),
    device: str = typer.Option("cpu", "--device"),
    out: Path = typer.Option(..., "--out"),
    workers: int = typer.Option(1, "--workers", help="GPU default MUST stay 1."),
    continue_on_error: bool = typer.Option(False, "--continue-on-error"),
    strict: bool = typer.Option(False, "--strict"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Batch MedSAM2 2D segmentation over a list of images (order-preserving)."""
    from visionservex.medical.medsam2_batch import run_medsam2_batch
    from visionservex.medical.medsam2_runtime import MedSAM2RuntimeError

    paths = [ln.strip() for ln in inputs.read_text().splitlines() if ln.strip()]
    try:
        manifest = run_medsam2_batch(
            paths,
            checkpoint=checkpoint,
            config=config or None,
            device=device,
            out_dir=out,
            workers=workers,
            continue_on_error=continue_on_error and not strict,
            overwrite=overwrite,
        )
    except MedSAM2RuntimeError as exc:
        _print(exc.to_dict(), json_)
        raise typer.Exit(3) from exc
    _print(manifest, json_)


@app.command("benchmark-smoke")
def benchmark_smoke(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    config: str = typer.Option("", "--config"),
    device: str = typer.Option("cpu", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Contract-only timing/memory smoke (NOT an accuracy benchmark)."""
    from visionservex.medical.medsam2_runtime import (
        MedSAM2RuntimeError,
        load_medsam2_runtime,
        segment_2d,
    )

    try:
        import numpy as np

        rt = load_medsam2_runtime(checkpoint, config=config or None, device=device)
        img = np.zeros((256, 256, 3), np.uint8)
        img[64:192, 64:192] = 200
        t0 = time.perf_counter()
        result = segment_2d(rt, img, boxes=[[64, 64, 192, 192]])
        infer_s = round(time.perf_counter() - t0, 3)
    except MedSAM2RuntimeError as exc:
        _print(exc.to_dict(), json_)
        raise typer.Exit(3) from exc
    payload = {
        "status": "ok",
        "model_id": "medsam2",
        "device": device,
        "load_time_seconds": rt.load_time_seconds,
        "infer_time_seconds": infer_s,
        "input_size": [256, 256],
        "n_masks": len(result.segments),
        "note": "contract smoke only — NO accuracy/Dice claim without ground truth.",
        "commercial_safe": False,
        "disclaimer": _DISCLAIMER,
    }
    _print(payload, json_)


def _err(msg: str, code: str, json_: bool) -> int:
    _print({"status": "failed", "code": code, "message": msg}, json_)
    return 3


__all__ = ["app"]

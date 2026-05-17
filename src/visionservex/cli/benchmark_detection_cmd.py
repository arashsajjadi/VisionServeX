# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""benchmark-detection / benchmark-ultralytics commands.

These are thin, honest wrappers around the existing competitiveness benchmark
engine in ``benchmark_commands._run_ap_benchmark``. They expose simpler
ergonomics for the common cases:

- ``benchmark-detection``: one VisionServeX detection model, AP50/AP75/mAP50:95
  on a labelled dataset (no synthetic fake-data mode).
- ``benchmark-ultralytics``: same dataset, runs both the chosen VisionServeX
  model and an Ultralytics YOLO baseline side-by-side.

Both commands require ``--dataset yolo:<path>`` or ``--dataset coco-json:...``.
Synthetic-only mode is intentionally NOT exposed — claiming AP without ground
truth would be dishonest.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app_det = typer.Typer(help="Detection benchmark: AP50/AP75/mAP50:95 + latency p50/p95.")
app_ult = typer.Typer(help="Detection benchmark against an Ultralytics YOLO baseline.")
console = Console()


@app_det.callback(invoke_without_command=True)
def benchmark_detection(
    model: str = typer.Option(..., "--model", help="VisionServeX detection model id"),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="yolo:<path>  or  coco-json:<images_dir>:<ann_file>",
    ),
    max_images: int = typer.Option(50, "--max-images", min=1, max=10_000),
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Single-model AP benchmark on a labelled dataset."""
    if dataset.startswith("synthetic"):
        console.print("[red]benchmark-detection requires a labelled dataset.[/red]")
        console.print("Use --dataset yolo:<path> or --dataset coco-json:<images>:<ann>.")
        raise typer.Exit(2)

    from visionservex.cli.benchmark_commands import (
        _run_ap_benchmark,
        _run_ap_benchmark_isolated,
    )

    if isolate_process:
        _run_ap_benchmark_isolated([model], dataset, max_images, device, out, json_)
    else:
        _run_ap_benchmark([model], dataset, max_images, device, out, json_)


@app_ult.callback(invoke_without_command=True)
def benchmark_ultralytics(
    model: str = typer.Option(..., "--model", help="VisionServeX detection model id"),
    yolo: str = typer.Option(
        "yolo11n",
        "--yolo",
        help="Ultralytics baseline (yolo11n, yolo11s, yolov8n, ...)",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="yolo:<path>  or  coco-json:<images_dir>:<ann_file>",
    ),
    max_images: int = typer.Option(50, "--max-images", min=1, max=10_000),
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Detection benchmark, head-to-head vs an Ultralytics YOLO baseline."""
    if dataset.startswith("synthetic"):
        console.print("[red]benchmark-ultralytics requires a labelled dataset.[/red]")
        raise typer.Exit(2)

    from visionservex.cli.benchmark_commands import (
        _run_ap_benchmark,
        _run_ap_benchmark_isolated,
    )

    ids = [model, f"ultralytics:{yolo}"]
    if isolate_process:
        _run_ap_benchmark_isolated(ids, dataset, max_images, device, out, json_)
    else:
        _run_ap_benchmark(ids, dataset, max_images, device, out, json_)


__all__ = ["app_det", "app_ult"]

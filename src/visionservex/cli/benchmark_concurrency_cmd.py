# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.18.0: shared-model concurrent request benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="v2.18.0: shared-model concurrent request benchmark.")
console = Console()


@app.callback(invoke_without_command=True)
def benchmark_concurrency(
    dataset: str = typer.Option(..., "--dataset", help="yolo:<path> or coco-json:<dir>:<ann>"),
    models: str = typer.Option(..., "--models"),
    max_images: int = typer.Option(20, "--max-images"),
    device: str = typer.Option("cuda", "--device"),
    require_gpu: bool = typer.Option(False, "--require-gpu"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    gpu_sample_interval: float = typer.Option(0.5, "--gpu-sample-interval"),
    concurrency: str = typer.Option("1,2", "--concurrency"),
    request_mode: str = typer.Option(
        "shared-model", "--request-mode", help="shared-model | separate-process"
    ),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Run shared-model concurrency benchmark."""
    from visionservex.runtime.concurrency import run_concurrency_benchmark
    from visionservex.runtime.evaluation import load_coco_json, load_yolo_format

    json_mode = fmt == "json"

    # Parse dataset
    samples = None
    if dataset.startswith("yolo:"):
        ypath = Path(dataset[5:])
        if not ypath.exists():
            payload = {
                "status": "failed",
                "code": "DATASET_NOT_FOUND",
                "dataset": str(ypath),
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=2))
            if json_mode:
                typer.echo(json.dumps(payload, indent=2))
            raise typer.Exit(2)
        samples, _ = load_yolo_format(ypath, max_images=max_images)
    elif dataset.startswith("coco-json:"):
        parts = dataset[10:].split(":", 1)
        images_dir = Path(parts[0])
        ann_file = Path(parts[1])
        samples, _ = load_coco_json(images_dir, ann_file, max_images=max_images)
    else:
        ypath = Path(dataset)
        samples, _ = load_yolo_format(ypath, max_images=max_images)

    if not samples:
        payload = {"status": "failed", "code": "NO_IMAGES_IN_DATASET", "dataset": dataset}
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_mode:
            typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(2)

    image_paths = [Path(s.image_path) for s in samples]
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    levels = [int(x.strip()) for x in concurrency.split(",") if x.strip()]

    all_runs: dict = {
        "benchmark_type": "concurrency_v2180",
        "dataset": dataset,
        "n_images": len(image_paths),
        "device_requested": device,
        "require_gpu": require_gpu,
        "concurrency_levels": levels,
        "request_mode": request_mode,
        "models": [],
    }
    for mid in model_ids:
        if not json_mode:
            console.print(f"  concurrency benchmark [cyan]{mid}[/cyan] @ levels {levels} ...")
        result = run_concurrency_benchmark(
            model_id=mid,
            image_paths=image_paths,
            device=device,
            require_gpu=require_gpu,
            concurrency_levels=levels,
            request_mode=request_mode,
            sample_gpu=sample_gpu,
            gpu_sample_interval=gpu_sample_interval,
        )
        all_runs["models"].append(result)
        if not json_mode:
            for r in result.get("runs", []):
                console.print(
                    f"    c={r['concurrency']}: {r['throughput_req_per_sec']:.1f} req/s, "
                    f"p50={r['latency_ms_p50']}ms p95={r['latency_ms_p95']}ms "
                    f"vram_peak={r.get('vram_peak_gb', '-')}GB"
                )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_runs, indent=2, default=str))
    if json_mode:
        typer.echo(json.dumps(all_runs, indent=2, default=str))


__all__ = ["app"]

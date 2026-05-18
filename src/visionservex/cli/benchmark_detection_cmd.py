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
    model: str = typer.Option(
        None,
        "--model",
        help="VisionServeX detection model id (legacy single-model flag).",
    ),
    models: str = typer.Option(
        "",
        "--models",
        help=(
            "Comma-separated detection model IDs. v2.17.0: mocks and aliases are "
            "rejected at the entry point unless --include-mocks or --include-aliases."
        ),
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="yolo:<path>  or  coco-json:<images_dir>:<ann_file>",
    ),
    max_images: int = typer.Option(50, "--max-images", min=1, max=10_000),
    device: str = typer.Option("auto", "--device"),
    require_gpu: bool = typer.Option(
        False,
        "--require-gpu",
        help=(
            "v2.17.0: fail with GPU_REQUIRED_NOT_USED if device_actual is not "
            "CUDA. Prevents silent CPU fallback during benchmarking."
        ),
    ),
    sample_gpu: bool = typer.Option(
        False,
        "--sample-gpu",
        help="v2.17.0: spawn an nvidia-smi sampler during the benchmark.",
    ),
    gpu_sample_interval: float = typer.Option(
        0.5,
        "--gpu-sample-interval",
        help="Seconds between GPU samples.",
    ),
    include_mocks: bool = typer.Option(
        False,
        "--include-mocks",
        help="Allow mock-* models to enter the benchmark (off by default).",
    ),
    include_aliases: bool = typer.Option(
        False,
        "--include-aliases",
        help="Keep alias rows instead of collapsing to canonical (off by default).",
    ),
    out: Path | None = typer.Option(None, "--out"),
    report_md: Path | None = typer.Option(
        None, "--report-md", help="Optional Markdown report path."
    ),
    draw_dir: Path | None = typer.Option(
        None, "--draw-dir", help="Reserved for future per-model draw output."
    ),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text or json (notebook contract)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Single-model or multi-model AP benchmark on a labelled dataset.

    v2.17.0: when ``--require-gpu`` or ``--sample-gpu`` is passed, the
    persistent-load benchmark from ``runtime.persistent_benchmark`` is
    used (load_count=1 per model, timing breakdown, optional GPU sampler).
    Without those flags the legacy code path is preserved for backwards
    compatibility.
    """
    if dataset.startswith("synthetic"):
        console.print("[red]benchmark-detection requires a labelled dataset.[/red]")
        console.print("Use --dataset yolo:<path> or --dataset coco-json:<images>:<ann>.")
        raise typer.Exit(2)

    # Resolve model list
    raw_ids: list[str] = []
    if model:
        raw_ids.append(model)
    if models:
        raw_ids.extend([m.strip() for m in models.split(",") if m.strip()])
    if not raw_ids:
        console.print("[red]Provide --model ID or --models a,b,c[/red]")
        raise typer.Exit(2)

    json_mode = json_ or fmt == "json"

    # v2.17.0: filter mocks/aliases at the entry point.
    from visionservex.runtime.leaderboard import (
        MOCK_MODEL_IDS,
        canonicalize_model_id,
    )

    filtered_ids: list[str] = []
    skipped: list[dict] = []
    seen_canonical: dict[str, str] = {}
    for mid in raw_ids:
        if mid in MOCK_MODEL_IDS or mid.startswith("mock-"):
            if include_mocks:
                filtered_ids.append(mid)
            else:
                skipped.append({"model_id": mid, "reason": "MOCK_MODEL"})
            continue
        canonical, is_alias = canonicalize_model_id(mid)
        if is_alias and not include_aliases:
            skipped.append({"model_id": mid, "reason": "ALIAS_DUPLICATE", "alias_of": canonical})
            continue
        if canonical in seen_canonical and not include_aliases:
            skipped.append(
                {
                    "model_id": mid,
                    "reason": "ALIAS_DUPLICATE",
                    "alias_of": seen_canonical[canonical],
                }
            )
            continue
        seen_canonical[canonical] = mid
        filtered_ids.append(mid)

    if not filtered_ids:
        payload = {
            "status": "failed",
            "code": "ALL_MODELS_REJECTED",
            "skipped": skipped,
            "message": (
                "Every requested model was rejected (mock/alias). Pass --include-mocks "
                "or --include-aliases to override."
            ),
        }
        import json as _json

        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_json.dumps(payload, indent=2))
        if json_mode:
            typer.echo(_json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['code']}[/red]: {payload['message']}")
            for s in skipped:
                console.print(f"  [dim]{s['model_id']}[/dim] — {s['reason']}")
        raise typer.Exit(2)

    if skipped and not json_mode:
        console.print(f"[yellow]Skipped {len(skipped)} model(s):[/yellow]")
        for s in skipped:
            console.print(f"  [dim]{s['model_id']}[/dim] — {s['reason']}")

    # New v2.17.0 path: persistent-load benchmark when GPU flags are set.
    if require_gpu or sample_gpu or fmt == "json":
        _run_v217_persistent_benchmark(
            model_ids=filtered_ids,
            dataset_str=dataset,
            max_images=max_images,
            device=device,
            require_gpu=require_gpu,
            sample_gpu=sample_gpu,
            gpu_sample_interval=gpu_sample_interval,
            out=out,
            report_md=report_md,
            skipped=skipped,
            json_mode=json_mode,
        )
        return

    # Legacy path (preserved for backwards-compat)
    from visionservex.cli.benchmark_commands import (
        _run_ap_benchmark,
        _run_ap_benchmark_isolated,
    )

    if isolate_process:
        _run_ap_benchmark_isolated(filtered_ids, dataset, max_images, device, out, json_mode)
    else:
        _run_ap_benchmark(filtered_ids, dataset, max_images, device, out, json_mode)


def _run_v217_persistent_benchmark(
    *,
    model_ids: list[str],
    dataset_str: str,
    max_images: int,
    device: str,
    require_gpu: bool,
    sample_gpu: bool,
    gpu_sample_interval: float,
    out: Path | None,
    report_md: Path | None,
    skipped: list[dict],
    json_mode: bool,
) -> None:
    """v2.17.0 path — persistent load + GPU enforcement + timing breakdown."""
    import json as _json

    from visionservex.runtime.evaluation import load_coco_json, load_yolo_format
    from visionservex.runtime.persistent_benchmark import (
        run_persistent_detection_benchmark,
    )

    # Dataset
    samples = None
    dataset_name = "unknown"
    if dataset_str.startswith("yolo:"):
        ypath = Path(dataset_str[5:])
        if not ypath.exists():
            payload = {
                "status": "failed",
                "code": "DATASET_NOT_FOUND",
                "dataset": str(ypath),
            }
            if out:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(_json.dumps(payload, indent=2))
            if json_mode:
                typer.echo(_json.dumps(payload, indent=2))
            else:
                console.print(f"[red]DATASET_NOT_FOUND[/red]: {ypath}")
            raise typer.Exit(2)
        samples, _ = load_yolo_format(ypath, max_images=max_images)
        dataset_name = f"yolo:{ypath.name}"
    elif dataset_str.startswith("coco-json:"):
        parts = dataset_str[10:].split(":", 1)
        if len(parts) != 2:
            console.print("[red]coco-json format: coco-json:<images_dir>:<ann_file>[/red]")
            raise typer.Exit(2)
        images_dir = Path(parts[0])
        ann_file = Path(parts[1])
        samples, _ = load_coco_json(images_dir, ann_file, max_images=max_images)
        dataset_name = f"coco-json:{ann_file.stem}"
    else:
        ypath = Path(dataset_str)
        if not ypath.exists():
            console.print(f"[red]Dataset not found: {ypath}[/red]")
            raise typer.Exit(2)
        samples, _ = load_yolo_format(ypath, max_images=max_images)
        dataset_name = ypath.name

    if not samples:
        payload = {
            "status": "failed",
            "code": "NO_IMAGES_IN_DATASET",
            "dataset": dataset_str,
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_json.dumps(payload, indent=2))
        if json_mode:
            typer.echo(_json.dumps(payload, indent=2))
        else:
            console.print("[red]No images in dataset.[/red]")
        raise typer.Exit(2)

    rows: list[dict] = []
    for mid in model_ids:
        if not json_mode:
            console.print(f"  benchmarking [cyan]{mid}[/cyan] ...", end=" ")
        row = run_persistent_detection_benchmark(
            model_id=mid,
            samples=samples,
            device_requested=device,
            require_gpu=require_gpu,
            sample_gpu=sample_gpu,
            gpu_sample_interval=gpu_sample_interval,
            dataset_name=dataset_name,
        )
        rows.append(row)
        if not json_mode:
            if row["status"] == "ok":
                console.print(
                    f"[green]ok[/green] AP50={row['ap50']} mAP50:95={row['map50_95']} "
                    f"p50={row['total_latency_ms_p50']}ms load_count={row['load_count']}"
                )
            else:
                console.print(f"[red]{row['code']}[/red] {row['errors'][:1]}")

    summary = {
        "benchmark_type": "persistent_detection_v2170",
        "dataset": dataset_name,
        "n_models": len(model_ids),
        "n_images_per_model_requested": max_images,
        "device_requested": device,
        "require_gpu": require_gpu,
        "sample_gpu": sample_gpu,
        "skipped_at_entry": skipped,
        "models": rows,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps(summary, indent=2, default=str))
    if report_md:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(_md_report_v217(summary))
    if json_mode:
        typer.echo(_json.dumps(summary, indent=2, default=str))


def _md_report_v217(summary: dict) -> str:
    lines = [
        f"# benchmark-detection (v2.17.0 persistent path) — {summary['dataset']}",
        "",
        f"- device_requested: `{summary['device_requested']}`",
        f"- require_gpu: `{summary['require_gpu']}`",
        f"- sample_gpu: `{summary['sample_gpu']}`",
        f"- n_models: {summary['n_models']}",
        f"- n_images_per_model_requested: {summary['n_images_per_model_requested']}",
        "",
        "## Models",
        "",
        "| model_id | status | code | device_actual | load_count | n_eval | ap50 | mAP50:95 | p50_ms | GPU mean % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in summary["models"]:
        gpu = r.get("gpu_utilization") or {}
        lines.append(
            f"| `{r['model_id']}` | {r['status']} | {r['code']} | {r['device_actual']} | "
            f"{r['load_count']} | {r['n_images_evaluated']} | {r['ap50']} | {r['map50_95']} | "
            f"{r['total_latency_ms_p50']} | {gpu.get('utilization_mean', '-')} |"
        )
    return "\n".join(lines) + "\n"


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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Open-vocabulary detection benchmark.

Runs a prompt-set over an image folder for a chosen open-vocab detector
(OWLv2 / Grounding DINO / Florence-2 OD) and emits structured per-model
metrics. The metrics chosen here are honest for the open-vocab setting:

- mean #detections per image per prompt
- mean top-1 score per prompt
- fraction of images with at least one match per prompt
- per-prompt latency

We deliberately do NOT report mAP unless the caller provides labelled
ground-truth in a separate path. If GT path is missing, the benchmark
honestly reports retrieval-style metrics only.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Open-vocab detection benchmark (OWLv2 / Grounding DINO / Florence-2 OD).")
console = Console()


@dataclass
class PromptMetrics:
    prompt: str
    n_images: int
    n_images_with_match: int
    mean_detections: float
    mean_top1_score: float
    p50_latency_ms: float
    p95_latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelBenchmark:
    model_id: str
    threshold: float
    total_images: int
    prompts: list[PromptMetrics]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prompts"] = [p if isinstance(p, dict) else p.to_dict() for p in self.prompts]
        return d


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[k]


def _benchmark_one_model(
    model_id: str,
    *,
    images: list[Path],
    prompts: list[str],
    threshold: float,
    auto_pull: bool,
) -> ModelBenchmark:
    from visionservex import VisionModel

    model = VisionModel(model_id, auto_pull=auto_pull)
    prompt_metrics: list[PromptMetrics] = []

    for prompt in prompts:
        per_image_n: list[int] = []
        per_image_top1: list[float] = []
        per_image_latency: list[float] = []
        images_with_match = 0
        for img_path in images:
            from PIL import Image

            img = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            result = model.predict(img, prompt=prompt, threshold=threshold)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            detections = getattr(result, "detections", []) or []
            per_image_n.append(len(detections))
            per_image_latency.append(dt_ms)
            if detections:
                images_with_match += 1
                top1 = max(float(d.score) for d in detections)
                per_image_top1.append(top1)
            else:
                per_image_top1.append(0.0)

        prompt_metrics.append(
            PromptMetrics(
                prompt=prompt,
                n_images=len(images),
                n_images_with_match=images_with_match,
                mean_detections=statistics.fmean(per_image_n) if per_image_n else 0.0,
                mean_top1_score=statistics.fmean(per_image_top1) if per_image_top1 else 0.0,
                p50_latency_ms=_quantile(per_image_latency, 0.5),
                p95_latency_ms=_quantile(per_image_latency, 0.95),
            )
        )

    return ModelBenchmark(
        model_id=model_id,
        threshold=threshold,
        total_images=len(images),
        prompts=prompt_metrics,
    )


@app.callback(invoke_without_command=True)
def benchmark_open_vocab(
    images_dir: Path = typer.Argument(..., help="Folder of evaluation images."),
    prompts: str = typer.Option(
        "person, car, dog, traffic light",
        "--prompts",
        help="Comma-separated free-form text queries.",
    ),
    models: str = typer.Option(
        "owlv2-base-patch16",
        "--models",
        help="Comma-separated open-vocab model ids.",
    ),
    threshold: float = typer.Option(0.1, "--threshold"),
    max_images: int = typer.Option(50, "--max-images"),
    out: Path = typer.Option(None, "--out", help="Write JSON to this path."),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run an open-vocab detection benchmark with prompt-set metrics."""
    if not images_dir.exists() or not images_dir.is_dir():
        console.print(f"[red]images_dir not found:[/red] {images_dir}")
        raise typer.Exit(2)

    image_paths = sorted(
        f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )[:max_images]
    if not image_paths:
        console.print("[red]No images found in folder.[/red]")
        raise typer.Exit(2)

    prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
    model_ids = [m.strip() for m in models.split(",") if m.strip()]

    benchmarks: list[ModelBenchmark] = []
    for mid in model_ids:
        bm = _benchmark_one_model(
            mid,
            images=image_paths,
            prompts=prompt_list,
            threshold=threshold,
            auto_pull=auto_pull,
        )
        benchmarks.append(bm)

    payload = {
        "benchmark": "open_vocab",
        "images_dir": str(images_dir),
        "n_images": len(image_paths),
        "threshold": threshold,
        "models": [bm.to_dict() for bm in benchmarks],
        "notes": (
            "Retrieval-style metrics only. mAP requires labelled ground truth; "
            "use benchmark-competitiveness for AP."
        ),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if json_:
        print(json.dumps(payload, indent=2))
        return

    for bm in benchmarks:
        console.rule(f"[bold cyan]{bm.model_id}")
        table = Table(show_header=True)
        table.add_column("Prompt")
        table.add_column("Hits/Imgs", no_wrap=True)
        table.add_column("Mean Dets", no_wrap=True)
        table.add_column("Mean Top1", no_wrap=True)
        table.add_column("P50 ms", no_wrap=True)
        table.add_column("P95 ms", no_wrap=True)
        for p in bm.prompts:
            table.add_row(
                p.prompt,
                f"{p.n_images_with_match}/{p.n_images}",
                f"{p.mean_detections:.2f}",
                f"{p.mean_top1_score:.2f}",
                f"{p.p50_latency_ms:.1f}",
                f"{p.p95_latency_ms:.1f}",
            )
        console.print(table)


__all__ = ["ModelBenchmark", "PromptMetrics", "app"]

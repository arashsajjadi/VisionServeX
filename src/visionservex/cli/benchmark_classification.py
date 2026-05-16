# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""benchmark-classification — real image classification benchmark.

Evaluates one or more classification models on a folder-based or CSV dataset.

Dataset format (folder):
    dataset/
        cat/*.jpg
        dog/*.jpg
        car/*.jpg

Dataset format (CSV):
    image_path,label
    /path/img1.jpg,cat
    /path/img2.jpg,dog

Metrics:
- top1_accuracy
- top5_accuracy (if k >= 5)
- per_class_accuracy
- latency_p50_ms
- latency_p95_ms
- throughput_images_per_sec
- peak_ram_mb
- failures

Does NOT report mAP. Correct metric for classification is accuracy/top-k.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Classification model benchmark (top-k accuracy, per-class, latency).",
)
console = Console()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ClassificationBenchmarkResult:
    model_id: str
    dataset: str
    n_images: int
    n_failures: int
    top1_accuracy: float
    top5_accuracy: float
    per_class_accuracy: dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_images_per_sec: float
    peak_ram_mb: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_dataset(dataset_spec: str, max_images: int) -> list[tuple[Path, str]]:
    """Load (image_path, label) pairs from folder: or csv: spec."""
    if dataset_spec.startswith("folder:"):
        root = Path(dataset_spec[len("folder:") :])
        if not root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        pairs: list[tuple[Path, str]] = []
        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in _IMAGE_EXTS:
                    pairs.append((img, label))
        return pairs[:max_images]

    if dataset_spec.startswith("csv:"):
        csv_path = Path(dataset_spec[len("csv:") :])
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        pairs = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                pairs.append((Path(row["image_path"]), row["label"]))
        return pairs[:max_images]

    raise ValueError(
        f"Unsupported dataset spec {dataset_spec!r}. Use folder:/path or csv:/path.csv"
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[k]


def _benchmark_model(
    model_id: str,
    dataset: list[tuple[Path, str]],
    *,
    top_k: int,
    device: str,
    auto_pull: bool,
) -> ClassificationBenchmarkResult:
    import psutil
    from PIL import Image

    from visionservex import VisionModel

    try:
        model = VisionModel(model_id, auto_pull=auto_pull)
        model._ensure_loaded()
    except Exception as exc:
        return ClassificationBenchmarkResult(
            model_id=model_id,
            dataset="",
            n_images=0,
            n_failures=0,
            top1_accuracy=0.0,
            top5_accuracy=0.0,
            per_class_accuracy={},
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            throughput_images_per_sec=0.0,
            peak_ram_mb=None,
            error=str(exc)[:200],
        )

    process = psutil.Process()
    peak_ram = 0.0
    latencies: list[float] = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    failures = 0
    per_class_correct: dict[str, int] = {}
    per_class_total: dict[str, int] = {}

    for img_path, true_label in dataset:
        try:
            img = Image.open(img_path).convert("RGB")
            t0 = time.perf_counter()
            result = model.predict(img, top_k=top_k)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt_ms)
            ram_mb = process.memory_info().rss / 1024**2
            peak_ram = max(peak_ram, ram_mb)

            top_labels = [lbl for lbl, _score in (result.top_k or [])]
            per_class_total[true_label] = per_class_total.get(true_label, 0) + 1

            if top_labels and top_labels[0] == true_label:
                correct_top1 += 1
                per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1
            if true_label in top_labels[:5]:
                correct_top5 += 1
            total += 1
        except Exception:
            failures += 1

    total_time = sum(latencies)
    return ClassificationBenchmarkResult(
        model_id=model_id,
        dataset="",
        n_images=total,
        n_failures=failures,
        top1_accuracy=correct_top1 / max(total, 1),
        top5_accuracy=correct_top5 / max(total, 1),
        per_class_accuracy={
            cls: per_class_correct.get(cls, 0) / max(n, 1) for cls, n in per_class_total.items()
        },
        latency_p50_ms=_quantile(latencies, 0.5),
        latency_p95_ms=_quantile(latencies, 0.95),
        throughput_images_per_sec=total / max(total_time / 1000.0, 1e-6),
        peak_ram_mb=round(peak_ram, 1),
    )


def _build_classification_md(payload: dict) -> str:
    lines = [
        "# Classification Benchmark Report",
        f"\n**Dataset:** `{payload['dataset']}`  ",
        f"**Images per model:** {payload['n_images_per_model']}  ",
        f"**Top-k:** {payload['top_k']}",
        "\n## Results\n",
        "| Model | Top-1 | Top-5 | Lat p50 (ms) | Lat p95 (ms) | Failures |",
        "|-------|-------|-------|------------|------------|---------|",
    ]
    for r in payload.get("models", []):
        if r.get("error"):
            lines.append(f"| {r['model_id']} | ERROR | ERROR | — | — | {r.get('error', '')[:60]} |")
        else:
            lines.append(
                f"| {r['model_id']} | {r['top1_accuracy']:.3f} | {r['top5_accuracy']:.3f}"
                f" | {r['latency_p50_ms']:.1f} | {r['latency_p95_ms']:.1f} | {r['n_failures']} |"
            )
    lines.append(f"\n*{payload.get('notes', '')}*")
    return "\n".join(lines) + "\n"


def _build_per_class_csv(results: list[ClassificationBenchmarkResult]) -> str:
    import io

    buf = io.StringIO()
    buf.write("model_id,class,accuracy\n")
    for r in results:
        if r.per_class_accuracy:
            for cls, acc in sorted(r.per_class_accuracy.items()):
                buf.write(f"{r.model_id},{cls},{acc:.4f}\n")
    return buf.getvalue()


@app.callback(invoke_without_command=True)
def benchmark_classification(
    dataset: str = typer.Option(
        ...,
        "--dataset",
        help="Dataset spec: folder:/path or csv:/path.csv",
    ),
    models: str = typer.Option(
        "convnextv2-tiny",
        "--models",
        help="Comma-separated model IDs.",
    ),
    top_k: int = typer.Option(5, "--top-k"),
    max_images: int = typer.Option(100, "--max-images"),
    device: str = typer.Option("auto", "--device"),
    out: Path = typer.Option(None, "--out"),
    report_md: Path = typer.Option(None, "--report-md", help="Write Markdown report."),
    per_class_csv: Path = typer.Option(
        None, "--per-class-csv", help="Write per-class accuracy CSV."
    ),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark classification models: top-k accuracy, per-class, latency."""
    try:
        pairs = _load_dataset(dataset, max_images)
    except (FileNotFoundError, ValueError) as exc:
        payload = {
            "code": "DATASET_SCHEMA_REQUIRED",
            "message": str(exc),
            "expected_format": {
                "folder": "folder:/path — subdir per class",
                "csv": "csv:/path.csv — columns: image_path,label",
            },
        }
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['code']}[/red]: {payload['message']}")
        raise typer.Exit(2)

    if not pairs:
        console.print("[red]Dataset is empty.[/red]")
        raise typer.Exit(2)

    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    results: list[ClassificationBenchmarkResult] = []

    for mid in model_ids:
        if not json_:
            console.print(f"[cyan]Benchmarking {mid}...[/cyan]")
        r = _benchmark_model(mid, pairs, top_k=top_k, device=device, auto_pull=auto_pull)
        r.dataset = dataset
        results.append(r)

    payload = {
        "benchmark": "classification",
        "dataset": dataset,
        "n_images_per_model": len(pairs),
        "top_k": top_k,
        "notes": "Correct metric for classification is top-k accuracy, not mAP.",
        "models": [r.to_dict() for r in results],
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if report_md:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(_build_classification_md(payload))
    if per_class_csv:
        per_class_csv.parent.mkdir(parents=True, exist_ok=True)
        per_class_csv.write_text(_build_per_class_csv(results))
    if json_:
        print(json.dumps(payload, indent=2))
        return

    for r in results:
        if r.error:
            console.print(f"[red]{r.model_id}[/red]: {r.error}")
            continue
        console.rule(f"[bold cyan]{r.model_id}")
        table = Table(show_header=True)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Top-1 accuracy", f"{r.top1_accuracy:.3f}")
        table.add_row("Top-5 accuracy", f"{r.top5_accuracy:.3f}")
        table.add_row("Latency p50", f"{r.latency_p50_ms:.1f} ms")
        table.add_row("Latency p95", f"{r.latency_p95_ms:.1f} ms")
        table.add_row("Throughput", f"{r.throughput_images_per_sec:.1f} img/s")
        table.add_row("Failures", str(r.n_failures))
        console.print(table)


__all__ = ["ClassificationBenchmarkResult", "app"]

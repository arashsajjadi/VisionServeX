# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""benchmark-anomaly — industrial anomaly detection benchmark.

Evaluates PatchCore (or other anomalib algorithms) on an MVTec-like dataset.

Dataset format (MVTec-like):
    category/
        train/good/*.png         ← normal images for training
        test/good/*.png          ← normal test images
        test/<defect_type>/*.png ← anomalous test images
        ground_truth/<defect_type>/*.png (optional)

Dataset format (simple):
    dataset_name/normal/*.png    ← normal images
    dataset_name/test/*.png      ← test images (label from CSV or assumed anomalous)

Metrics (when labels available):
- image_auroc
- anomaly_score_mean_normal
- anomaly_score_mean_anomaly
- score_separation

Honest limitation: pixel_auroc requires pixel-level ground-truth masks.
If only image-level labels are available, pixel_auroc is NOT reported.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="Anomaly detection benchmark (PatchCore + anomalib family).",
)
console = Console()

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class AnomalyBenchmarkResult:
    model: str
    dataset: str
    n_normal_train: int
    n_normal_test: int
    n_anomaly_test: int
    image_auroc: float | None
    anomaly_score_mean_normal: float
    anomaly_score_mean_anomaly: float
    score_separation: float
    latency_p50_ms: float
    latency_p95_ms: float
    heatmaps_saved: int
    notes: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_images(path: Path, exts: set[str] = _IMAGE_EXTS) -> list[Path]:
    return [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[k]


def _compute_auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float | None:
    """Simple approximation of image AUROC via U-statistic."""
    if not normal_scores or not anomaly_scores:
        return None
    n_pos = len(anomaly_scores)
    n_neg = len(normal_scores)
    n_correct = sum(a > n for a in anomaly_scores for n in normal_scores)
    return n_correct / max(n_pos * n_neg, 1)


def _build_anomaly_md(payload: dict, result: AnomalyBenchmarkResult) -> str:
    r = result
    auroc_str = f"{r.image_auroc:.3f}" if r.image_auroc is not None else "n/a (labels required)"
    lines = [
        "# Anomaly Detection Benchmark Report",
        f"\n**Model:** `{r.model}`  ",
        f"**Dataset:** `{r.dataset}`",
        "\n## Metrics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| image_auroc | {auroc_str} |",
        f"| score_mean_normal | {r.anomaly_score_mean_normal:.4f} |",
        f"| score_mean_anomaly | {r.anomaly_score_mean_anomaly:.4f} |",
        f"| score_separation | {r.score_separation:.4f} |",
        f"| latency_p50_ms | {r.latency_p50_ms:.1f} |",
        f"| latency_p95_ms | {r.latency_p95_ms:.1f} |",
        f"| n_normal_train | {r.n_normal_train} |",
        f"| n_anomaly_test | {r.n_anomaly_test} |",
    ]
    if r.image_auroc is None:
        lines.append(
            "\n> **Note:** image_auroc unavailable — no labeled normal/anomaly split found."
        )
    if r.error:
        lines.append(f"\n> **Error:** {r.error[:200]}")
    lines.append(f"\n*{r.notes}*")
    return "\n".join(lines) + "\n"


_MOCK_MODELS = {"mock-anomaly", "mock"}
_ANOMALIB_MODELS = {
    "patchcore",
    "padim",
    "fastflow",
    "efficientad",
    "winclip",
    "draem",
    "reverse_distillation",
}


def _pixel_anomaly_score(img_path: Path) -> float:
    """Cheap proxy anomaly score: mean abs deviation from mid-gray (128/255).

    Normal images from synthetic fixtures cluster around a specific brightness.
    Anomalous images are offset. This gives plausible relative scores for smoke
    testing without requiring any trained model.
    """
    try:
        from PIL import Image

        img = Image.open(img_path).convert("L").resize((64, 64))
        pixels = list(img.getdata())
        mean = sum(pixels) / max(len(pixels), 1)
        mad = sum(abs(p - mean) for p in pixels) / max(len(pixels), 1)
        return round(mad / 128.0, 4)
    except Exception:
        return 0.5


@app.callback(invoke_without_command=True)
def benchmark_anomaly(
    dataset: str = typer.Option(..., "--dataset", help="Dataset spec: mvtec:/path or simple:/path"),
    model: str = typer.Option("patchcore", "--model"),
    max_images: int = typer.Option(50, "--max-images"),
    out: Path = typer.Option(None, "--out"),
    report_md: Path = typer.Option(None, "--report-md", help="Write Markdown report."),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text or json (notebook contract)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark anomaly detection. Use --model mock-anomaly to run without [anomaly] installed."""
    # v2.16.0: --format json behaves like --json (notebook contract).
    if fmt == "json":
        json_ = True
    import importlib
    import time

    is_mock = model in _MOCK_MODELS

    if not is_mock:
        try:
            importlib.import_module("anomalib")
            anomalib_available = True
        except ImportError:
            anomalib_available = False

        if not anomalib_available:
            payload = {
                "code": "ANOMALIB_REQUIRED",
                "model": model,
                "dataset": dataset,
                "fix": "pip install 'visionservex[anomaly]'",
                "alternative": "Use --model mock-anomaly to benchmark without anomalib.",
            }
            if json_:
                print(json.dumps(payload, indent=2))
            else:
                console.print(f"[red]{payload['code']}[/red]: anomalib not installed")
                console.print(f"  fix: {payload['fix']}")
                console.print(f"  tip: {payload['alternative']}")
            raise typer.Exit(3)

    # Parse dataset
    root = Path(dataset.split(":", 1)[1])

    if not root.exists():
        payload = {
            "code": "ANOMALY_DATASET_REQUIRED",
            "message": f"Dataset path not found: {root}",
            "expected_format": "mvtec:<path>  — category/train/good/*.png + category/test/.../*.png",
        }
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['code']}[/red]: {payload['message']}")
        raise typer.Exit(3)

    # Collect images
    train_good: list[Path] = []
    test_normal: list[Path] = []
    test_anomaly: list[Path] = []

    if dataset.startswith("mvtec:"):
        train_dir = root / "train" / "good"
        test_dir = root / "test"
        if train_dir.exists():
            train_good = _iter_images(train_dir)[:max_images]
        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                imgs = _iter_images(sub)[: max_images // 2]
                if sub.name == "good":
                    test_normal.extend(imgs)
                else:
                    test_anomaly.extend(imgs)
    else:
        normal_dir = root / "normal"
        test_dir = root / "test"
        if normal_dir.exists():
            train_good = _iter_images(normal_dir)[:max_images]
        if test_dir.exists():
            test_anomaly = _iter_images(test_dir)[:max_images]

    if not train_good and not json_:
        console.print("[yellow]No training images found. Scores are unsupervised proxies.[/yellow]")

    # --- Mock path (no anomalib required) ------------------------------------
    if is_mock:
        t0_all = time.perf_counter()
        normal_scores: list[float] = [
            _pixel_anomaly_score(p) for p in (train_good + test_normal)[:max_images]
        ]
        anomaly_scores: list[float] = [_pixel_anomaly_score(p) for p in test_anomaly[:max_images]]
        total_ms = (time.perf_counter() - t0_all) * 1000.0
        n_scored = len(normal_scores) + len(anomaly_scores)
        lat = total_ms / max(n_scored, 1)
        auroc = _compute_auroc(normal_scores, anomaly_scores)
        result = AnomalyBenchmarkResult(
            model=model,
            dataset=dataset,
            n_normal_train=len(train_good),
            n_normal_test=len(test_normal),
            n_anomaly_test=len(test_anomaly),
            image_auroc=auroc,
            anomaly_score_mean_normal=sum(normal_scores) / max(len(normal_scores), 1),
            anomaly_score_mean_anomaly=sum(anomaly_scores) / max(len(anomaly_scores), 1),
            score_separation=(
                (sum(anomaly_scores) / max(len(anomaly_scores), 1))
                - (sum(normal_scores) / max(len(normal_scores), 1))
            ),
            latency_p50_ms=lat,
            latency_p95_ms=lat,
            heatmaps_saved=0,
            notes=(
                "mock-anomaly: pixel MAD proxy scores. No trained model. "
                "Use --model patchcore with [anomaly] for real metrics."
            ),
        )
    else:
        # --- Anomalib PatchCore path -----------------------------------------
        import shutil
        import tempfile

        try:
            from anomalib.data import Folder  # type: ignore
            from anomalib.engine import Engine  # type: ignore
            from anomalib.models import Patchcore  # type: ignore

            tmp = tempfile.mkdtemp(prefix="vsx_anomaly_bench_")
            try:
                normal_dest = Path(tmp) / "dataset" / "train" / "good"
                test_normal_dest = Path(tmp) / "dataset" / "test" / "good"
                test_anomaly_dest = Path(tmp) / "dataset" / "test" / "anomaly"
                for d in [normal_dest, test_normal_dest, test_anomaly_dest]:
                    d.mkdir(parents=True)
                for i, p in enumerate(train_good[:20]):
                    shutil.copy(p, normal_dest / f"train_{i:03d}{p.suffix}")
                for i, p in enumerate(test_normal[:10]):
                    shutil.copy(p, test_normal_dest / f"tn_{i:03d}{p.suffix}")
                for i, p in enumerate(test_anomaly[:10]):
                    shutil.copy(p, test_anomaly_dest / f"ta_{i:03d}{p.suffix}")

                pc_model = Patchcore()
                dm = Folder(root=str(Path(tmp) / "dataset"), normal_dir="train/good")
                engine = Engine(max_epochs=1, default_root_dir=tmp)
                engine.fit(model=pc_model, datamodule=dm)

                an_normal: list[float] = []
                an_anomaly: list[float] = []
                latencies: list[float] = []

                for _img_path in list(test_normal_dest.iterdir())[:5]:
                    t0 = time.perf_counter()
                    pred = engine.predict(model=pc_model, dataloaders=[dm])
                    latencies.append((time.perf_counter() - t0) * 1000)
                    if pred and hasattr(pred[0], "pred_score"):
                        an_normal.append(float(pred[0].pred_score))

                for _img_path in list(test_anomaly_dest.iterdir())[:5]:
                    t0 = time.perf_counter()
                    pred = engine.predict(model=pc_model, dataloaders=[dm])
                    latencies.append((time.perf_counter() - t0) * 1000)
                    if pred and hasattr(pred[0], "pred_score"):
                        an_anomaly.append(float(pred[0].pred_score))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

            auroc = _compute_auroc(an_normal, an_anomaly)
            result = AnomalyBenchmarkResult(
                model=model,
                dataset=dataset,
                n_normal_train=len(train_good),
                n_normal_test=len(test_normal),
                n_anomaly_test=len(test_anomaly),
                image_auroc=auroc,
                anomaly_score_mean_normal=sum(an_normal) / max(len(an_normal), 1),
                anomaly_score_mean_anomaly=sum(an_anomaly) / max(len(an_anomaly), 1),
                score_separation=(
                    sum(an_anomaly) / max(len(an_anomaly), 1)
                    - sum(an_normal) / max(len(an_normal), 1)
                ),
                latency_p50_ms=_quantile(latencies, 0.5),
                latency_p95_ms=_quantile(latencies, 0.95),
                heatmaps_saved=0,
                notes="PatchCore via anomalib Engine. max_epochs=1 (smoke validation only).",
            )
        except Exception as exc:
            result = AnomalyBenchmarkResult(
                model=model,
                dataset=dataset,
                n_normal_train=len(train_good),
                n_normal_test=len(test_normal),
                n_anomaly_test=len(test_anomaly),
                image_auroc=None,
                anomaly_score_mean_normal=0.0,
                anomaly_score_mean_anomaly=0.0,
                score_separation=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                heatmaps_saved=0,
                notes="anomalib Engine API invocation failed — see error field.",
                error=str(exc)[:300],
            )

    payload = {
        "benchmark": "anomaly",
        "model": model,
        "dataset": dataset,
        "notes": (
            "image_auroc requires labeled test set (normal vs anomaly directories). "
            "pixel_auroc requires ground-truth masks and is not computed here."
        ),
        "result": result.to_dict(),
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if report_md:
        report_md.parent.mkdir(parents=True, exist_ok=True)
        report_md.write_text(_build_anomaly_md(payload, result))
    if json_:
        print(json.dumps(payload, indent=2))
        return

    r = result
    console.print(f"[bold]Anomaly benchmark — {r.model}[/bold]")
    console.print(f"  n_train_normal: {r.n_normal_train}")
    if r.image_auroc is not None:
        console.print(f"  image_auroc: {r.image_auroc:.3f}")
    else:
        console.print("  image_auroc: n/a — labels required (normal vs anomaly split)")
    console.print(f"  score sep: {r.score_separation:.3f}")
    if r.error:
        console.print(f"  [red]error:[/red] {r.error[:120]}")


__all__ = ["AnomalyBenchmarkResult", "app"]

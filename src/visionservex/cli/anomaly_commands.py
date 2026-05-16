# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Industrial / anomaly detection commands.

Optional extra: ``pip install 'visionservex[anomaly]'`` pulls Anomalib.

This module never fake-trains: if anomalib is missing, every command returns
``ANOMALIB_REQUIRED`` with the exact install command. If a dataset directory
is missing or empty, it returns ``DATASET_REQUIRED``.

Reference:
- https://github.com/open-edge-platform/anomalib
- https://anomalib.readthedocs.io/
- https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Industrial anomaly detection (Anomalib PatchCore family).", no_args_is_help=True
)
console = Console()


SUPPORTED_ALGOS: dict[str, dict] = {
    "patchcore": {
        "anomalib_class": "Patchcore",
        "description": "PatchCore — memory-bank anomaly detection (CVPR 2022).",
        "paper": "https://arxiv.org/abs/2106.08265",
    },
    "padim": {
        "anomalib_class": "Padim",
        "description": "PaDiM — patch distribution modeling.",
        "paper": "https://arxiv.org/abs/2011.08785",
    },
    "fastflow": {
        "anomalib_class": "Fastflow",
        "description": "FastFlow — normalizing flows for anomaly localization.",
        "paper": "https://arxiv.org/abs/2111.07677",
    },
    "efficientad": {
        "anomalib_class": "EfficientAd",
        "description": "EfficientAD — millisecond-latency anomaly detection.",
        "paper": "https://arxiv.org/abs/2303.14535",
    },
    "winclip": {
        "anomalib_class": "WinClip",
        "description": "WinCLIP — zero-/few-shot anomaly classification.",
        "paper": "https://arxiv.org/abs/2303.14814",
    },
    "draem": {
        "anomalib_class": "Draem",
        "description": "DRAEM — discriminatively trained reconstruction.",
        "paper": "https://arxiv.org/abs/2108.07610",
    },
    "reverse_distillation": {
        "anomalib_class": "ReverseDistillation",
        "description": "Reverse distillation for anomaly detection (CVPR 2022).",
        "paper": "https://arxiv.org/abs/2201.10703",
    },
}


@dataclass
class AnomalyError:
    code: str
    message: str
    fix: str

    def to_dict(self) -> dict:
        return asdict(self)


def _anomalib_available() -> tuple[bool, str | None]:
    """Return (installed, version) for the anomalib package."""
    try:
        mod = importlib.import_module("anomalib")
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def _require_anomalib() -> AnomalyError | None:
    """Return an AnomalyError if anomalib is missing, else None."""
    installed, _ = _anomalib_available()
    if installed:
        return None
    return AnomalyError(
        code="ANOMALIB_REQUIRED",
        message="anomalib is not installed",
        fix=(
            "pip install 'visionservex[anomaly]'   "
            "(or: pip install anomalib  — see https://anomalib.readthedocs.io/)"
        ),
    )


def _require_dataset(path: Path) -> AnomalyError | None:
    if not path.exists():
        return AnomalyError(
            code="DATASET_REQUIRED",
            message=f"Dataset path does not exist: {path}",
            fix=f"Create or download data, e.g.: mkdir -p {path} && cp normal_images/*.png {path}/",
        )
    if path.is_dir():
        n_images = sum(
            1
            for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if n_images == 0:
            return AnomalyError(
                code="DATASET_REQUIRED",
                message=f"Dataset directory {path} has no images",
                fix=f"Add training images: cp *.png {path}/  (recommended: at least 50 normal images)",
            )
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("list")
def list_algos(json_: bool = typer.Option(False, "--json")) -> None:
    """List supported anomaly algorithms."""
    available, version = _anomalib_available()
    payload = {
        "anomalib_installed": available,
        "anomalib_version": version,
        "algorithms": SUPPORTED_ALGOS,
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    table = Table(title="Supported anomaly algorithms", show_header=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Anomalib class", style="dim")
    table.add_column("Description")
    for name, info in SUPPORTED_ALGOS.items():
        table.add_row(name, info["anomalib_class"], info["description"])
    console.print(table)
    if available:
        console.print(f"[green]anomalib installed:[/green] version {version}")
    else:
        console.print(
            "[yellow]anomalib not installed.[/yellow] Run: pip install 'visionservex[anomaly]'"
        )


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Check anomalib install and report version + available models."""
    available, version = _anomalib_available()
    payload: dict = {"anomalib_installed": available, "anomalib_version": version}
    if available:
        try:
            anomalib_models = importlib.import_module("anomalib.models")
            payload["module_path"] = getattr(anomalib_models, "__file__", "")
        except Exception as exc:
            payload["module_load_error"] = str(exc)[:200]
    else:
        err = _require_anomalib()
        if err:
            payload["error"] = err.to_dict()
    if json_:
        print(json.dumps(payload, indent=2))
        return
    if available:
        console.print(f"[green]anomalib {version} is installed[/green]")
    else:
        console.print("[yellow]anomalib is NOT installed[/yellow]")
        console.print(f"  fix: {_require_anomalib().fix}")


@app.command("train")
def train(
    algo: str = typer.Argument(..., help=f"Algorithm name. One of: {', '.join(SUPPORTED_ALGOS)}"),
    data: Path = typer.Option(..., "--data", help="Directory of NORMAL training images."),
    out: Path = typer.Option(..., "--out", help="Output directory for the trained model."),
    image_size: int = typer.Option(256, "--image-size"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan but do not start training."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Train an anomaly model on a folder of normal images."""
    if algo not in SUPPORTED_ALGOS:
        console.print(
            f"[red]Unknown algorithm {algo!r}.[/red] Choose from: {list(SUPPORTED_ALGOS)}"
        )
        raise typer.Exit(2)

    err = _require_anomalib()
    if err is None:
        err = _require_dataset(data)
    if err is not None:
        payload = err.to_dict()
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{err.code}[/red]: {err.message}")
            console.print(f"  fix: {err.fix}")
        raise typer.Exit(3)

    plan = {
        "algo": algo,
        "anomalib_class": SUPPORTED_ALGOS[algo]["anomalib_class"],
        "data": str(data),
        "out": str(out),
        "image_size": image_size,
        "dry_run": dry_run,
    }
    if dry_run:
        plan["status"] = "dry_run"
        if json_:
            print(json.dumps(plan, indent=2))
            return
        console.print("[bold]Training plan (dry-run)[/bold]")
        for k, v in plan.items():
            console.print(f"  {k}: {v}")
        return

    # Real training path — guarded by anomalib import.
    out.mkdir(parents=True, exist_ok=True)
    try:
        anomalib_models = importlib.import_module("anomalib.models")
        cls = getattr(anomalib_models, SUPPORTED_ALGOS[algo]["anomalib_class"])
        # We DO NOT auto-train heavyweight models here. We initialize the model
        # and write a manifest so the user can complete training via anomalib
        # CLI / engine. This keeps the command honest: it does not silently
        # spin a multi-hour job.
        model = cls()
        _ = model  # not invoked
        manifest = {
            **plan,
            "anomalib_model": SUPPORTED_ALGOS[algo]["anomalib_class"],
            "status": "scaffold_only",
            "next_step": (
                "Run anomalib's own Engine with the same data path. VisionServeX "
                "intentionally does not start training in a long-running CLI; use:\n"
                f"  anomalib train --model {algo} --data {data}"
            ),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if json_:
            print(json.dumps(manifest, indent=2))
            return
        console.print(f"[green]Scaffold written to[/green] {out}/manifest.json")
        console.print(f"[dim]{manifest['next_step']}[/dim]")
    except Exception as exc:
        console.print(f"[red]anomalib invocation failed:[/red] {exc}")
        raise typer.Exit(4) from exc


@app.command("predict")
def predict_cmd(
    model_dir: Path = typer.Argument(..., help="Trained anomaly model directory."),
    image: Path = typer.Argument(..., help="Image to score."),
    save_heatmap: Path = typer.Option(None, "--save-heatmap"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Score one image against a trained anomaly model."""
    err = _require_anomalib()
    if err is None and not model_dir.exists():
        err = AnomalyError(
            code="MODEL_REQUIRED",
            message=f"Trained model dir not found: {model_dir}",
            fix=f"Run: visionservex anomaly train patchcore --data NORMAL --out {model_dir}",
        )
    if err is None and not image.exists():
        err = AnomalyError(
            code="INPUT_NOT_FOUND",
            message=f"Image not found: {image}",
            fix=f"Check path: {image}",
        )
    if err is not None:
        if json_:
            print(json.dumps(err.to_dict(), indent=2))
        else:
            console.print(f"[red]{err.code}[/red]: {err.message}\n  fix: {err.fix}")
        raise typer.Exit(3)

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        msg = f"{model_dir} has no manifest.json — was the model fully trained via anomalib?"
        if json_:
            print(json.dumps({"code": "MODEL_REQUIRED", "message": msg}, indent=2))
        else:
            console.print(f"[red]MODEL_REQUIRED[/red]: {msg}")
        raise typer.Exit(3)

    # Per design: VisionServeX wrapper does not duplicate anomalib's predict
    # loop. Print the exact anomalib CLI invocation.
    manifest = json.loads(manifest_path.read_text())
    out = {
        "model_dir": str(model_dir),
        "image": str(image),
        "save_heatmap": str(save_heatmap) if save_heatmap else None,
        "status": "delegate_to_anomalib_cli",
        "next_step": (
            f"anomalib predict --model {manifest.get('algo', 'patchcore')} "
            f"--data {image} --ckpt-path {model_dir}/weights.ckpt"
        ),
    }
    if json_:
        print(json.dumps(out, indent=2))
        return
    console.print(out["next_step"])


@app.command("benchmark")
def benchmark_cmd(
    dataset: str = typer.Option(..., "--dataset", help="Dataset spec, e.g. mvtec:/path/to/mvtec"),
    model: str = typer.Option("patchcore", "--model"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run a structured anomaly benchmark.

    For v1.9.0 this always returns BENCHMARK_NOT_IMPLEMENTED with the exact
    expected data format. PatchCore image-level AUROC / pixel-level AUROC
    will land in a later release.
    """
    payload = {
        "code": "BENCHMARK_NOT_IMPLEMENTED",
        "dataset": dataset,
        "model": model,
        "expected_format": {
            "dataset_layout": "mvtec:/path: should contain category subfolders, each with train/good/ and test/good/+test/<defect>/+ground_truth/<defect>/",
            "metrics": ["image_auroc", "pixel_auroc", "image_f1max"],
        },
        "roadmap": (
            "Once anomalib's Engine integration lands in VisionServeX, this command "
            "will run anomalib's evaluator and report image-level + pixel-level AUROC."
        ),
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[yellow]{payload['code']}[/yellow]")
    console.print(f"  Roadmap: {payload['roadmap']}")


__all__ = ["SUPPORTED_ALGOS", "AnomalyError", "app"]

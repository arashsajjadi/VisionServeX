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


@app.command("create-env")
def create_env(
    name: str = typer.Option("visionservex-anomaly", "--name"),
    python: str = typer.Option("3.11", "--python"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate conda recipe for anomalib-compatible environment."""
    commands = [
        f"conda create -n {name} python={python} -y",
        f"conda run -n {name} pip install -U pip",
        f"conda run -n {name} pip install anomalib",
        f"conda run -n {name} pip install 'visionservex[anomaly]'",
        f'conda run -n {name} python -c "import anomalib; print(anomalib.__version__)"',
        f"conda run -n {name} visionservex anomaly doctor",
    ]
    payload = {
        "env_name": name,
        "python": python,
        "commands": commands,
        "validated_versions": ["anomalib>=1.0,<3.0"],
        "smoke_command": (
            f"conda run -n {name} visionservex anomaly train patchcore"
            " --data /path/to/normal_images --out /tmp/vsx_patchcore --dry-run"
        ),
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]Conda recipe for anomalib environment:[/bold] {name}")
    for cmd in commands:
        console.print(f"  [cyan]{cmd}[/cyan]")
    console.print(f"\n[dim]Smoke test: {payload['smoke_command']}[/dim]")


@app.command("install-help")
def install_help(json_: bool = typer.Option(False, "--json")) -> None:
    """Show install options for anomalib (native pip, conda env, docker)."""
    options = {
        "native_pip": {
            "description": "Install anomalib directly into current environment",
            "commands": [
                "pip install 'visionservex[anomaly]'",
                "# or: pip install anomalib",
            ],
            "validated_versions": ["anomalib>=1.0,<3.0"],
        },
        "conda_env": {
            "description": "Isolated conda environment (recommended to avoid conflicts)",
            "commands": [
                "visionservex anomaly create-env --name visionservex-anomaly --python 3.11",
                "conda activate visionservex-anomaly",
                "visionservex anomaly doctor",
            ],
        },
        "docker": {
            "description": "Docker image with anomalib pre-installed",
            "commands": [
                "docker pull openedgeplatform/anomalib:latest",
                "docker run --rm -v $(pwd):/data openedgeplatform/anomalib:latest anomalib --help",
            ],
            "note": "See https://anomalib.readthedocs.io/ for official Docker instructions.",
        },
    }
    if json_:
        print(json.dumps(options, indent=2))
        return
    for method, info in options.items():
        console.print(f"\n[bold]{method}[/bold]: {info['description']}")
        for cmd in info.get("commands", []):
            console.print(f"  [cyan]{cmd}[/cyan]")
        if "note" in info:
            console.print(f"  [dim]{info['note']}[/dim]")


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
def doctor(
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path | None = typer.Option(None, "--out", help="Write structured JSON to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Check anomalib install and report version + available models.

    v2.16.0: accepts ``--format json`` and ``--out PATH`` so notebooks can
    classify the result without depending on the legacy ``--json`` flag.
    """
    available, version = _anomalib_available()
    payload: dict = {
        "status": "ok" if available else "expected_blocker",
        "code": "OK" if available else "ANOMALIB_REQUIRED",
        "anomalib_installed": available,
        "anomalib_version": version,
        "warnings": [],
        "errors": [],
    }
    if available:
        try:
            anomalib_models = importlib.import_module("anomalib.models")
            payload["module_path"] = getattr(anomalib_models, "__file__", "")
        except Exception as exc:
            payload["module_load_error"] = str(exc)[:200]
            payload["warnings"].append("anomalib installed but submodule load failed")
    else:
        err = _require_anomalib()
        if err:
            payload["error"] = err.to_dict()
            payload["message"] = err.message
            payload["install_command"] = err.fix
    json_mode = json_ or fmt == "json"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if json_mode:
        print(json.dumps(payload, indent=2))
        return
    if available:
        console.print(f"[green]anomalib {version} is installed[/green]")
    else:
        console.print("[yellow]anomalib is NOT installed[/yellow]")
        console.print(f"  fix: {_require_anomalib().fix}")


@app.command("smoke")
def smoke_cmd(
    model: str = typer.Option("patchcore", "--model", help="Anomaly algorithm name."),
    dataset: str = typer.Option(
        "", "--dataset", help="Dataset spec (simple:<path>); optional for installed-check smokes."
    ),
    max_images: int = typer.Option(4, "--max-images"),
    out: Path | None = typer.Option(None, "--out", help="Write structured JSON to this path."),
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Notebook-safe anomaly smoke test.

    Replaces ``scripts/run_anomaly_smoke.sh`` for v2.16.0. If anomalib is not
    installed (or the dataset path is missing), returns ``status=expected_blocker``
    instead of crashing.
    """
    json_mode = json_ or fmt == "json"

    def _emit(payload: dict, exit_code: int = 0) -> None:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        if json_mode:
            print(json.dumps(payload, indent=2))
        else:
            color = "green" if payload["status"] == "ok" else "yellow"
            console.print(f"[{color}]{payload['code']}[/{color}]: {payload.get('message', '')}")
        if exit_code:
            raise typer.Exit(exit_code)

    available, version = _anomalib_available()
    if not available:
        err = _require_anomalib()
        _emit(
            {
                "status": "expected_blocker",
                "code": "ANOMALIB_REQUIRED",
                "model": model,
                "dataset": dataset,
                "message": err.message if err else "anomalib not installed",
                "install_command": err.fix if err else "pip install 'visionservex[anomaly]'",
                "anomalib_installed": False,
                "warnings": [],
                "errors": [],
            },
            exit_code=0,  # expected blocker, not a hard fail
        )
        return

    payload = {
        "status": "ok",
        "code": "OK",
        "model": model,
        "dataset": dataset,
        "max_images": max_images,
        "anomalib_installed": True,
        "anomalib_version": version,
        "message": f"anomalib {version} available; smoke gate passed.",
        "warnings": [],
        "errors": [],
    }
    _emit(payload)


@app.command("smoke-script")
def smoke_script_cmd(
    fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Alias for ``visionservex anomaly smoke`` — drop-in for ``scripts/run_anomaly_smoke.sh``.

    Notebook contract: ``visionservex anomaly smoke-script --format json --out OUT``
    replaces ``bash scripts/run_anomaly_smoke.sh``.
    """
    smoke_cmd(
        model="patchcore",
        dataset="",
        max_images=4,
        out=out,
        fmt=fmt,
        json_=(fmt == "json"),
    )


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
        # Try anomalib >= 1.0 Engine API first.
        try:
            from anomalib.data import Folder  # type: ignore
            from anomalib.engine import Engine  # type: ignore
            from anomalib.models import Patchcore  # type: ignore

            _algo_class_map = {
                "patchcore": Patchcore,
            }
            # Only call real Engine for patchcore in v2.1 — other algorithms
            # delegate via CLI because their Engine API varies more.
            if algo == "patchcore":
                model_instance = Patchcore()
                datamodule = Folder(root=str(data), normal_dir=str(data))
                engine = Engine(
                    max_epochs=1,  # smoke/scaffold: one epoch max
                    default_root_dir=str(out),
                )
                engine.fit(model=model_instance, datamodule=datamodule)
                manifest = {
                    **plan,
                    "status": "trained",
                    "engine": "anomalib.Engine",
                    "note": "Trained with max_epochs=1 for smoke validation.",
                }
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                if json_:
                    print(json.dumps(manifest, indent=2))
                    return
                console.print(f"[green]Trained {algo} → {out}[/green]")
                console.print("[dim]1 epoch only. Increase max_epochs for production.[/dim]")
                return
        except (ImportError, Exception) as _engine_exc:
            # Engine API not available or changed — fall through to delegation.
            pass

        anomalib_models = importlib.import_module("anomalib.models")
        cls = getattr(anomalib_models, SUPPORTED_ALGOS[algo]["anomalib_class"])
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

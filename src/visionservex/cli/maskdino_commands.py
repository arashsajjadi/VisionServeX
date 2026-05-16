# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""MaskDINO / Detectron2 expert sidecar commands.

MaskDINO is Apache-2.0 but ships custom CUDA ops via Detectron2, so it
cannot ride VisionServeX's permissive default core. These commands
return structured status, pinned environment recipes, and exact blocker
messages — never silent failure.

References:
- https://github.com/IDEA-Research/MaskDINO
- https://github.com/facebookresearch/detectron2
"""

from __future__ import annotations

import importlib
import json

import typer
from rich.console import Console

app = typer.Typer(
    help="MaskDINO / Detectron2 expert sidecar — create-env, install-help, doctor, validate.",
    no_args_is_help=True,
)
console = Console()


_MASKDINO_MODELS: dict[str, dict] = {
    "maskdino-swinl-coco": {
        "display": "MaskDINO (Swin-L, COCO instance)",
        "source_repo": "https://github.com/IDEA-Research/MaskDINO",
        "license": "Apache-2.0",
        "task": "segment",
        "config_path": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",
        "checkpoint_url": None,
        "checkpoint_note": (
            "MaskDINO checkpoint URL was NOT FOUND in retrieved source set. "
            "Obtain from the official model zoo at "
            "https://github.com/IDEA-Research/MaskDINO/blob/main/README.md "
            "and pass the local path."
        ),
        "install": ("Detectron2 sidecar — see `visionservex maskdino create-env`."),
    },
    "maskdino-r50-coco": {
        "display": "MaskDINO (R50, COCO instance)",
        "source_repo": "https://github.com/IDEA-Research/MaskDINO",
        "license": "Apache-2.0",
        "task": "segment",
        "config_path": "configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_4s.yaml",
        "checkpoint_url": None,
        "checkpoint_note": (
            "MaskDINO checkpoint URL was NOT FOUND in retrieved source set. "
            "Obtain from the official README and pass the local path."
        ),
        "install": ("Detectron2 sidecar — see `visionservex maskdino create-env`."),
    },
}


def _probe(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


@app.command("create-env")
def create_env(
    name: str = typer.Option("visionservex-maskdino", "--name"),
    python: str = typer.Option("3.10", "--python"),
    cuda: str = typer.Option("11.8", "--cuda", help="CUDA toolkit version pin."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print a pinned conda recipe for a MaskDINO/Detectron2 sidecar env."""
    commands = [
        f"conda create -n {name} python={python} -y",
        f"conda activate {name}",
        "pip install -U pip",
        # Detectron2 needs a matching torch build with custom CUDA ops.
        f"pip install torch torchvision --index-url https://download.pytorch.org/whl/cu{cuda.replace('.', '')}",
        ("python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"),
        "git clone https://github.com/IDEA-Research/MaskDINO",
        "cd MaskDINO && pip install -r requirements.txt && cd ..",
        "pip install 'visionservex'",
        "visionservex maskdino doctor",
    ]
    payload = {
        "env_name": name,
        "python": python,
        "cuda": cuda,
        "commands": commands,
        "validated_versions": ["detectron2>=0.6", "torch>=2.0"],
        "license": "Apache-2.0",
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]Conda recipe for MaskDINO sidecar:[/bold] {name}")
    for cmd in commands:
        console.print(f"  [cyan]{cmd}[/cyan]")


@app.command("install-help")
def install_help(json_: bool = typer.Option(False, "--json")) -> None:
    options = {
        "isolated_env": {
            "description": "Recommended — isolated conda env with pinned CUDA.",
            "commands": [
                "visionservex maskdino create-env --name visionservex-maskdino --python 3.10",
                "conda activate visionservex-maskdino",
                "visionservex maskdino doctor",
            ],
        },
        "manual": {
            "description": "Manual install (advanced users).",
            "commands": [
                "pip install 'git+https://github.com/facebookresearch/detectron2.git'",
                "git clone https://github.com/IDEA-Research/MaskDINO",
                "cd MaskDINO && pip install -r requirements.txt",
            ],
        },
    }
    if json_:
        typer.echo(json.dumps(options, indent=2))
        return
    for method, info in options.items():
        console.print(f"\n[bold]{method}[/bold]: {info['description']}")
        for cmd in info.get("commands", []):
            console.print(f"  [cyan]{cmd}[/cyan]")


@app.command("doctor")
def doctor_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """Probe detectron2 and report install status."""
    payload = {
        "detectron2_installed": _probe("detectron2"),
        "torch_installed": _probe("torch"),
        "maskdino_repo_importable": _probe("maskdino"),
    }
    if not payload["detectron2_installed"]:
        payload.update(
            {
                "code": "DETECTRON2_REQUIRED",
                "fix": (
                    "pip install 'git+https://github.com/facebookresearch/detectron2.git' "
                    "(see `visionservex maskdino create-env` for a pinned recipe)."
                ),
            }
        )
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    if payload["detectron2_installed"]:
        console.print("[green]detectron2 installed[/green]")
    else:
        console.print("[yellow]detectron2 NOT installed[/yellow]")
        console.print(f"  fix: {payload['fix']}")


@app.command("validate")
def validate_cmd(
    model_id: str = typer.Argument(...),
    checkpoint: str = typer.Option("", "--checkpoint", help="Local path to MaskDINO checkpoint."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    meta = _MASKDINO_MODELS.get(model_id)
    if meta is None:
        payload = {
            "code": "CONFIG_REQUIRED",
            "message": f"Unknown model {model_id!r}.",
            "available": sorted(_MASKDINO_MODELS),
        }
        _emit(payload, json_)
        raise typer.Exit(2)

    checkpoint_present = bool(checkpoint) and __import__("os").path.exists(checkpoint)
    detectron2_ok = _probe("detectron2")

    payload: dict = {
        "model_id": model_id,
        **meta,
        "detectron2_installed": detectron2_ok,
        "checkpoint": checkpoint or None,
        "checkpoint_present": checkpoint_present,
    }
    if not detectron2_ok:
        payload.update(
            {
                "status": "error",
                "structured_error_code": "DETECTRON2_REQUIRED",
                "message": (
                    "Detectron2 is not installed. MaskDINO requires Detectron2's custom CUDA ops."
                ),
                "fix": (
                    "pip install 'git+https://github.com/facebookresearch/detectron2.git' "
                    "(or visionservex maskdino create-env)"
                ),
            }
        )
    elif not checkpoint_present:
        payload.update(
            {
                "status": "error",
                "structured_error_code": "CHECKPOINT_REQUIRED",
                "message": meta.get("checkpoint_note", "MaskDINO checkpoint is required."),
                "fix": (
                    "Download from the official MaskDINO README and pass "
                    "`--checkpoint /path/to/maskdino_*.pth`."
                ),
            }
        )
    else:
        payload["status"] = "ok"
        payload["message"] = "Detectron2 + MaskDINO checkpoint present."

    _emit(payload, json_)


@app.command("smoke-test")
def smoke_test(
    model_id: str = typer.Argument(...),
    image: str = typer.Argument(..., help="Image path."),
    checkpoint: str = typer.Option("", "--checkpoint"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run a MaskDINO sidecar smoke-test. Returns structured blockers when deps missing."""
    info = _MASKDINO_MODELS.get(model_id)
    if info is None:
        payload = {
            "code": "CONFIG_REQUIRED",
            "message": f"Unknown model {model_id!r}.",
            "available": sorted(_MASKDINO_MODELS),
        }
        _emit(payload, json_)
        raise typer.Exit(2)

    if not _probe("detectron2"):
        payload = {
            "model_id": model_id,
            "status": "error",
            "structured_error_code": "DETECTRON2_REQUIRED",
            "message": "MaskDINO requires Detectron2.",
            "fix": (
                "pip install 'git+https://github.com/facebookresearch/detectron2.git' "
                "(or visionservex maskdino create-env)"
            ),
        }
        _emit(payload, json_)
        raise typer.Exit(3)

    if not checkpoint:
        payload = {
            "model_id": model_id,
            "status": "error",
            "structured_error_code": "CHECKPOINT_REQUIRED",
            "message": info.get("checkpoint_note", "MaskDINO checkpoint required."),
            "fix": "Pass --checkpoint /path/to/maskdino_*.pth",
        }
        _emit(payload, json_)
        raise typer.Exit(3)

    # Real inference happens via the official MaskDINO repo; do not silently
    # invoke its CLI here — return a structured next-step.
    payload = {
        "model_id": model_id,
        "status": "sidecar_run_required",
        "message": (
            "Run inference inside the MaskDINO repo. VisionServeX wraps "
            "validation only because Detectron2's custom ops require its env."
        ),
        "next_step": (
            f"cd MaskDINO && python demo/demo.py --config-file {info['config_path']} "
            f"--input {image} --opts MODEL.WEIGHTS {checkpoint}"
        ),
    }
    _emit(payload, json_)


@app.command("list")
def list_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    payload = [{"id": k, **v} for k, v in _MASKDINO_MODELS.items()]
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    for entry in payload:
        console.print(f"[bold]{entry['id']}[/bold]  —  {entry['display']}")
        console.print(f"  license: {entry['license']}  task: {entry['task']}")
        console.print(f"  source: {entry['source_repo']}")


def _emit(payload: dict, json_: bool) -> None:
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    code = payload.get("structured_error_code") or payload.get("code") or payload.get("status")
    if code:
        console.print(f"[yellow]{code}[/yellow] — {payload.get('message', '')}")
    if payload.get("fix"):
        console.print(f"  fix: {payload['fix']}")
    if payload.get("next_step"):
        console.print(f"  next: {payload['next_step']}")


__all__ = ["app"]

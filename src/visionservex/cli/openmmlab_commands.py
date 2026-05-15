# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""OpenMMLab expert model management commands."""

from __future__ import annotations

import json
import shutil
import subprocess

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="OpenMMLab expert model management (RTMPose, RTMDet-R, Co-DINO, etc.).")
console = Console()

_OPENMMLAB_MODELS = [
    "rtmpose-t",
    "rtmpose-s",
    "rtmpose-m",
    "rtmpose-l",
    "rtmdet-r-t",
    "rtmdet-r-s",
    "rtmdet-r-m",
    "rtmdet-r-l",
    "rtmdet-r2-t",
    "rtmdet-r2-s",
    "rtmdet-r2-m",
    "rtmdet-r2-l",
    "co-dino-inst-vit-l-coco",
    "co-dino-inst-vit-l-lvis",
    "internimage-t",
    "internimage-s",
    "internimage-b",
    "internimage-l",
    "internimage-h",
]

_DOCKER_AVAILABLE_NOTICE = """
OpenMMLab models require the OpenMMLab toolchain (mmengine, mmcv, mmpose,
mmdet, mmrotate, mmpretrain). These can be installed natively or run via
the VisionServeX OpenMMLab Docker image.

  Native:   pip install openmim && mim install mmengine mmcv mmpose mmdet mmrotate
  Docker:   visionservex openmmlab docker-build && visionservex openmmlab docker-run
  Docs:     docs/openmmlab_expert_models.md
"""


@app.command("doctor", help="Check OpenMMLab dependency availability.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Probe which OpenMMLab packages are installed."""
    import importlib

    packages = {
        "mmengine": "pip install openmim && mim install mmengine",
        "mmcv": "pip install openmim && mim install mmcv",
        "mmpose": "mim install mmpose",
        "mmdet": "mim install mmdet",
        "mmrotate": "mim install mmrotate",
        "mmpretrain": "mim install mmpretrain",
    }

    results = {}
    for pkg, hint in packages.items():
        try:
            mod = importlib.import_module(pkg)
            results[pkg] = {
                "installed": True,
                "version": getattr(mod, "__version__", "unknown"),
                "hint": "",
            }
        except Exception:
            results[pkg] = {"installed": False, "version": None, "hint": hint}

    docker_available = shutil.which("docker") is not None
    payload = {
        "packages": results,
        "docker_available": docker_available,
        "all_installed": all(r["installed"] for r in results.values()),
        "notice": _DOCKER_AVAILABLE_NOTICE.strip(),
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print("[bold]OpenMMLab Doctor[/bold]")
    table = Table(show_header=True)
    table.add_column("Package")
    table.add_column("Installed")
    table.add_column("Version")
    table.add_column("Install hint")
    for pkg, info in results.items():
        installed = "[green]yes[/green]" if info["installed"] else "[grey50]no[/grey50]"
        table.add_row(pkg, installed, str(info.get("version") or "-"), info.get("hint") or "")
    console.print(table)
    console.print(
        f"\nDocker available: {'[green]yes[/green]' if docker_available else '[grey50]no[/grey50]'}"
    )

    if not payload["all_installed"]:
        console.print("\n[yellow]Not all packages installed.[/yellow]")
        console.print("Quickest path: [cyan]visionservex openmmlab docker-build[/cyan]")
        console.print("See: [dim]docs/openmmlab_expert_models.md[/dim]")


@app.command("docker-build", help="Build the VisionServeX OpenMMLab Docker image.")
def docker_build(
    tag: str = typer.Option("visionservex-openmmlab:latest", "--tag"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    """Build the OpenMMLab Docker image from docker/openmmlab/Dockerfile."""
    if shutil.which("docker") is None:
        console.print("[red]Docker is not installed or not on PATH.[/red]")
        raise typer.Exit(1)

    cmd = ["docker", "build", "-t", tag, "-f", "docker/openmmlab/Dockerfile", "."]
    if no_cache:
        cmd.insert(2, "--no-cache")

    console.print(f"Building {tag} ...")
    console.print(f"  [dim]{' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print("[red]Docker build failed.[/red]")
        raise typer.Exit(result.returncode)
    console.print(f"[green]Built {tag}[/green]")


@app.command("docker-run", help="Start the VisionServeX OpenMMLab Docker container.")
def docker_run(
    tag: str = typer.Option("visionservex-openmmlab:latest", "--tag"),
    port: int = typer.Option(8081, "--port"),
    gpu: bool = typer.Option(True, "--gpu/--no-gpu"),
) -> None:
    """Run the OpenMMLab container. Listens on localhost:port."""
    if shutil.which("docker") is None:
        console.print("[red]Docker is not installed or not on PATH.[/red]")
        raise typer.Exit(1)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-p",
        f"127.0.0.1:{port}:8080",
        "-v",
        "~/.cache/visionservex:/cache/visionservex",
    ]
    if gpu:
        cmd += ["--gpus", "all"]
    cmd.append(tag)

    console.print(f"Starting {tag} on http://127.0.0.1:{port} ...")
    console.print(f"  [dim]{' '.join(cmd)}[/dim]")
    subprocess.run(cmd, check=False)


@app.command("status", help="Show status of all OpenMMLab models in the registry.")
def status(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    models = [
        entry.model_dump()
        for entry in reg.list()
        if entry.family in {"rtmpose", "rtmdet", "co-dino", "internimage"}
    ]

    if json_:
        typer.echo(json.dumps(models, indent=2, default=str))
        return

    table = Table(title="OpenMMLab Models")
    for col in ("ID", "Task", "Status", "Impl", "Install"):
        table.add_column(col)
    for m in models:
        table.add_row(
            m["id"],
            m["task"],
            m["status"],
            m["implementation_status"],
            m.get("install_extra") or "openmmlab",
        )
    console.print(table)
    console.print("\n[dim]Use `visionservex openmmlab doctor` to check dependencies.[/dim]")
    console.print("[dim]Use `visionservex openmmlab docker-build` for Docker path.[/dim]")


@app.command(
    "smoke-test", help="Run a quick smoke test on an OpenMMLab model (requires dependencies)."
)
def smoke_test(
    model_id: str = typer.Argument("rtmpose-s"),
    json_: bool = typer.Option(False, "--json"),
) -> None:

    try:
        import mmpose  # noqa: F401

        has_mm = True
    except ImportError:
        has_mm = False

    if not has_mm:
        console.print("[yellow]OpenMMLab not installed.[/yellow]")
        console.print("Run: [cyan]pip install openmim && mim install mmengine mmcv mmpose[/cyan]")
        console.print(
            "Or:  [cyan]visionservex openmmlab docker-build && visionservex openmmlab docker-run[/cyan]"
        )
        if json_:
            typer.echo(
                json.dumps(
                    {"model_id": model_id, "status": "skip", "reason": "mmpose not installed"}
                )
            )
        raise typer.Exit(0)

    console.print(f"Testing {model_id} (OpenMMLab native path)...")
    console.print("[yellow]OpenMMLab native integration is currently manual/expert.[/yellow]")
    console.print(
        "Download and configure the model config/checkpoint manually, then use MMPose API directly."
    )
    console.print("See: docs/openmmlab_expert_models.md")


@app.command("list", help="List all available OpenMMLab models.")
def list_models(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.registry import default_registry

    reg = default_registry()
    models = [
        {
            "id": e.id,
            "task": e.task,
            "family": e.family,
            "status": e.status,
            "implementation_status": e.implementation_status,
            "difficulty": e.difficulty,
            "auto_download": e.auto_download,
        }
        for e in reg.list()
        if e.family in {"rtmpose", "rtmdet", "co-dino", "internimage"}
    ]

    if json_:
        typer.echo(json.dumps(models, indent=2, default=str))
        return

    table = Table(title=f"OpenMMLab Models ({len(models)})")
    for col in ("ID", "Task", "Status", "Difficulty"):
        table.add_column(col)
    for m in models:
        table.add_row(m["id"], m["task"], m["status"], m["difficulty"])
    console.print(table)


__all__ = ["app"]

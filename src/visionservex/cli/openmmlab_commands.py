# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""OpenMMLab expert model management commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

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
    "rtmdet-l-coco",
    "rtmdet-r-t",
    "rtmdet-r-s",
    "rtmdet-r-m",
    "rtmdet-r-l",
    "rtmdet-r2-t",
    "rtmdet-r2-s",
    "rtmdet-r2-m",
    "rtmdet-r2-l",
    "oriented-rcnn",
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


@app.command("create-env", help="Generate conda environment recipe for OpenMMLab models.")
def create_env(
    name: str = typer.Option("visionservex-openmmlab", "--name"),
    python: str = typer.Option("3.10", "--python"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate (or print) the conda/pip recipe for OpenMMLab environment."""
    commands = [
        f"conda create -n {name} python={python} -y",
        f"conda run -n {name} pip install -U pip",
        f"conda run -n {name} pip install openmim",
        f"conda run -n {name} mim install mmengine",
        f'conda run -n {name} mim install "mmcv>=2.0.0"',
        f"conda run -n {name} mim install mmpose",
        f"conda run -n {name} mim install mmdet",
        f"conda run -n {name} mim install mmrotate",
    ]
    if json_:
        typer.echo(json.dumps({"env_name": name, "python": python, "commands": commands}, indent=2))
        return
    console.print(f"[bold]OpenMMLab conda env recipe (name={name}, python={python})[/bold]")
    for cmd in commands:
        console.print(f"  [cyan]{cmd}[/cyan]")
    console.print("\n[dim]Copy and run these commands in your terminal.[/dim]")


@app.command("install-help", help="Print OpenMMLab install instructions.")
def install_help(json_: bool = typer.Option(False, "--json")) -> None:
    """Print exact commands to install OpenMMLab in current environment."""
    commands = {
        "native": [
            "pip install openmim",
            "mim install mmengine",
            'mim install "mmcv>=2.0.0"',
            "mim install mmpose",
            "mim install mmdet",
            "mim install mmrotate",
        ],
        "conda_env": "visionservex openmmlab create-env --name visionservex-openmmlab",
        "docker": "visionservex openmmlab docker-build && visionservex openmmlab docker-run",
    }
    if json_:
        print(json.dumps(commands, indent=2))
    else:
        console.print("[bold]OpenMMLab install options:[/bold]")
        console.print("\n[cyan]Native:[/cyan]")
        for cmd in commands["native"]:
            console.print(f"  {cmd}")
        console.print(f"\n[cyan]Conda env:[/cyan] {commands['conda_env']}")
        console.print(f"[cyan]Docker:[/cyan] {commands['docker']}")


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
    image: str = typer.Option("", "--image", help="Optional input image path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate dependencies and (when present) run a real Inferencer call."""
    info = _validate_openmmlab_model(model_id)
    if info.get("status") != "ok":
        if json_:
            typer.echo(json.dumps(info, indent=2))
            return
        code = info.get("structured_error_code", "ERROR")
        console.print(f"[yellow]{code}[/yellow] — {info.get('message', '')}")
        if info.get("fix"):
            console.print(f"  fix: {info['fix']}")
        return

    meta = _PULL_METADATA[model_id]
    task = meta.get("task", "")
    inferencer = None
    try:
        if task == "pose":
            from mmpose.apis import MMPoseInferencer  # type: ignore

            inferencer = MMPoseInferencer(meta.get("config_name", "rtmpose-m"))
        elif task == "detect":
            from mmdet.apis import DetInferencer  # type: ignore

            inferencer = DetInferencer(meta.get("config_name", "rtmdet_l_8xb32-300e_coco"))
        elif task == "obb":
            # mmrotate does not (yet) ship a Inferencer; report sidecar.
            payload = {
                "model_id": model_id,
                "status": "sidecar",
                "structured_error_code": "OBB_INFERENCER_UNAVAILABLE",
                "message": (
                    "mmrotate does not expose a high-level Inferencer. "
                    "Load config + checkpoint manually via mmrotate.apis."
                ),
                "checkpoint": info["checkpoint_path"],
            }
            if json_:
                typer.echo(json.dumps(payload, indent=2))
            else:
                console.print(f"[yellow]{payload['structured_error_code']}[/yellow]")
                console.print(f"  {payload['message']}")
            return
    except Exception as exc:  # pragma: no cover - real env-specific
        payload = {
            "model_id": model_id,
            "status": "error",
            "structured_error_code": "OPENMMLAB_API_UNSUPPORTED",
            "message": str(exc)[:300],
            "fix": meta.get("install", ""),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['structured_error_code']}[/red]: {payload['message']}")
        raise typer.Exit(4)

    if inferencer is None:
        payload = {"model_id": model_id, "status": "skipped", "reason": "unknown task"}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        return
    if not image:
        payload = {
            "model_id": model_id,
            "status": "dry_run",
            "message": "Inferencer constructed. Pass --image PATH to run inference.",
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]{payload['status']}[/green]: {payload['message']}")
        return

    try:
        result = inferencer(image)
    except Exception as exc:  # pragma: no cover - real env-specific
        payload = {
            "model_id": model_id,
            "status": "error",
            "structured_error_code": "OPENMMLAB_INFERENCE_FAILED",
            "message": str(exc)[:300],
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]{payload['structured_error_code']}[/red]: {payload['message']}")
        raise typer.Exit(5) from exc

    payload = {
        "model_id": model_id,
        "status": "ok",
        "image": image,
        "result_type": type(result).__name__,
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[green]OK[/green] {model_id} on {image}: {payload['result_type']}")


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


# Official OpenMMLab checkpoint pull metadata.
# download_url may be None when the exact file path is model-dependent or version-locked.
# In that case the pull command prints exact download instructions.
_PULL_METADATA: dict[str, dict] = {
    "rtmpose-s": {
        "display": "RTMPose (Small)",
        "config_repo": "https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose/coco",
        "model_zoo": "https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#rtmpose",
        "checkpoint_page": "https://github.com/open-mmlab/mmpose/blob/main/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py",
        "install": "pip install openmim && mim install mmengine mmcv mmpose",
        "cache_subdir": "rtmpose-s",
        "checkpoint_filename": "rtmpose-s_simcc-body7_pt-body7_420e-256x192.pth",
        "download_url": None,  # User must obtain from official model zoo
    },
    "rtmpose-t": {
        "display": "RTMPose (Tiny)",
        "config_repo": "https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose/coco",
        "model_zoo": "https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#rtmpose",
        "install": "pip install openmim && mim install mmengine mmcv mmpose",
        "cache_subdir": "rtmpose-t",
        "checkpoint_filename": "rtmpose-t_simcc-body7_pt-body7_420e-256x192.pth",
        "download_url": None,
    },
    "rtmdet-r2-s": {
        "display": "RTMDet-R2 (Small)",
        "config_repo": "https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_rtmdet",
        "model_zoo": "https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_rtmdet/README.md",
        "checkpoint_page": "https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_rtmdet",
        "install": "pip install openmim && mim install mmengine mmcv mmdet mmrotate",
        "cache_subdir": "rtmdet-r2-s",
        "checkpoint_filename": "rtmdet-r2-s.pth",
        "download_url": None,
    },
    "rtmdet-r2-t": {
        "display": "RTMDet-R2 (Tiny)",
        "config_repo": "https://github.com/open-mmlab/mmrotate/tree/main/configs/rotated_rtmdet",
        "model_zoo": "https://github.com/open-mmlab/mmrotate/blob/main/configs/rotated_rtmdet/README.md",
        "install": "pip install openmim && mim install mmengine mmcv mmdet mmrotate",
        "cache_subdir": "rtmdet-r2-t",
        "checkpoint_filename": "rtmdet-r2-t.pth",
        "download_url": None,
    },
    "rtmdet-l-coco": {
        "display": "RTMDet (Large, COCO)",
        "config_repo": "https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet",
        "model_zoo": "https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md",
        "checkpoint_page": "https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet",
        "install": "pip install openmim && mim install mmengine mmcv mmdet",
        "cache_subdir": "rtmdet-l-coco",
        "checkpoint_filename": "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
        "download_url": (
            "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
            "rtmdet_l_8xb32-300e_coco/"
            "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
        ),
        "license": "Apache-2.0",
        "task": "detect",
        "inferencer": "mmdet.apis.DetInferencer",
        "config_name": "rtmdet_l_8xb32-300e_coco",
    },
    "rtmpose-m": {
        "display": "RTMPose (Medium, Halpe26)",
        "config_repo": "https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose",
        "model_zoo": (
            "https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#rtmpose"
        ),
        "checkpoint_page": (
            "https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmpose"
        ),
        "install": "pip install openmim && mim install mmengine mmcv mmpose",
        "cache_subdir": "rtmpose-m",
        "checkpoint_filename": (
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
        ),
        "download_url": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"
        ),
        "license": "Apache-2.0",
        "task": "pose",
        "inferencer": "mmpose.apis.MMPoseInferencer",
        "config_name": "rtmpose-m",
    },
    "oriented-rcnn": {
        "display": "Oriented R-CNN (OBB)",
        "config_repo": ("https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_rcnn"),
        "model_zoo": (
            "https://github.com/open-mmlab/mmrotate/blob/main/configs/oriented_rcnn/README.md"
        ),
        "checkpoint_page": (
            "https://github.com/open-mmlab/mmrotate/tree/main/configs/oriented_rcnn"
        ),
        "install": "pip install openmim && mim install mmengine mmcv mmdet mmrotate",
        "cache_subdir": "oriented-rcnn",
        "checkpoint_filename": "oriented_rcnn_r50_fpn_1x_dota_le90.pth",
        "download_url": None,  # mmrotate hosts model zoo; URL varies per version
        "license": "Apache-2.0",
        "task": "obb",
        "inferencer": "mmrotate (config + checkpoint loader)",
        "config_name": "oriented_rcnn_r50_fpn_1x_dota_le90",
        "note": (
            "OBB output is [x_center, y_center, width, height, theta]. Do NOT flatten to xyxy."
        ),
    },
}


@app.command("pull", help="Download an OpenMMLab checkpoint to the VisionServeX cache.")
def pull(
    model_id: str = typer.Argument(..., help="Model ID to pull (e.g. rtmpose-s, rtmdet-r2-s)."),
    from_url: str | None = typer.Option(
        None,
        "--from-url",
        help="Direct download URL for the checkpoint (from official model zoo).",
    ),
    cache_dir: str | None = typer.Option(
        None,
        "--cache-dir",
        help="Override cache directory (default: ~/.cache/visionservex/openmmlab/).",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Download an OpenMMLab checkpoint to the local VisionServeX cache.

    VisionServeX cannot auto-download OpenMMLab checkpoints because they are
    hosted on OpenMMLab's CDN and require the user to accept the upstream
    terms. This command prepares the cache directory and, when --from-url is
    provided, downloads the checkpoint.

    Usage:
      # Show download instructions
      visionservex openmmlab pull rtmpose-s

      # Download once you have the URL from the official model zoo
      visionservex openmmlab pull rtmpose-s --from-url https://download.openmmlab.com/...
    """
    import pathlib
    import urllib.request

    meta = _PULL_METADATA.get(model_id)
    base_cache = pathlib.Path(
        cache_dir or pathlib.Path.home() / ".cache" / "visionservex" / "openmmlab"
    )

    model_cache = base_cache / meta["cache_subdir"] if meta else base_cache / model_id

    model_cache.mkdir(parents=True, exist_ok=True)

    if not meta:
        msg = (
            f"No pull metadata for model '{model_id}'. "
            f"Supported: {', '.join(_PULL_METADATA)}. "
            "Use --from-url to download a checkpoint directly."
        )
        if json_:
            typer.echo(
                json.dumps(
                    {
                        "model_id": model_id,
                        "status": "no_metadata",
                        "message": msg,
                        "cache_dir": str(model_cache),
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[yellow]{msg}[/yellow]")
        return

    checkpoint_path = model_cache / meta["checkpoint_filename"]

    # Check if already cached
    if checkpoint_path.exists():
        if json_:
            typer.echo(
                json.dumps(
                    {
                        "model_id": model_id,
                        "status": "cached",
                        "path": str(checkpoint_path),
                        "size_bytes": checkpoint_path.stat().st_size,
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[green]Checkpoint already cached:[/green] {checkpoint_path}")
        return

    # Determine download URL
    url = from_url or meta.get("download_url")

    if not url:
        # No auto-download URL — print instructions
        instructions = f"""
[bold]Manual checkpoint download required for {meta["display"]}[/bold]

1. Install the OpenMMLab toolchain:
   [cyan]{meta["install"]}[/cyan]

2. Browse the official model zoo:
   [cyan]{meta.get("model_zoo") or meta["config_repo"]}[/cyan]

3. Download the checkpoint and place it at:
   [cyan]{checkpoint_path}[/cyan]

4. Or use mim to download directly (requires mmpose/mmrotate installed):
   [cyan]mim download mmpose --config <config_name>[/cyan]

5. Then run this command again or provide the URL:
   [cyan]visionservex openmmlab pull {model_id} --from-url <url>[/cyan]
"""
        if json_:
            typer.echo(
                json.dumps(
                    {
                        "model_id": model_id,
                        "status": "CHECKPOINT_REQUIRED",
                        "cache_dir": str(model_cache),
                        "expected_path": str(checkpoint_path),
                        "model_zoo": meta.get("model_zoo"),
                        "config_repo": meta["config_repo"],
                        "install_cmd": meta["install"],
                        "hint": f"Place checkpoint at {checkpoint_path} or use --from-url <url>",
                    },
                    indent=2,
                )
            )
        else:
            console.print(instructions)
        return

    # Attempt download from provided URL
    if not json_:
        console.print(f"Downloading [bold]{meta['display']}[/bold] checkpoint...")
        console.print(f"  URL:  {url}")
        console.print(f"  Dest: {checkpoint_path}")

    try:
        tmp_path = checkpoint_path.with_suffix(".tmp")

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            if not json_ and total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                if block_num % 100 == 0:
                    console.print(f"  {pct}%", end="\r")

        urllib.request.urlretrieve(url, tmp_path, reporthook=_report)
        tmp_path.rename(checkpoint_path)

        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        if json_:
            typer.echo(
                json.dumps(
                    {
                        "model_id": model_id,
                        "status": "ok",
                        "path": str(checkpoint_path),
                        "size_mb": round(size_mb, 1),
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"\n[green]Downloaded {size_mb:.1f} MB → {checkpoint_path}[/green]")
            console.print(
                "\nNext step: Start the sidecar and set VISIONSERVEX_OPENMMLAB_SIDECAR_URL."
            )
            console.print("  See: [cyan]docs/openmmlab_expert_models.md[/cyan]")

    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if json_:
            typer.echo(
                json.dumps(
                    {
                        "model_id": model_id,
                        "status": "error",
                        "error": str(exc),
                    },
                    indent=2,
                )
            )
        else:
            console.print(f"[red]Download failed:[/red] {exc}")
            console.print(f"  Place the checkpoint manually at: [cyan]{checkpoint_path}[/cyan]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# validate — config + checkpoint dependency check (v1.9.0)
# ---------------------------------------------------------------------------


def _validate_openmmlab_model(model_id: str) -> dict:
    """Inspect dependencies, config, and checkpoint for an OpenMMLab model."""
    meta = _PULL_METADATA.get(model_id)
    if meta is None:
        return {
            "model_id": model_id,
            "status": "error",
            "structured_error_code": "CONFIG_REQUIRED",
            "message": f"No pull metadata for {model_id!r}",
            "fix": "Add a _PULL_METADATA entry in cli/openmmlab_commands.py",
        }

    import importlib

    modules_needed = ["mmcv", "mmengine"]
    if model_id.startswith("rtmpose"):
        modules_needed.append("mmpose")
    elif model_id == "rtmdet-l-coco":
        modules_needed.append("mmdet")
    elif model_id.startswith("rtmdet") or model_id == "oriented-rcnn":
        modules_needed.extend(["mmdet", "mmrotate"])
    elif model_id.startswith("co-dino") or model_id.startswith("co-detr"):
        modules_needed.append("mmdet")

    missing = []
    for mod in modules_needed:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)

    import os

    cache_dir = os.environ.get("VISIONSERVEX_CACHE_DIR") or os.path.expanduser(
        "~/.cache/visionservex/openmmlab"
    )
    cache_path = Path(cache_dir) / meta["cache_subdir"]
    ckpt_path = cache_path / meta["checkpoint_filename"]
    has_ckpt = ckpt_path.exists()

    payload: dict = {
        "model_id": model_id,
        "required_modules": modules_needed,
        "missing_modules": missing,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_present": has_ckpt,
        "config_repo": meta.get("config_repo"),
        "model_zoo": meta.get("model_zoo"),
    }
    if missing:
        payload["status"] = "error"
        payload["structured_error_code"] = "OPENMMLAB_REQUIRED"
        payload["message"] = f"Required modules missing: {missing}"
        payload["fix"] = meta["install"]
    elif not has_ckpt:
        payload["status"] = "error"
        payload["structured_error_code"] = "CHECKPOINT_REQUIRED"
        payload["message"] = f"Checkpoint not in cache: {ckpt_path}"
        payload["fix"] = f"visionservex openmmlab pull {model_id} --from-url <URL>"
    else:
        payload["status"] = "ok"
        payload["message"] = "Dependencies and checkpoint are present."
    return payload


@app.command(
    "validate",
    help="Validate OpenMMLab dependencies and checkpoint for a model (no inference run).",
)
def validate_cmd(
    model_id: str = typer.Argument(
        ..., help="Model ID, e.g. rtmpose-s, rtmdet-r2-s, co-dino-inst-vit-l-coco"
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate that the required modules are installed and the checkpoint is cached.

    Does NOT load the model into memory.
    """
    from pathlib import Path  # noqa: F401  (Path used in inner helper)

    payload = _validate_openmmlab_model(model_id)
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    color = "green" if payload["status"] == "ok" else "yellow"
    code = payload.get("structured_error_code", "OK")
    console.print(f"[{color}]{code}[/{color}] — {payload.get('message', '')}")
    if payload["status"] != "ok":
        fix = payload.get("fix", "")
        if isinstance(fix, str):
            console.print(f"  fix: {fix}")
    else:
        console.print(f"  checkpoint: {payload['checkpoint_path']}")


@app.command(
    "model-card",
    help="Show structured pull metadata for an OpenMMLab model (URL, license, inferencer).",
)
def model_card_cmd(
    model_id: str = typer.Argument(..., help="Model ID, e.g. rtmdet-l-coco, rtmpose-m"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    meta = _PULL_METADATA.get(model_id)
    if meta is None:
        payload = {
            "model_id": model_id,
            "code": "CONFIG_REQUIRED",
            "message": f"No pull metadata for {model_id!r}",
            "available": sorted(_PULL_METADATA),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]CONFIG_REQUIRED[/red]: {payload['message']}")
            console.print(f"  available: {', '.join(payload['available'])}")
        raise typer.Exit(2)

    payload = {"model_id": model_id, **meta}
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{model_id}[/bold]  —  {meta.get('display', '')}")
    for key in (
        "license",
        "task",
        "inferencer",
        "config_name",
        "config_repo",
        "model_zoo",
        "checkpoint_filename",
        "download_url",
        "install",
        "note",
    ):
        val = meta.get(key)
        if val:
            console.print(f"  [cyan]{key}[/cyan]: {val}")


__all__ = ["app"]

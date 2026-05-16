# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Expert sidecar install + dry-run commands.

VisionServeX never auto-installs heavy frameworks like OpenMMLab or Detectron2
into the user's environment. Instead, these commands print the exact install
recipe and verify what's missing. Use ``--dry-run`` to see the steps without
running them; use the printed commands to actually install.

Reference:
- OpenMMLab: https://github.com/open-mmlab/mmdetection / mmrotate / mmpose
- Detectron2: https://github.com/facebookresearch/detectron2
- Co-DETR: https://github.com/Sense-X/Co-DETR
- MaskDINO: https://github.com/IDEA-Research/MaskDINO
"""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import asdict, dataclass

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Heavyweight expert sidecars (OpenMMLab, Detectron2, MaskDINO, Co-DETR).",
    no_args_is_help=True,
)
console = Console()


@dataclass
class ExpertInfo:
    name: str
    description: str
    upstream_url: str
    required_modules: tuple[str, ...]
    install_commands: tuple[str, ...]
    structured_error_code: str

    def to_dict(self) -> dict:
        return asdict(self)


# Single source of truth for expert sidecar metadata.
EXPERTS: dict[str, ExpertInfo] = {
    "openmmlab": ExpertInfo(
        name="OpenMMLab (mmdet / mmrotate / mmpose)",
        description="MMDetection + MMRotate + MMPose. Required for Co-DINO, Co-DETR, RTMDet-R/R2, RTMPose, MaskDINO, etc.",
        upstream_url="https://github.com/open-mmlab/mmdetection",
        required_modules=("mmcv", "mmdet"),
        install_commands=(
            "pip install -U openmim",
            "mim install 'mmcv>=2.0.0' 'mmengine>=0.8.0'",
            "mim install 'mmdet>=3.0.0'",
            "mim install 'mmrotate>=1.0.0rc0'  # for OBB models (RTMDet-R/R2)",
            "mim install 'mmpose>=1.0.0'      # for RTMPose",
        ),
        structured_error_code="OPENMMLAB_REQUIRED",
    ),
    "mmdet": ExpertInfo(
        name="MMDetection",
        description="Subset of OpenMMLab — detection only.",
        upstream_url="https://github.com/open-mmlab/mmdetection",
        required_modules=("mmcv", "mmdet"),
        install_commands=(
            "pip install -U openmim",
            "mim install 'mmcv>=2.0.0' 'mmengine>=0.8.0'",
            "mim install 'mmdet>=3.0.0'",
        ),
        structured_error_code="MMDET_REQUIRED",
    ),
    "mmrotate": ExpertInfo(
        name="MMRotate",
        description="Oriented bounding-box (OBB) detection — RTMDet-R/R2, Oriented R-CNN.",
        upstream_url="https://github.com/open-mmlab/mmrotate",
        required_modules=("mmcv", "mmdet", "mmrotate"),
        install_commands=(
            "pip install -U openmim",
            "mim install 'mmcv>=2.0.0' 'mmengine>=0.8.0' 'mmdet>=3.0.0'",
            "mim install 'mmrotate>=1.0.0rc0'",
        ),
        structured_error_code="MMROTATE_REQUIRED",
    ),
    "mmpose": ExpertInfo(
        name="MMPose",
        description="Human pose estimation — RTMPose family.",
        upstream_url="https://github.com/open-mmlab/mmpose",
        required_modules=("mmcv", "mmpose"),
        install_commands=(
            "pip install -U openmim",
            "mim install 'mmcv>=2.0.0' 'mmengine>=0.8.0'",
            "mim install 'mmpose>=1.0.0'",
        ),
        structured_error_code="MMPOSE_REQUIRED",
    ),
    "detectron2": ExpertInfo(
        name="Detectron2",
        description="Facebook AI Research framework. Required for MaskDINO and some Co-DETR builds.",
        upstream_url="https://github.com/facebookresearch/detectron2",
        required_modules=("detectron2",),
        install_commands=(
            "# Install from source (binary wheels are not always available for current torch):",
            "python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'",
            "# Or follow https://detectron2.readthedocs.io/en/latest/tutorials/install.html",
        ),
        structured_error_code="DETECTRON2_REQUIRED",
    ),
    "maskdino": ExpertInfo(
        name="MaskDINO",
        description="MaskDINO instance/panoptic segmentation. Requires Detectron2 + upstream repo.",
        upstream_url="https://github.com/IDEA-Research/MaskDINO",
        required_modules=("detectron2", "maskdino"),
        install_commands=(
            "# Detectron2 (see 'detectron2' recipe above).",
            "git clone https://github.com/IDEA-Research/MaskDINO.git",
            "cd MaskDINO && pip install -e .",
            "# Download checkpoint from the README of the upstream repo.",
        ),
        structured_error_code="MASKDINO_REQUIRED",
    ),
    "co-detr": ExpertInfo(
        name="Co-DETR / Co-DINO",
        description="Collaborative DETR — requires MMDetection.",
        upstream_url="https://github.com/Sense-X/Co-DETR",
        required_modules=("mmcv", "mmdet"),
        install_commands=(
            "# See 'mmdet' recipe to set up the base framework, then:",
            "git clone https://github.com/Sense-X/Co-DETR.git",
            "cd Co-DETR && pip install -e .",
        ),
        structured_error_code="CO_DETR_REQUIRED",
    ),
}


def _module_present(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _missing(info: ExpertInfo) -> list[str]:
    return [m for m in info.required_modules if not _module_present(m)]


@app.command("list")
def list_experts(json_: bool = typer.Option(False, "--json")) -> None:
    """List all expert sidecars with current install status."""
    rows = []
    for key, info in EXPERTS.items():
        missing = _missing(info)
        rows.append(
            {
                "id": key,
                "name": info.name,
                "upstream": info.upstream_url,
                "installed": not missing,
                "missing_modules": missing,
                "error_code_if_missing": info.structured_error_code,
            }
        )
    if json_:
        print(json.dumps(rows, indent=2))
        return

    table = Table(title="Expert sidecars", show_header=True)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Name")
    table.add_column("Installed", no_wrap=True)
    table.add_column("Missing modules")
    table.add_column("Error code", style="dim", no_wrap=True)
    for row in rows:
        installed = "[green]yes[/green]" if row["installed"] else "[yellow]no[/yellow]"
        miss = ", ".join(row["missing_modules"]) or "—"
        table.add_row(row["id"], row["name"], installed, miss, row["error_code_if_missing"])
    console.print(table)


@app.command("install")
def install_expert(
    expert: str = typer.Argument(
        ..., help="Expert sidecar id (e.g. openmmlab, detectron2, maskdino)."
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Print install commands without running them (default: dry-run for safety).",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print the install recipe for an expert sidecar.

    By default this is a dry-run — no commands are executed. Use --no-dry-run
    only if you intend to run the commands yourself in a separate shell.
    VisionServeX never installs heavy frameworks into your environment.
    """
    info = EXPERTS.get(expert.lower())
    if info is None:
        console.print(f"[red]Unknown expert sidecar {expert!r}.[/red] Available: {list(EXPERTS)}")
        raise typer.Exit(2)

    missing = _missing(info)
    payload = {
        "expert": expert,
        "name": info.name,
        "upstream_url": info.upstream_url,
        "required_modules": list(info.required_modules),
        "missing_modules": missing,
        "installed": not missing,
        "structured_error_code_if_missing": info.structured_error_code,
        "install_commands": list(info.install_commands),
        "dry_run": dry_run,
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return

    console.print(f"[bold cyan]{info.name}[/bold cyan]")
    console.print(f"  Upstream: {info.upstream_url}")
    console.print(f"  Required modules: {', '.join(info.required_modules)}")
    if missing:
        console.print(f"  [yellow]Missing:[/yellow] {', '.join(missing)}")
    else:
        console.print("  [green]All required modules already installed.[/green]")

    console.print(
        "\n[bold]Install commands[/bold]"
        + (" [dim](dry-run — not executed)[/dim]" if dry_run else "")
        + ":"
    )
    for c in info.install_commands:
        console.print(f"  [cyan]{c}[/cyan]")

    if not dry_run:
        console.print(
            "\n[red]VisionServeX intentionally does NOT auto-run these commands.[/red] "
            "Copy them into a shell to install."
        )


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Detailed dependency check across all expert sidecars."""
    payload: dict = {"experts": {}}
    for key, info in EXPERTS.items():
        missing = _missing(info)
        payload["experts"][key] = {
            "installed": not missing,
            "missing_modules": missing,
            "structured_error_code": None if not missing else info.structured_error_code,
        }
    payload["mim_available"] = shutil.which("mim") is not None
    payload["pip_available"] = shutil.which("pip") is not None

    if json_:
        print(json.dumps(payload, indent=2))
        return

    table = Table(title="Expert sidecar doctor", show_header=True)
    table.add_column("Expert", style="cyan", no_wrap=True)
    table.add_column("Installed", no_wrap=True)
    table.add_column("Missing")
    table.add_column("Error code if missing", style="dim", no_wrap=True)
    for key, st in payload["experts"].items():
        installed = "[green]yes[/green]" if st["installed"] else "[yellow]no[/yellow]"
        miss = ", ".join(st["missing_modules"]) or "—"
        table.add_row(key, installed, miss, st["structured_error_code"] or "—")
    console.print(table)
    console.print(f"\nmim available: {payload['mim_available']}")
    console.print(f"pip available: {payload['pip_available']}")


__all__ = ["EXPERTS", "ExpertInfo", "app"]

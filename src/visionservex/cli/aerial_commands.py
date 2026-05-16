# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Aerial / drone / remote-sensing domain commands.

Links:
- MMRotate: https://github.com/open-mmlab/mmrotate
- VisDrone: https://github.com/VisDrone/VisDrone-Dataset
- Prithvi EO: https://github.com/NASA-IMPACT/Prithvi-EO-2.0
- IBM/NASA HF: https://huggingface.co/ibm-nasa-geospatial
- DOTA benchmark: https://captain-whu.github.io/DOTA/
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Aerial / drone / remote-sensing domain commands.", no_args_is_help=True)
console = Console()

_AERIAL_MODELS: dict[str, dict] = {
    "rtmdet-r2-s": {
        "task": "oriented_detection",
        "category": "OBB (MMRotate sidecar)",
        "status": "expert_sidecar",
        "install": "pip install openmim && mim install mmcv mmdet mmrotate",
        "pull": "visionservex openmmlab pull rtmdet-r2-s --from-url <URL>",
        "smoke": "visionservex openmmlab smoke-test rtmdet-r2-s",
        "blocker": "Requires mmrotate; no standalone package.",
    },
    "oriented-rcnn": {
        "task": "oriented_detection",
        "category": "OBB (MMRotate sidecar)",
        "status": "expert_sidecar",
        "install": "pip install openmim && mim install mmcv mmdet mmrotate",
        "pull": "visionservex openmmlab pull oriented-rcnn --from-url <URL>",
        "smoke": "visionservex openmmlab smoke-test oriented-rcnn",
        "blocker": "Requires mmrotate.",
    },
    "prithvi-eo-2.0-300m": {
        "task": "geospatial_embed",
        "category": "Remote sensing (HF IBM/NASA)",
        "status": "audit_only",
        "install": "pip install 'visionservex[hf]'",
        "pull": "visionservex model pull prithvi-eo-2.0-300m",
        "smoke": "visionservex embed prithvi-eo-2.0-300m image.tif",
        "blocker": "Multispectral input (6-channel HLS satellite), not standard RGB. Requires rasterio.",
    },
    "remoteclip": {
        "task": "remote_sensing_embed",
        "category": "Remote sensing CLIP (HF)",
        "status": "audit_only",
        "install": "pip install 'visionservex[hf]'",
        "pull": "visionservex model pull remoteclip",
        "smoke": "visionservex embed remoteclip aerial.jpg",
        "blocker": "No registry entry yet; need to verify HF model id and license.",
    },
}


@app.command("doctor")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Check aerial domain dependency availability."""
    import importlib

    checks = {
        "mmrotate (OBB models)": "mmrotate",
        "mmdet (co-dep)": "mmdet",
        "rasterio (multispectral)": "rasterio",
        "owlv2 (drone detection)": "visionservex.engines.owlv2",
        "rfdetr (drone detection)": "visionservex.engines.rfdetr",
    }
    rows = []
    for name, mod in checks.items():
        try:
            importlib.import_module(mod)
            rows.append({"component": name, "status": "available"})
        except ImportError:
            rows.append({"component": name, "status": "not_available"})
    payload = {"components": rows}
    if json_:
        print(json.dumps(payload, indent=2))
        return
    table = Table(title="Aerial domain components", show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", no_wrap=True)
    for r in rows:
        st = (
            "[green]available[/green]" if r["status"] == "available" else "[dim]not_available[/dim]"
        )
        table.add_row(r["component"], st)
    console.print(table)


@app.command("recommend")
def recommend(
    goal: str = typer.Option(
        ..., "--goal", help="e.g. oriented-detection, drone-tracking, satellite-embedding"
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Recommend a model for an aerial task."""
    g = goal.lower().replace(" ", "-").replace("_", "-")
    if "orient" in g or "obb" in g:
        rec = {
            "goal": goal,
            "recommendation": "rtmdet-r2-s (via MMRotate expert sidecar)",
            "install": "pip install openmim && mim install mmcv mmdet mmrotate",
            "validate": "visionservex openmmlab validate rtmdet-r2-s",
            "metric": "OBB mAP50 on DOTA (rotated IoU — NOT axis-aligned IoU)",
            "note": "DOTA-1.5 benchmark. OBB requires rotated IoU; do not reuse detection AP.",
        }
    elif "drone" in g or "track" in g:
        rec = {
            "goal": goal,
            "recommendation": "rfdetr-small + simple-iou tracker",
            "install": "pip install 'visionservex[rfdetr]'",
            "cli": "visionservex video-search index drone_frames/ --detector rfdetr-small --tracker simple-iou --out indexes/drone",
            "metric": "MOTA/MOTP on VisDrone (not AP)",
            "note": "rfdetr-small is runnable CPU/GPU. ByteTrack would improve MOTA.",
        }
    elif "satellite" in g or "geo" in g:
        rec = {
            "goal": goal,
            "recommendation": "prithvi-eo-2.0-300m (multispectral HLS data) — audit_only today",
            "blocker": "Requires 6-channel HLS satellite input + rasterio. Not standard RGB.",
            "workaround": "Use dinov2-base as a fallback for RGB aerial imagery.",
        }
    else:
        rec = {
            "goal": goal,
            "recommendation": "Use rfdetr-small for general aerial detection (Apache-2.0, CPU-capable)",
        }

    if json_:
        print(json.dumps(rec, indent=2))
        return
    for k, v in rec.items():
        console.print(f"  [cyan]{k}:[/cyan] {v}")


@app.command("dataset")
def dataset(
    command: str = typer.Argument(..., help="validate-dota, validate-visdrone, or list"),
    path: Path = typer.Option(None, "--path", help="Dataset root path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Dataset validation and info for aerial benchmarks."""
    if command == "list":
        payload = {
            "supported": ["dota", "visdrone", "xview", "spacenet"],
            "notes": "Pass --path and use validate-dota or validate-visdrone to check layout.",
        }
        if json_:
            print(json.dumps(payload, indent=2))
            return
        console.print("Supported aerial datasets: DOTA, VisDrone, xView, SpaceNet")
        return

    if path is None:
        console.print("[red]--path required for dataset validation.[/red]")
        raise typer.Exit(2)

    if command == "validate-dota":
        required_dirs = ["images", "labelTxt"]
        missing = [d for d in required_dirs if not (path / d).exists()]
        payload = {
            "dataset": "DOTA",
            "path": str(path),
            "required_dirs": required_dirs,
            "missing": missing,
            "valid": not missing,
            "expected_label_format": "<classname> x1 y1 x2 y2 x3 y3 x4 y4 difficulty",
            "metric": "OBB mAP50 with rotated IoU — do not use axis-aligned IoU",
        }
    elif command == "validate-visdrone":
        required_dirs = ["images", "annotations"]
        missing = [d for d in required_dirs if not (path / d).exists()]
        payload = {
            "dataset": "VisDrone",
            "path": str(path),
            "required_dirs": required_dirs,
            "missing": missing,
            "valid": not missing,
            "expected_label_format": "<left>,<top>,<width>,<height>,<score>,<category>,<truncation>,<occlusion>",
            "metric": "MOTA, MOTP for tracking; AP for detection",
        }
    else:
        console.print(
            f"[red]Unknown command {command!r}.[/red] Use: validate-dota, validate-visdrone, list"
        )
        raise typer.Exit(2)

    if json_:
        print(json.dumps(payload, indent=2))
        return
    status = "[green]valid[/green]" if payload["valid"] else "[red]invalid[/red]"
    console.print(f"Dataset: {payload['dataset']} at {path} — {status}")
    if payload["missing"]:
        console.print(f"  Missing dirs: {payload['missing']}")
    console.print(f"  Metric: {payload['metric']}")


__all__ = ["app"]

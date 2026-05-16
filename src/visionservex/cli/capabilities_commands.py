# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Capabilities report command — answers 'what can VisionServeX do right now?'"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from visionservex import __version__
from visionservex.registry import default_registry
from visionservex.runtime.device import available_devices, best_device

app = typer.Typer(help="Generate a comprehensive capabilities report.")
console = Console()

_OPTIONAL_EXTRAS = {
    "server": ["fastapi", "uvicorn"],
    "hf": ["transformers", "huggingface_hub"],
    "rfdetr": ["rfdetr"],
    "onnx": ["onnxruntime"],
    "openmmlab": ["mmpose", "mmdet", "mmrotate", "mmengine"],
    "dev": ["pytest", "ruff"],
}


def _probe_extra(extra: str) -> dict[str, Any]:
    pkgs = _OPTIONAL_EXTRAS.get(extra, [])
    installed = []
    missing = []
    for pkg in pkgs:
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "?")
            installed.append({"package": pkg, "version": ver})
        except ImportError:
            missing.append(pkg)
    return {
        "extra": extra,
        "available": len(missing) == 0,
        "installed_packages": installed,
        "missing_packages": missing,
    }


def _group_models() -> dict[str, Any]:
    reg = default_registry()
    all_entries = reg.list()

    by_task: dict[str, list[dict]] = {}
    by_category: dict[str, list[str]] = {}
    runnable: list[str] = []
    unavailable: list[dict] = []

    for e in all_entries:
        by_task.setdefault(e.task, []).append(
            {
                "id": e.id,
                "category": e.model_category,
                "impl": e.implementation_status,
                "status": e.status,
                "license": e.license,
                "auto_dl": e.auto_download,
                "install_extra": e.install_extra,
            }
        )
        cat = e.model_category or "unknown"
        by_category.setdefault(cat, []).append(e.id)

        if e.implementation_status == "wired" and e.status not in ("external", "manual"):
            runnable.append(e.id)

        if e.model_category == "unavailable_with_reason":
            unavailable.append(
                {
                    "id": e.id,
                    "reason": e.unavailable_reason or "not specified",
                    "task": e.task,
                }
            )

    return {
        "total": len(all_entries),
        "runnable_count": len(runnable),
        "by_task": {k: len(v) for k, v in by_task.items()},
        "by_category": {k: len(v) for k, v in by_category.items()},
        "runnable_model_ids": runnable,
        "unavailable": unavailable,
    }


def _security_status() -> dict[str, Any]:
    from visionservex.config import get_settings

    try:
        settings = get_settings()
        warnings = settings.public_safety_warnings()
        return {
            "auth_enabled": settings.auth.enabled,
            "public_mode": settings.server.public_mode,
            "warnings": warnings,
            "security_level": "low" if warnings else "ok",
        }
    except Exception as exc:
        return {"error": str(exc)[:100]}


def _gpu_status() -> dict[str, Any]:
    try:
        best = best_device()
        return {
            "best_device": best.name,
            "vram_total_gb": best.total_vram_gb,
            "vram_free_gb": best.free_vram_gb,
            "detail": best.detail,
        }
    except Exception as exc:
        return {"error": str(exc)[:100]}


def build_capabilities_payload() -> dict[str, Any]:
    """Build the full capabilities payload."""
    from visionservex.utils.system import collect

    sysinfo = collect()
    devices = [d.to_dict() for d in available_devices()]
    extras = {extra: _probe_extra(extra) for extra in _OPTIONAL_EXTRAS}
    models = _group_models()
    security = _security_status()
    gpu = _gpu_status()

    # Recommended models
    from visionservex.runtime.recommendations import recommend

    recs = {}
    for goal in ("accuracy", "fastest_demo", "best_colab", "best_segmentation", "best_open_vocab"):
        try:
            top = recommend(goal=goal, limit=2)
            recs[goal] = [
                r.entry.id for r in top if r.entry.model_category != "unavailable_with_reason"
            ][:2]
        except Exception:
            recs[goal] = []

    return {
        "version": __version__,
        "python": sys.version,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "system": sysinfo.to_dict(),
        "devices": devices,
        "gpu_status": gpu,
        "installed_extras": extras,
        "models": models,
        "recommendations_by_goal": recs,
        "security": security,
        "known_limitations": [
            "AP50/mAP computation requires user-provided annotated dataset.",
            "DEIM, DEIMv2, RT-DETRv4: experimental_sota stubs — not runnable.",
            "OpenMMLab (RTMPose, RTMDet-R, Co-DINO, InternImage): expert_sidecar — manual install.",
            "TensorRT engine build requires trtexec (not installed by default).",
            "MPS (Apple Silicon): implemented but not maintainer-verified.",
            "AP claims from benchmark-competitiveness require ground-truth annotations.",
        ],
    }


def _format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# VisionServeX Capabilities Report",
        "",
        f"**Version:** {payload['version']}  ",
        f"**Python:** {payload['python'].split()[0]}  ",
        f"**OS:** {payload['os']}  ",
        "",
        "## Devices",
    ]
    for d in payload["devices"]:
        avail = "✓" if d["available"] else "✗"
        vram = f" ({d.get('total_vram_gb')} GB VRAM)" if d.get("total_vram_gb") else ""
        lines.append(f"- **{d['name']}** {avail}{vram}: {d['detail'][:80]}")

    lines += ["", "## Installed Extras"]
    for extra, info in payload["installed_extras"].items():
        status = (
            "✓ available"
            if info["available"]
            else "✗ missing: " + ", ".join(info["missing_packages"])
        )
        lines.append(f"- **{extra}**: {status}")

    models = payload["models"]
    lines += [
        "",
        "## Models",
        f"- Total in registry: {models['total']}",
        f"- Runnable (wired + non-external): {models['runnable_count']}",
        "",
        "### By task",
    ]
    for task, count in sorted(models["by_task"].items()):
        lines.append(f"- {task}: {count}")
    lines += ["", "### By category"]
    for cat, count in sorted(models["by_category"].items()):
        lines.append(f"- {cat}: {count}")

    if models["unavailable"]:
        lines += ["", "### Unavailable models (with reason)"]
        for u in models["unavailable"][:10]:
            lines.append(f"- `{u['id']}` ({u['task']}): {u['reason'][:80]}")

    lines += ["", "## Recommended models by goal"]
    for goal, ids in payload["recommendations_by_goal"].items():
        lines.append(f"- **{goal}**: {', '.join(ids) or 'none'}")

    lines += ["", "## Known limitations"]
    for lim in payload["known_limitations"]:
        lines.append(f"- {lim}")

    return "\n".join(lines)


@app.command(
    "report",
    help="Print a comprehensive capabilities report: devices, extras, models, recommendations.",
)
def report(
    format_: str = typer.Option("human", "--format", help="Output format: human|json|markdown"),
    out: Path | None = typer.Option(None, "--out", help="Write report to this file."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a comprehensive capabilities report."""
    if json_:
        format_ = "json"

    payload = build_capabilities_payload()

    if format_ == "json":
        text = json.dumps(payload, indent=2, default=str)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"Report written to {out}")
        else:
            typer.echo(text)
        return

    if format_ == "markdown":
        text = _format_markdown(payload)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"Report written to {out}")
        else:
            typer.echo(text)
        return

    # Human-readable rich output
    from rich.panel import Panel

    console.print(
        Panel.fit(
            f"[bold]VisionServeX {payload['version']} — Capabilities Report[/bold]\n"
            f"{payload['os']} | Python {payload['python'].split()[0]}",
            border_style="cyan",
        )
    )

    # Devices
    dt = Table(title="Devices", show_lines=False)
    for col in ("Device", "Available", "VRAM", "Detail"):
        dt.add_column(col)
    for d in payload["devices"]:
        avail = "[green]yes[/green]" if d["available"] else "[grey50]no[/grey50]"
        vram = f"{d.get('total_vram_gb')} GB" if d.get("total_vram_gb") else "-"
        dt.add_row(d["name"], avail, vram, d["detail"][:60])
    console.print(dt)

    # Extras
    et = Table(title="Optional extras")
    for col in ("Extra", "Status", "Missing packages"):
        et.add_column(col)
    for extra, info in payload["installed_extras"].items():
        status = (
            "[green]available[/green]" if info["available"] else "[yellow]partial/missing[/yellow]"
        )
        missing = ", ".join(info["missing_packages"]) or "-"
        et.add_row(extra, status, missing)
    console.print(et)

    # Model counts
    models = payload["models"]
    console.print(
        f"\n[bold]Models:[/bold] {models['total']} total, "
        f"[green]{models['runnable_count']} runnable[/green]"
    )
    cat_table = Table(title="By category")
    cat_table.add_column("Category")
    cat_table.add_column("Count")
    for cat, count in sorted(models["by_category"].items()):
        color = {
            "accuracy_grade": "green",
            "production_recommended": "cyan",
            "demo_fast": "yellow",
            "experimental_sota": "magenta",
            "unavailable_with_reason": "red",
        }.get(cat, "white")
        cat_table.add_row(f"[{color}]{cat}[/{color}]", str(count))
    console.print(cat_table)

    # Recommendations
    console.print("\n[bold]Recommended models by goal:[/bold]")
    for goal, ids in payload["recommendations_by_goal"].items():
        ids_str = ", ".join(f"[cyan]{m}[/cyan]" for m in ids) if ids else "[grey50]none[/grey50]"
        console.print(f"  {goal}: {ids_str}")

    # Limitations
    console.print("\n[bold]Known limitations:[/bold]")
    for lim in payload["known_limitations"]:
        console.print(f"  [yellow]•[/yellow] {lim}")


__all__ = ["app", "build_capabilities_payload"]

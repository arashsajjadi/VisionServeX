# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Download audit and metadata inspection commands."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Audit and validate model download metadata.")
console = Console()

_REQUIRED_FIELDS = [
    "download_type",
    "license",
    "upstream_url",
    "difficulty",
    "install_extra",
]

_RECOMMENDED_FIELDS = [
    "minimum_vram_gb",
    "license_notes",
    "auto_download",
]


@app.command("audit", help="Audit download metadata completeness for all registry models.")
def audit(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Scan the registry and report missing download/license/device metadata."""
    from visionservex.registry import default_registry
    from visionservex.runtime.downloads import is_cached

    reg = default_registry()
    issues = []
    counts = {
        "total": 0,
        "auto_downloadable": 0,
        "manual": 0,
        "external": 0,
        "synthetic": 0,
        "missing_required": 0,
        "missing_recommended": 0,
        "cached": 0,
    }

    for entry in reg.list():
        counts["total"] += 1
        model_issues = []

        # Count by download type
        dt = entry.download_type
        if dt == "synthetic":
            counts["synthetic"] += 1
        elif entry.auto_download:
            counts["auto_downloadable"] += 1
        elif dt == "manual":
            counts["manual"] += 1
        elif dt == "external_api":
            counts["external"] += 1

        # Count cached
        try:
            if is_cached(entry):
                counts["cached"] += 1
        except Exception:
            pass

        # Check required fields — skip install_extra for synthetic/external models
        skip_install_extra = dt in {"synthetic", "external_api", "not_available"}
        for field in _REQUIRED_FIELDS:
            if field == "install_extra" and skip_install_extra:
                continue
            val = getattr(entry, field, None)
            if not val and val != 0:
                model_issues.append(f"missing required: {field}")

        # Check recommended fields
        for field in _RECOMMENDED_FIELDS:
            val = getattr(entry, field, None)
            if val is None:
                model_issues.append(f"missing recommended: {field}")

        if any("required" in iss for iss in model_issues):
            counts["missing_required"] += 1
        if any("recommended" in iss for iss in model_issues):
            counts["missing_recommended"] += 1

        if model_issues:
            issues.append({"id": entry.id, "issues": model_issues})

    payload = {
        "counts": counts,
        "issues": issues
        if verbose or json_
        else [i for i in issues if any("required" in x for x in i["issues"])],
        "summary": {
            "clean": len(issues) == 0,
            "models_with_issues": len(issues),
        },
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print(f"\n[bold]Download Audit[/bold] — {counts['total']} models")
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("Auto-downloadable", str(counts["auto_downloadable"]))
    table.add_row("Manual", str(counts["manual"]))
    table.add_row("External/API", str(counts["external"]))
    table.add_row("Built-in (synthetic)", str(counts["synthetic"]))
    table.add_row("Currently cached", str(counts["cached"]))
    table.add_row("Missing required metadata", str(counts["missing_required"]))
    table.add_row("Missing recommended metadata", str(counts["missing_recommended"]))
    console.print(table)

    missing_req = [i for i in issues if any("required" in x for x in i["issues"])]
    if missing_req:
        console.print("\n[red]Models with missing required fields:[/red]")
        for item in missing_req[:20]:
            console.print(
                f"  {item['id']}: {', '.join(i for i in item['issues'] if 'required' in i)}"
            )
    else:
        console.print("\n[green]All models have required metadata.[/green]")

    if verbose and issues:
        console.print("\n[yellow]Models with missing recommended fields:[/yellow]")
        for item in issues:
            rec = [i for i in item["issues"] if "recommended" in i]
            if rec:
                console.print(f"  {item['id']}: {', '.join(rec)}")


@app.command("info", help="Show download and device metadata for a model.")
def info_cmd(model_id: str, json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import cached_path, is_cached

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    cp = cached_path(entry)
    payload = entry.model_dump()
    payload["cached"] = is_cached(entry)
    payload["cache_path"] = str(cp) if cp else None

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    table = Table(title=f"Download info: {model_id}", show_header=False, box=None)
    table.add_column("k", style="cyan")
    table.add_column("v")
    for field in [
        "download_type",
        "auto_download",
        "hf_repo_id",
        "size_bytes",
        "requires_auth",
        "requires_optional_extra",
        "install_extra",
        "difficulty",
        "minimum_vram_gb",
        "recommended_vram_gb",
        "implementation_status",
        "status",
        "license",
        "license_notes",
        "upstream_url",
    ]:
        val = getattr(entry, field, None)
        if val is not None:
            table.add_row(field, str(val))
    table.add_row("cached", str(payload["cached"]))
    if payload["cache_path"]:
        table.add_row("cache_path", payload["cache_path"])
    console.print(table)

    if entry.warnings:
        for w in entry.warnings:
            console.print(f"[yellow]⚠[/yellow]  {w}")


def audit_missing_required_count() -> int:
    """Return the number of models missing required download metadata."""
    from visionservex.registry import default_registry

    reg = default_registry()
    count = 0
    for entry in reg.list():
        dt = entry.download_type
        skip_install_extra = dt in {"synthetic", "external_api", "not_available"}
        for field in _REQUIRED_FIELDS:
            if field == "install_extra" and skip_install_extra:
                continue
            val = getattr(entry, field, None)
            if not val and val != 0:
                count += 1
                break
    return count


__all__ = ["app", "audit_missing_required_count"]

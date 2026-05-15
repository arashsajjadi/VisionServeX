# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Privacy management CLI commands."""

from __future__ import annotations

import json

import typer
from rich.console import Console

app = typer.Typer(help="Privacy controls: retention, cleanup, and inspection.")
console = Console()


@app.command("cleanup", help="Remove temporary upload files created by VisionServeX.")
def cleanup(
    dry_run: bool = typer.Option(False, "--dry-run", help="List files without deleting."),
    json_: bool = typer.Option(False, "--json"),
    tmpdir: str | None = typer.Option(None, "--tmpdir"),
) -> None:
    from visionservex.runtime.temp_manager import cleanup_temp_dir, inspect_temp_dir

    files = inspect_temp_dir(tmpdir)
    if dry_run:
        payload = {"dry_run": True, "files": files, "count": len(files)}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            if files:
                for f in files:
                    console.print(f"  {f['path']}  ({f['size_bytes']} bytes)")
            console.print(f"\n[dim]{len(files)} file(s) would be removed.[/dim]")
        return

    removed = cleanup_temp_dir(tmpdir)
    payload = {"removed": removed}
    if json_:
        typer.echo(json.dumps(payload))
    else:
        console.print(f"[green]Removed {removed} temp file(s).[/green]")


@app.command("inspect-cache", help="List temp and cache files without revealing contents.")
def inspect_cache(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.runtime.temp_manager import inspect_temp_dir

    files = inspect_temp_dir()
    if json_:
        typer.echo(json.dumps({"temp_files": files}, indent=2))
        return
    if not files:
        console.print("[dim]No VisionServeX temp files found.[/dim]")
    else:
        for f in files:
            console.print(f"  {f['path']}  ({f['size_bytes']} bytes)")


@app.command("retention", help="Show or set the data retention policy.")
def retention(
    mode: str | None = typer.Argument(None, help="none|metadata_only|outputs|full_debug"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.config import get_settings

    s = get_settings()
    current = s.privacy.retention_mode
    if mode is None:
        payload = {
            "current_mode": current,
            "save_inputs": s.privacy.save_inputs,
            "save_outputs": s.privacy.save_outputs,
            "job_payload_retention": s.privacy.job_payload_retention,
            "explanation": {
                "none": "Nothing persisted. Memory only.",
                "metadata_only": "Only request_id, model_id, status, timestamps, latency.",
                "outputs": "metadata + annotated output images if --save used.",
                "full_debug": "Everything including payloads. For development only.",
            }.get(current, "unknown"),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"Retention mode: [cyan]{current}[/cyan]")
            console.print(f"  Save inputs:  {s.privacy.save_inputs}")
            console.print(f"  Save outputs: {s.privacy.save_outputs}")
        return

    valid = ("none", "metadata_only", "outputs", "full_debug")
    if mode not in valid:
        console.print(f"[red]Invalid mode. Choose: {', '.join(valid)}[/red]")
        raise typer.Exit(1)

    if mode == "full_debug":
        console.print(
            "[yellow]WARNING: full_debug stores all payloads including images. Development use only.[/yellow]"
        )

    console.print(f"Set: [cyan]export VISIONSERVEX_PRIVACY__RETENTION_MODE={mode}[/cyan]")


__all__ = ["app"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model lifecycle CLI: pull, cache, checkpoint-info, verify, remove, list-local."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Model lifecycle: pull, cache, checkpoint info, verify, register.")
console = Console()


@app.command("info", help="Show full model info from registry + cache status.")
def model_info(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import cached_path, is_cached

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        if json_:
            typer.echo(json.dumps({"error": str(exc)}, indent=2))
        else:
            console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1)

    cp = cached_path(entry)
    size_bytes = 0
    if cp and cp.exists():
        if cp.is_dir():
            size_bytes = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
        else:
            size_bytes = cp.stat().st_size

    payload = entry.model_dump()
    payload["cached"] = is_cached(entry)
    payload["cache_path"] = str(cp) if cp else None
    payload["cache_size_mb"] = round(size_bytes / (1024 * 1024), 2) if size_bytes else 0
    payload["checkpoint_source"] = entry.download_type
    payload["checkpoint_trust_level"] = (
        "community_hf"
        if entry.download_type == "huggingface"
        else "package_managed"
        if entry.download_type == "package_managed"
        else "manual"
    )

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    from rich.panel import Panel

    cat_color = {
        "accuracy_grade": "green",
        "production_recommended": "cyan",
        "demo_fast": "yellow",
        "experimental_sota": "magenta",
        "expert_sidecar": "grey50",
        "unavailable_with_reason": "red",
    }.get(entry.model_category or "", "white")

    console.print(
        Panel.fit(
            f"[bold]{entry.display_name}[/bold]\n"
            f"ID: {entry.id} | Task: {entry.task} | "
            f"Category: [{cat_color}]{entry.model_category}[/{cat_color}]",
            border_style="cyan",
        )
    )
    cached = "[green]yes[/green]" if is_cached(entry) else "[grey50]no[/grey50]"
    console.print(f"  Cached:      {cached}")
    if cp:
        console.print(f"  Cache path:  {cp}")
        if size_bytes:
            console.print(f"  Size:        {size_bytes / (1024 * 1024):.1f} MB")
    console.print(
        f"  License:     {entry.license}{'  ⚠ uncertain' if entry.license_uncertain else ''}"
    )
    console.print(f"  Status:      {entry.status} / {entry.implementation_status}")
    console.print(f"  Upstream:    {entry.upstream_url}")
    if entry.auto_download:
        console.print(f"\n  [cyan]$[/cyan] visionservex model pull {entry.id}")
    else:
        console.print(f"\n  [yellow]Manual download required.[/yellow] See: {entry.upstream_url}")


@app.command("pull", help="Download model checkpoint.")
def pull_model(
    model_id: str,
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be downloaded without downloading."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import (
        DownloadError,
        ManualDownloadRequired,
        download,
        is_cached,
    )

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    if dry_run:
        payload = {
            "model_id": model_id,
            "download_type": entry.download_type,
            "hf_repo_id": getattr(entry, "hf_repo_id", None),
            "auto_download": entry.auto_download,
            "already_cached": is_cached(entry),
            "dry_run": True,
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[dim]dry-run:[/dim] would download {model_id}")
            console.print(f"  Source: {entry.download_type}")
            if getattr(entry, "hf_repo_id", None):
                console.print(f"  HF repo: {entry.hf_repo_id}")
            console.print(f"  Already cached: {'yes' if is_cached(entry) else 'no'}")
        return

    try:
        path = download(entry, force=force)
        payload = {"model_id": model_id, "path": str(path), "status": "ok"}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]✓[/green] {model_id} → {path}")
    except ManualDownloadRequired as exc:
        _die(str(exc), json_mode=json_, code="MANUAL_DOWNLOAD_REQUIRED", exit_code=2)
    except DownloadError as exc:
        _die(str(exc), json_mode=json_, code="DOWNLOAD_FAILED")


@app.command("checkpoint-info", help="Show checkpoint provenance and trust metadata.")
def checkpoint_info(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import cached_path, is_cached

    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    cp = cached_path(entry)
    payload = {
        "model_id": model_id,
        "source": entry.download_type,
        "hf_repo_id": getattr(entry, "hf_repo_id", None),
        "upstream_url": entry.upstream_url,
        "license": entry.license,
        "license_uncertain": entry.license_uncertain or False,
        "implementation_status": entry.implementation_status,
        "checkpoint_trust_level": (
            "community_hf"
            if entry.download_type == "huggingface"
            else "package_managed"
            if entry.download_type == "package_managed"
            else "manual"
            if entry.download_type == "manual"
            else "synthetic"
        ),
        "cached": is_cached(entry),
        "cache_path": str(cp) if cp else None,
        "official_ap_claim": "see model-card for upstream benchmark claims",
        "verified_by_visionservex": "latency_tested_only — use benchmark-competitiveness --dataset for AP",
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    console.print(f"[bold]Checkpoint info:[/bold] {model_id}")
    console.print(f"  Source:          {payload['source']}")
    if payload["hf_repo_id"]:
        console.print(f"  HF repo:         {payload['hf_repo_id']}")
    console.print(
        f"  License:         {payload['license']}{'  ⚠ uncertain' if payload['license_uncertain'] else ''}"
    )
    console.print(f"  Trust level:     {payload['checkpoint_trust_level']}")
    console.print(f"  Cached:          {'yes' if payload['cached'] else 'no'}")
    if payload["cache_path"]:
        console.print(f"  Cache path:      {payload['cache_path']}")
    console.print(f"\n  [dim]{payload['verified_by_visionservex']}[/dim]")


@app.command("cache", help="Show cache info for a model.")
def model_cache(
    model_id: str | None = typer.Argument(None, help="Model ID (omit for all)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import default_registry
    from visionservex.runtime.downloads import cache_listing, cached_path, is_cached

    if model_id:
        try:
            entry = default_registry().get(model_id)
        except Exception as exc:
            _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
            return
        cp = cached_path(entry)
        size = 0
        if cp and cp.exists():
            if cp.is_dir():
                size = sum(f.stat().st_size for f in cp.rglob("*") if f.is_file())
            else:
                size = cp.stat().st_size
        payload = {
            "model_id": model_id,
            "cached": is_cached(entry),
            "path": str(cp) if cp else None,
            "size_mb": round(size / (1024 * 1024), 2),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            cached = "[green]yes[/green]" if payload["cached"] else "[grey50]no[/grey50]"
            console.print(
                f"{model_id}: cached={cached} path={payload['path']} size={payload['size_mb']} MB"
            )
        return

    items = cache_listing()
    if json_:
        typer.echo(json.dumps(items, indent=2))
        return
    if not items:
        console.print("No models cached.")
        return
    table = Table(title="Cached models")
    table.add_column("ID")
    table.add_column("Size MiB")
    table.add_column("Path")
    for item in items:
        table.add_row(item["model_id"], f"{item['size_bytes'] / (1024 * 1024):.1f}", item["path"])
    console.print(table)


@app.command("verify", help="Verify cached model files.")
def verify_model(
    model_id: str | None = typer.Argument(None),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.downloads import cache_verify

    report = cache_verify(model_id)
    if json_:
        typer.echo(json.dumps(report, indent=2))
        return
    if not report:
        console.print("nothing cached to verify")
        return
    for r in report:
        status = "[green]ok[/green]" if r["ok"] else "[red]bad[/red]"
        console.print(f"  {r['model_id']}: {status} — {r['reason']}")


@app.command("clear-cache", help="Delete cached files for a model.")
def clear_cache_cmd(
    model_id: str,
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if not yes and not typer.confirm(f"Delete cached files for {model_id}?"):
        raise typer.Exit(1)
    from visionservex.runtime.downloads import cache_clean

    freed = cache_clean(model_id)
    payload = {"model_id": model_id, "bytes_freed": freed}
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[green]freed {freed / (1024 * 1024):.1f} MiB[/green] from {model_id}")


@app.command("list-local", help="List all locally cached models.")
def list_local(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.runtime.downloads import cache_listing

    items = cache_listing()
    if json_:
        typer.echo(json.dumps(items, indent=2))
        return
    if not items:
        console.print("[grey50]No models cached.[/grey50]")
        return
    table = Table(title=f"Locally cached models ({len(items)})")
    table.add_column("Model ID")
    table.add_column("Size")
    table.add_column("Path")
    for item in items:
        size = f"{item['size_bytes'] / (1024 * 1024):.1f} MiB"
        table.add_row(item["model_id"], size, item["path"][:60])
    console.print(table)


def _die(message: str, *, json_mode: bool, code: str = "ERROR", exit_code: int = 1) -> None:
    if json_mode:
        typer.echo(json.dumps({"error": {"code": code, "message": message}}, indent=2), err=True)
    else:
        console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(exit_code)


__all__ = ["app"]

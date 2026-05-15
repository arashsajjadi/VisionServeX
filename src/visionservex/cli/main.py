# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""VisionServeX command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from visionservex import __version__
from visionservex.cli import (
    benchmark_commands,
    downloads_commands,
    gateway_commands,
    gpu_commands,
    openmmlab_commands,
    suite_commands,
    tensorrt_commands,
)
from visionservex.cli import tunnel as tunnel_cli
from visionservex.config import get_settings, reload_settings
from visionservex.core.model import VisionModel
from visionservex.registry import RegistryError, default_registry
from visionservex.runtime.device import available_devices, best_device
from visionservex.runtime.downloads import (
    DownloadError,
    DownloadProgress,
    ManualDownloadRequired,
    cache_clean,
    cache_listing,
    cache_repair,
    cache_root,
    cache_verify,
    cached_path,
    download,
    is_cached,
)
from visionservex.runtime.recommendations import first_beginner_pick, recommend
from visionservex.utils.logging import configure_logging
from visionservex.utils.system import collect, probe_dependencies

app = typer.Typer(
    name="visionservex",
    help="VisionServeX — serve permissive-license computer vision models locally and over Cloudflare Tunnel.",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

cache_app = typer.Typer(help="Manage the local model cache.")
config_app = typer.Typer(help="Show or set configuration values.")
example_app = typer.Typer(help="Run beginner examples.")
app.add_typer(cache_app, name="cache")
app.add_typer(config_app, name="config")
app.add_typer(example_app, name="run-example")
app.add_typer(tunnel_cli.app, name="tunnel")
app.add_typer(gpu_commands.app, name="gpu")
app.add_typer(benchmark_commands.app, name="benchmark", invoke_without_command=True)
app.add_typer(downloads_commands.app, name="downloads")
app.add_typer(openmmlab_commands.app, name="openmmlab")
app.add_typer(tensorrt_commands.app, name="tensorrt")
app.add_typer(gateway_commands.app, name="gateway")
app.add_typer(suite_commands.suite_app, name="suite")
app.add_typer(suite_commands.scheduler_app, name="scheduler")

console = Console()


# ---------- helpers ----------


def _emit(payload, *, json_mode: bool, summary: str | None = None) -> None:
    if json_mode:
        typer.echo(json.dumps(payload, indent=2, default=str))
    elif summary is not None:
        console.print(summary)
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


def _die(
    message: str, *, json_mode: bool, code: str = "ERROR", hint: str = "", exit_code: int = 1
) -> None:
    if json_mode:
        payload = {"error": {"code": code, "message": message, "hint": hint}}
        typer.echo(json.dumps(payload), err=True)
    else:
        console.print(f"[red]error:[/red] {message}")
        if hint:
            console.print(f"  [yellow]hint:[/yellow] {hint}")
    raise typer.Exit(exit_code)


@app.callback()
def _global(
    debug: bool = typer.Option(False, "--debug", help="Enable verbose logs and stack traces."),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help="Path to a YAML config file (also honored via VISIONSERVEX_CONFIG_FILE).",
    ),
) -> None:
    """Top-level options."""
    if debug:
        configure_logging("DEBUG")
        app.pretty_exceptions_enable = True
    else:
        configure_logging("INFO")
    if config_file:
        import os

        os.environ["VISIONSERVEX_CONFIG_FILE"] = str(config_file)
        reload_settings()


# ---------- meta ----------


@app.command(
    "getting-started",
    help="Beginner guide: check device, recommend a model, show exact next commands.",
)
def getting_started() -> None:
    get_settings()
    best = best_device()
    pick = first_beginner_pick(task="detect")
    first_beginner_pick(task="foundation_segment")

    console.print(
        Panel.fit(
            f"[bold]VisionServeX {__version__} — Getting Started[/bold]\n"
            f"by Arash Sajjadi · Apache-2.0",
            border_style="green",
        )
    )
    if best.name != "cpu":
        console.print(f"[green]GPU detected:[/green] {best.detail}")
        if best.total_vram_gb:
            console.print(
                f"  VRAM: {best.total_vram_gb} GB total, {best.free_vram_gb or '?'} GB free"
            )
    else:
        console.print(
            "[yellow]No GPU detected.[/yellow] VisionServeX runs on CPU (slower for large models)."
        )

    console.print()
    console.print("[bold]Step 1 — Get a full diagnosis:[/bold]")
    console.print("  [cyan]$[/cyan] visionservex doctor")
    console.print()
    console.print("[bold]Step 2 — See available models:[/bold]")
    console.print("  [cyan]$[/cyan] visionservex list-models --easy")
    console.print()
    console.print("[bold]Step 3 — Get a recommendation:[/bold]")
    console.print("  [cyan]$[/cyan] visionservex recommend --task detect --simple")
    console.print()
    if pick:
        console.print(f"[bold]Step 4 — Download {pick.id} (recommended for detection):[/bold]")
        console.print(f"  [cyan]$[/cyan] visionservex pull {pick.id}")
        console.print()
        console.print("[bold]Step 5 — Run a prediction:[/bold]")
        console.print(
            f"  [cyan]$[/cyan] visionservex predict {pick.id} examples/images/street.jpg --save outputs/out.jpg"
        )
    console.print()
    console.print("[bold]Step 6 — Start the API server:[/bold]")
    console.print("  [cyan]$[/cyan] visionservex serve")
    console.print()
    console.print("[bold]Step 7 — Call the API:[/bold]")
    console.print("  [cyan]$[/cyan] curl -F 'image=@examples/images/street.jpg' \\")
    console.print("       -F 'model_id=mock-detect' http://127.0.0.1:8080/detect | jq")
    console.print()
    console.print("[dim]For public exposure, see `visionservex tunnel doctor`.[/dim]")


@app.command(help="Show package version, server status, cache info, and next recommended action.")
def status(json_: bool = typer.Option(False, "--json")) -> None:
    settings = get_settings()
    best = best_device()
    from visionservex.runtime.downloads import cache_listing

    cached = cache_listing()
    pick = first_beginner_pick(task="detect")
    payload = {
        "version": __version__,
        "device": best.to_dict(),
        "cache_dir": str(settings.cache.cache_dir),
        "cached_models": len(cached),
        "cached_model_ids": [c["model_id"] for c in cached],
        "auth_enabled": settings.auth.enabled,
        "server_bind": f"{settings.server.host}:{settings.server.port}",
        "auto_pull": settings.models.auto_pull,
        "beginner_pick": pick.id if pick else None,
    }
    if json_:
        _emit(payload, json_mode=True)
        return
    console.print(
        Panel.fit(
            f"[bold]VisionServeX {__version__}[/bold]",
            border_style="cyan",
        )
    )
    console.print(f"Device:  {best.name} — {best.detail}")
    console.print(f"Cache:   {settings.cache.cache_dir} ({len(cached)} model(s) cached)")
    if cached:
        console.print(f"  Cached: {', '.join(c['model_id'] for c in cached[:6])}")
    console.print(f"Auth:    {'enabled' if settings.auth.enabled else 'disabled'}")
    console.print(f"Server:  {settings.server.host}:{settings.server.port}")
    if pick:
        console.print(f"\nNext recommended action: [cyan]visionservex pull {pick.id}[/cyan]")


@app.command(help="Print version information.")
def version(json_: bool = typer.Option(False, "--json")) -> None:
    payload = {
        "version": __version__,
        "author": "Arash Sajjadi <arash.sajjadi@usask.ca>",
        "affiliation": "PhD Candidate, Department of Computer Science, University of Saskatchewan",
        "supervisor": "Prof. Mark Eramian, University of Saskatchewan",
        "license": "Apache-2.0",
    }
    _emit(
        payload,
        json_mode=json_,
        summary=f"VisionServeX {__version__} — by Arash Sajjadi (Apache-2.0)",
    )


# ---------- doctor ----------


@app.command(help="Run friendly diagnostics: system, devices, dependencies, recommendations.")
def doctor(
    json_: bool = typer.Option(False, "--json"),
    fix_suggestions: bool = typer.Option(
        False, "--fix-suggestions", help="Print actionable fix commands."
    ),
) -> None:
    settings = get_settings()
    sysinfo = collect()
    devices = [d.to_dict() for d in available_devices()]
    deps = probe_dependencies()
    warnings = settings.public_safety_warnings()
    best = best_device()
    pick = first_beginner_pick(task="detect")

    payload = {
        "system": sysinfo.to_dict(),
        "devices": devices,
        "best_device": best.to_dict(),
        "dependencies": deps,
        "cache_dir": str(settings.cache.cache_dir),
        "auth_enabled": settings.auth.enabled,
        "server_host": settings.server.host,
        "server_port": settings.server.port,
        "public_mode": settings.server.public_mode,
        "warnings": warnings,
        "beginner_pick": pick.id if pick else None,
        "next_commands": _next_commands(pick),
    }
    if fix_suggestions:
        _print_fix_suggestions(deps, payload["warnings"])
        return
    if json_:
        payload["fix_suggestions"] = _compute_fix_suggestions(deps, payload["warnings"])
        _emit(payload, json_mode=True)
        return

    _print_doctor_human(payload)


def _compute_fix_suggestions(deps: dict, warnings: list[str]) -> list[dict]:
    suggestions = []
    if not deps.get("torch", {}).get("installed"):
        suggestions.append(
            {
                "issue": "PyTorch not installed",
                "fix": "pip install 'visionservex[torch]'",
                "docs": "docs/installation.md",
            }
        )
    if not deps.get("transformers", {}).get("installed"):
        suggestions.append(
            {
                "issue": "Hugging Face Transformers not installed",
                "fix": "pip install 'visionservex[hf]'",
                "docs": "docs/installation.md",
            }
        )
    if not deps.get("fastapi", {}).get("installed"):
        suggestions.append(
            {
                "issue": "FastAPI not installed (server mode unavailable)",
                "fix": "pip install 'visionservex[server]'",
                "docs": "docs/installation.md",
            }
        )
    for w in warnings:
        if "AUTH" in w:
            suggestions.append(
                {
                    "issue": w,
                    "fix": "export VISIONSERVEX_AUTH__ENABLED=true && "
                    'export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")',
                    "docs": "docs/security.md",
                }
            )
    return suggestions


def _print_fix_suggestions(deps: dict, warnings: list[str]) -> None:
    suggestions = _compute_fix_suggestions(deps, warnings)
    if not suggestions:
        console.print("[green]No actionable issues found.[/green]")
        return
    console.print("[bold]Fix suggestions:[/bold]")
    for s in suggestions:
        console.print(f"  [yellow]Issue:[/yellow] {s['issue']}")
        console.print(f"  [cyan]Fix  :[/cyan] {s['fix']}")
        if s.get("docs"):
            console.print(f"  [blue]Docs :[/blue] {s['docs']}")
        console.print()


def _next_commands(pick) -> list[str]:
    cmds = []
    if pick is None:
        return ["visionservex list-models", "visionservex predict mock-detect any.jpg"]
    cmds.append(f"visionservex info {pick.id}")
    if pick.auto_download:
        cmds.append(f"visionservex pull {pick.id}")
    cmds.append(f"visionservex predict {pick.id} examples/images/street.jpg --save outputs/out.jpg")
    cmds.append("visionservex serve")
    return cmds


def _print_doctor_human(p: dict) -> None:
    sys_ = p["system"]
    best = p["best_device"]
    console.print(
        Panel.fit(
            f"[bold]VisionServeX[/bold] {__version__}\n"
            f"by Arash Sajjadi · University of Saskatchewan",
            border_style="cyan",
        )
    )

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("OS", f"{sys_['os']['name']} {sys_['os']['release']} ({sys_['os']['machine']})")
    table.add_row("Python", sys_["python"])
    table.add_row("Package path", sys_["package_install_path"])
    table.add_row("Cache dir", sys_["cache_path"])
    mem = sys_["memory"]
    table.add_row(
        "Memory", f"{mem['total_gb']:.1f} GB total · {mem['available_gb']:.1f} GB available"
    )
    dsk = sys_["disk"]
    table.add_row("Disk (cache)", f"{dsk['free_gb']:.1f} GB free of {dsk['total_gb']:.1f} GB")
    table.add_row(
        "CPU",
        f"{sys_['cpu'].get('logical_cores')} logical cores ({sys_['cpu'].get('brand') or 'unknown'})",
    )
    console.print(table)

    console.print()
    dt = Table(title="Devices", show_lines=False)
    dt.add_column("Name")
    dt.add_column("Available")
    dt.add_column("Detail")
    dt.add_column("VRAM")
    for d in p["devices"]:
        avail = "[green]yes[/green]" if d["available"] else "[grey50]no[/grey50]"
        vram = ""
        if d.get("total_vram_gb"):
            vram = f"{d['total_vram_gb']} GB"
            if d.get("free_vram_gb"):
                vram += f" ({d['free_vram_gb']} free)"
        dt.add_row(d["name"], avail, d["detail"][:80], vram)
    console.print(dt)

    console.print()
    deptab = Table(title="Optional dependencies (only matters for real backends)")
    deptab.add_column("Package")
    deptab.add_column("Installed")
    deptab.add_column("Version")
    deptab.add_column("Install hint")
    for name, info in p["dependencies"].items():
        installed = "[green]yes[/green]" if info["installed"] else "[grey50]no[/grey50]"
        deptab.add_row(name, installed, str(info.get("version") or "-"), info.get("hint") or "")
    console.print(deptab)

    console.print()
    for w in p["warnings"]:
        console.print(f"[yellow]warning:[/yellow] {w}")

    console.print()
    if best.get("available") and best.get("name") != "cpu":
        console.print(
            f"[green]Good news:[/green] VisionServeX can use [bold]{best['name'].upper()}[/bold] on your system."
        )
        if best.get("total_vram_gb"):
            console.print(
                f"GPU memory: {best['total_vram_gb']} GB total · {best.get('free_vram_gb') or '?'} GB free."
            )
    else:
        console.print("[yellow]Note:[/yellow] no GPU detected. VisionServeX can still run on CPU.")

    if p["beginner_pick"]:
        console.print(f"\n[bold]Recommended first model:[/bold] {p['beginner_pick']}")
    console.print("\n[bold]Run this next:[/bold]")
    for cmd in p["next_commands"]:
        console.print(f"  [cyan]$[/cyan] {cmd}")


# ---------- devices ----------


@app.command(help="Show available compute devices with sanity check status.")
def devices(
    json_: bool = typer.Option(False, "--json"),
    benchmark_: bool = typer.Option(
        False, "--benchmark", help="Run a tiny synthetic benchmark on available devices."
    ),
    quick: bool = typer.Option(True, "--quick/--full", help="Quick benchmark (default) or full."),
) -> None:
    from visionservex.runtime.device import device_benchmark

    items = [d.to_dict() for d in available_devices()]
    if benchmark_:
        bm_results = {}
        for d in items:
            if d["available"] and d.get("sanity_ok") is not False:
                bm = device_benchmark(d["name"], quick=quick)
                bm_results[d["name"]] = bm
        if json_:
            _emit({"devices": items, "benchmark": bm_results}, json_mode=True)
            return
        table = Table(title="Device benchmark")
        table.add_column("Device")
        table.add_column("Available")
        table.add_column("VRAM")
        table.add_column("Sanity")
        table.add_column("Avg ms")
        table.add_column("GFLOPS")
        table.add_column("Detail")
        for d in items:
            avail = "[green]yes[/green]" if d["available"] else "[grey50]no[/grey50]"
            sanity = (
                "[green]ok[/green]"
                if d.get("sanity_ok")
                else ("[red]FAIL[/red]" if d.get("sanity_ok") is False else "-")
            )
            vram = (
                f"{d['total_vram_gb']}/{d.get('free_vram_gb') or '?'} GB"
                if d.get("total_vram_gb")
                else "-"
            )
            bm = bm_results.get(d["name"], {})
            table.add_row(
                d["name"],
                avail,
                vram,
                sanity,
                f"{bm.get('avg_ms', '?')} ms" if bm.get("ok") else "-",
                f"{bm.get('throughput_gflops', '?')}" if bm.get("ok") else "-",
                d["detail"][:60],
            )
        console.print(table)
        return
    if json_:
        _emit(items, json_mode=True)
        return
    table = Table(title="Compute devices")
    table.add_column("Name")
    table.add_column("Available")
    table.add_column("Sanity")
    table.add_column("Detail")
    table.add_column("VRAM (total/free)")
    for d in items:
        avail = "[green]yes[/green]" if d["available"] else "[grey50]no[/grey50]"
        sanity = (
            "[green]ok[/green]"
            if d.get("sanity_ok")
            else ("[red]FAIL[/red]" if d.get("sanity_ok") is False else "[grey50]-[/grey50]")
        )
        vram = "-"
        if d.get("total_vram_gb"):
            vram = f"{d['total_vram_gb']} / {d.get('free_vram_gb') or '?'} GB"
        table.add_row(d["name"], avail, sanity, d["detail"][:70], vram)
    console.print(table)


# ---------- list / info / recommend ----------


@app.command("list-models", help="List models available in the registry.")
def list_models(
    task: str | None = typer.Option(None, "--task"),
    status: str | None = typer.Option(None, "--status"),
    family: str | None = typer.Option(None, "--family"),
    easy: bool = typer.Option(False, "--easy", help="Show only beginner-friendly models."),
    can_run: bool = typer.Option(
        False, "--can-run", help="Show only models that can run on current devices."
    ),
    friendly: bool = typer.Option(
        False, "--friendly", help="Human-readable table with more details."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    reg = default_registry()
    entries = reg.list(task=task, status=status, family=family)  # type: ignore[arg-type]
    if easy:
        entries = [e for e in entries if e.beginner_recommendation]
    if can_run:
        avail = {d.name for d in available_devices() if d.available}
        entries = [e for e in entries if set(e.supported_devices) & avail]

    if json_:
        _emit([e.model_dump() for e in entries], json_mode=True)
        return

    if friendly:
        table = Table(
            title=f"Models ({len(entries)}) — use `visionservex info <id>` for full details"
        )
        for col in (
            "Model ID",
            "Task",
            "Difficulty",
            "Status",
            "Impl",
            "Auto-DL",
            "License",
            "Best for",
        ):
            table.add_column(col)
        for e in entries:
            impl_color = {"wired": "green", "partial": "yellow", "stub": "grey50"}.get(
                e.implementation_status, "white"
            )
            best_for = (e.best_for[0] if e.best_for else "") or ""
            table.add_row(
                e.id,
                e.task,
                e.difficulty,
                e.status,
                f"[{impl_color}]{e.implementation_status}[/{impl_color}]",
                "[green]yes[/green]" if e.auto_download else "[grey50]no[/grey50]",
                e.license,
                best_for[:30],
            )
        console.print(table)
        console.print(
            "\nLegend: impl=[green]wired[/green]=real backend, [yellow]partial[/yellow]=in progress, [grey50]stub[/grey50]=registry only"
        )
        return

    table = Table(title=f"Models ({len(entries)})")
    for col in ("id", "task", "status", "impl", "diff", "license", "devices", "auto-DL"):
        table.add_column(col)
    for e in entries:
        impl_color = {"wired": "green", "partial": "yellow", "stub": "grey50"}.get(
            e.implementation_status, "white"
        )
        table.add_row(
            e.id,
            e.task,
            e.status,
            f"[{impl_color}]{e.implementation_status}[/{impl_color}]",
            e.difficulty,
            e.license + (" (?)" if e.license_uncertain else ""),
            ",".join(e.supported_devices),
            "[green]yes[/green]" if e.auto_download else "[grey50]no[/grey50]",
        )
    console.print(table)


@app.command(help="Show full details for a model.")
def info(model_id: str, json_: bool = typer.Option(False, "--json")) -> None:
    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return
    payload = entry.model_dump()
    payload["cached"] = is_cached(entry)
    cp = cached_path(entry)
    payload["cache_path"] = str(cp) if cp else None
    _emit(payload, json_mode=json_)


@app.command(help="Recommend a model for the current device + task.")
def recommend_cmd(
    task: str | None = typer.Option(None, "--task"),
    device: str | None = typer.Option(None, "--device"),
    vram: float | None = typer.Option(None, "--vram", help="Available VRAM in GB."),
    simple: bool = typer.Option(False, "--simple", help="Prefer beginner-friendly models."),
    limit: int = typer.Option(5, "--limit"),
    json_: bool = typer.Option(False, "--json"),
):
    recs = recommend(task=task, device=device, vram_gb=vram, simple=simple, limit=limit)
    if json_:
        _emit([r.to_dict() for r in recs], json_mode=True)
        return
    if not recs:
        console.print("[yellow]No recommendations found for the given filters.[/yellow]")
        return
    table = Table(title="Recommendations")
    for col in ("id", "task", "status", "impl", "score", "device", "license"):
        table.add_column(col)
    for r in recs:
        e = r.entry
        table.add_row(
            e.id,
            e.task,
            e.status,
            e.implementation_status,
            f"{r.score:.1f}",
            ",".join(e.supported_devices),
            e.license,
        )
    console.print(table)
    top = recs[0].entry
    console.print(f"\nTop pick: [bold]{top.id}[/bold]")
    if top.auto_download:
        console.print(f"  [cyan]$[/cyan] visionservex pull {top.id}")
    else:
        console.print(
            f"  [cyan]$[/cyan] visionservex info {top.id}  # then follow upstream instructions"
        )


# we expose it as `recommend` for the CLI surface
app.command("recommend", help="Recommend a model for the current device + task.")(recommend_cmd)


# ---------- pull / pull-easy / pull-all ----------


def _pull_with_progress(entry, *, force: bool = False, offline: bool = False) -> Path:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"pulling {entry.id}", total=None)

        def _cb(ev: DownloadProgress) -> None:
            if ev.total_bytes:
                progress.update(task_id, total=ev.total_bytes)
            progress.update(
                task_id, completed=ev.downloaded_bytes, description=f"{entry.id}: {ev.phase}"
            )

        path = download(entry, progress=_cb, force=force, offline=offline)
        progress.update(task_id, description=f"{entry.id}: done")
        return path


@app.command(help="Download a model's checkpoint.")
def pull(
    model_id: str,
    force: bool = typer.Option(False, "--force"),
    offline: bool = typer.Option(False, "--offline"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    try:
        if json_:
            path = download(entry, force=force, offline=offline)
        else:
            path = _pull_with_progress(entry, force=force, offline=offline)
    except ManualDownloadRequired as exc:
        _die(
            str(exc),
            json_mode=json_,
            code="MANUAL_DOWNLOAD_REQUIRED",
            hint=f"see {entry.upstream_url}",
            exit_code=2,
        )
        return
    except DownloadError as exc:
        _die(str(exc), json_mode=json_, code="DOWNLOAD_FAILED")
        return

    _emit({"model_id": entry.id, "path": str(path)}, json_mode=json_, summary=f"saved to {path}")


@app.command("pull-easy", help="Download all beginner-friendly auto-downloadable models.")
def pull_easy(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    reg = default_registry()
    entries = [e for e in reg.list() if e.beginner_recommendation and e.auto_download]
    _pull_many(entries, yes=yes, json_mode=json_, label="beginner-friendly")


@app.command("pull-recommended", help="Download the top recommendation for each common task.")
def pull_recommended(
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    tasks = ["detect", "segment", "classify", "pose", "open_vocab_detect"]
    chosen = []
    for t in tasks:
        recs = recommend(task=t, simple=True, limit=1)
        if recs and recs[0].entry.auto_download:
            chosen.append(recs[0].entry)
    _pull_many(chosen, yes=yes, json_mode=json_, label="recommended")


@app.command("pull-all", help="Download many models. Use --task or --only-auto-downloadable.")
def pull_all(
    task: str | None = typer.Option(None, "--task"),
    only_auto: bool = typer.Option(True, "--only-auto-downloadable/--include-non-auto"),
    yes: bool = typer.Option(False, "--yes-i-understand-large-downloads"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    reg = default_registry()
    entries = reg.list(task=task)  # type: ignore[arg-type]
    if only_auto:
        entries = [e for e in entries if e.auto_download]
    if not yes and not json_:
        total = sum((e.size_bytes or 0) for e in entries)
        console.print(f"About to download {len(entries)} models.")
        if total:
            console.print(f"Approximate total size: {total / 1e9:.2f} GB")
        if not typer.confirm("Proceed?"):
            raise typer.Exit(1)
    _pull_many(entries, yes=True, json_mode=json_, label=f"pull-all task={task or 'any'}")


def _pull_many(entries: list, *, yes: bool, json_mode: bool, label: str) -> None:
    if not entries:
        _emit(
            {"pulled": [], "skipped": []},
            json_mode=json_mode,
            summary=f"no models matched ({label})",
        )
        return
    pulled, skipped = [], []
    for entry in entries:
        try:
            path = download(entry) if json_mode else _pull_with_progress(entry)
            pulled.append({"id": entry.id, "path": str(path)})
        except DownloadError as exc:
            skipped.append({"id": entry.id, "reason": str(exc)})
            if not json_mode:
                console.print(f"  [yellow]skipped[/yellow] {entry.id}: {exc}")
    _emit(
        {"pulled": pulled, "skipped": skipped},
        json_mode=json_mode,
        summary=f"pulled {len(pulled)}, skipped {len(skipped)}",
    )


# ---------- cache ----------


@cache_app.command("path", help="Print the cache directory path.")
def cache_path_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    p = cache_root()
    _emit({"cache_dir": str(p)}, json_mode=json_, summary=str(p))


@cache_app.command("list", help="List cached models.")
def cache_list(json_: bool = typer.Option(False, "--json")) -> None:
    items = cache_listing()
    if json_:
        _emit(items, json_mode=True)
        return
    table = Table(title="Cached models")
    table.add_column("id")
    table.add_column("size MiB")
    table.add_column("path")
    for item in items:
        table.add_row(item["model_id"], f"{item['size_bytes'] / (1024 * 1024):.1f}", item["path"])
    console.print(table)


@cache_app.command("clean", help="Delete cached files for a model (or all).")
def cache_clean_cmd(
    model_id: str | None = typer.Argument(None),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if not yes:
        what = model_id or "ALL cached models"
        if not typer.confirm(f"Delete {what}?"):
            raise typer.Exit(1)
    freed = cache_clean(model_id)
    _emit({"bytes_freed": freed}, json_mode=json_, summary=f"freed {freed / (1024 * 1024):.1f} MiB")


@cache_app.command("verify", help="Verify cached models (SHA-256 where known).")
def cache_verify_cmd(
    model_id: str | None = typer.Argument(None),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    report = cache_verify(model_id)
    if json_:
        _emit(report, json_mode=True)
        return
    if not report:
        console.print("nothing cached to verify")
        return
    table = Table(title="Cache verification")
    table.add_column("id")
    table.add_column("status")
    table.add_column("path")
    table.add_column("reason")
    for r in report:
        status = "[green]ok[/green]" if r["ok"] else "[red]bad[/red]"
        table.add_row(r["model_id"], status, r.get("path") or "-", r["reason"])
    console.print(table)


@cache_app.command("repair", help="Re-scan a cached model directory and rebuild its manifest.")
def cache_repair_cmd(model_id: str, json_: bool = typer.Option(False, "--json")) -> None:
    ok = cache_repair(model_id)
    _emit(
        {"repaired": ok, "model_id": model_id},
        json_mode=json_,
        summary="repaired" if ok else "could not repair",
    )


# ---------- predict / benchmark / export ----------


@app.command(help="Run inference from the command line.")
def predict(
    model_id: str,
    input_path: Path,
    save: Path | None = typer.Option(
        None, "--save", help="Where to write the annotated image or JSON."
    ),
    prompt: str | None = typer.Option(
        None, "--prompt", help="Comma-separated prompts for open-vocab models."
    ),
    auto_pull: bool = typer.Option(False, "--auto-pull", help="Download weights if missing."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if not input_path.exists():
        _die(f"file not found: {input_path}", json_mode=json_, code="FILE_NOT_FOUND")
        return
    prompts = [p.strip() for p in prompt.split(",")] if prompt else None
    try:
        model = VisionModel(model_id, auto_pull=auto_pull)
        result = model.predict(input_path, prompts=prompts)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return
    except (DownloadError, ManualDownloadRequired) as exc:
        _die(
            str(exc),
            json_mode=json_,
            code="DOWNLOAD_FAILED",
            hint=f"see `visionservex info {model_id}`",
        )
        return
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="PREDICT_FAILED")
        return

    payload = result.to_dict()
    if save:
        result.save(save)
        payload["saved_to"] = str(save)
    if json_:
        _emit(payload, json_mode=True)
        return

    console.print(f"[bold]{result.summary()}[/bold]")
    console.print(
        f"  device: {result.device}  precision: {result.precision}  backend: {result.backend}"
    )
    if result.model_loaded_from:
        console.print(f"  model loaded from: {result.model_loaded_from}")
    if result.cache_path:
        console.print(f"  cache path: {result.cache_path}")
    if save:
        console.print(f"  saved to: {save}")
    for w in result.warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")


@app.command(help="Benchmark model inference latency.")
def benchmark(
    model_id: str,
    input_path: Path,
    n: int = typer.Option(
        10, "--runs", "--n", min=1, max=1000, help="Number of warmup-excluded runs."
    ),
    warmup: int = typer.Option(
        2, "--warmup", min=0, max=20, help="Warmup runs before measurement."
    ),
    device: str | None = typer.Option(
        None, "--device", help="Override device (cpu|cuda|mps|auto)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Measure cold-load time, warm inference latency, and throughput.

    All CPU smoke tests; GPU is preferred when available. Run `visionservex
    doctor` first to verify device health.
    """
    if not input_path.exists():
        _die(f"file not found: {input_path}", json_mode=json_, code="FILE_NOT_FOUND")
        return
    import time

    try:
        t_cold0 = time.perf_counter()
        model = VisionModel(model_id, **({"device": device} if device else {}))
        model._ensure_loaded()
        cold_ms = (time.perf_counter() - t_cold0) * 1000

        for _ in range(warmup):
            model.predict(input_path)

        stats = model.benchmark(input_path, n=n)
        stats["cold_load_ms"] = round(cold_ms, 1)
        stats["warmup_runs"] = warmup
        stats["device"] = model.device
        stats["precision"] = model.precision
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="BENCHMARK_FAILED")
        return

    if json_:
        _emit(stats, json_mode=True)
        return

    console.print(f"[bold]Benchmark:[/bold] {model_id}")
    console.print(f"  Device:      {stats['device']}  Precision: {stats['precision']}")
    console.print(f"  Cold load:   {stats['cold_load_ms']:.0f} ms")
    console.print(f"  Warm p50:    {stats['p50_ms']:.1f} ms")
    console.print(f"  Warm p90:    {stats['p90_ms']:.1f} ms")
    console.print(f"  Warm p99:    {stats['p99_ms']:.1f} ms")
    console.print(f"  Throughput:  ~{1000 / stats['p50_ms']:.1f} req/s (single-thread estimate)")
    console.print("[dim]GPU preferred when healthy; run `visionservex doctor` to verify.[/dim]")


@app.command(help="Export a model to ONNX or another format (engine-dependent).")
def export(
    model_id: str,
    format: str = typer.Option("onnx", "--format"),
    out: Path = typer.Option(..., "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        model = VisionModel(model_id)
        path = model.export(format=format, output_path=out)
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="EXPORT_FAILED")
        return
    _emit({"path": str(path)}, json_mode=json_, summary=f"exported to {path}")


# ---------- examples ----------


@app.command(help="List built-in beginner examples.")
def examples(json_: bool = typer.Option(False, "--json")) -> None:
    items = _list_examples()
    if json_:
        _emit(items, json_mode=True)
        return
    table = Table(title="Beginner examples (in `examples/beginner/`)")
    table.add_column("Name")
    table.add_column("File")
    table.add_column("Description")
    for it in items:
        table.add_row(it["name"], it["file"], it["description"])
    console.print(table)
    console.print("\nRun one with: [cyan]visionservex run-example <name>[/cyan]")


def _list_examples() -> list[dict]:
    return [
        {
            "name": "check-device",
            "file": "examples/beginner/01_check_device.py",
            "description": "Print devices, recommended model, and next command.",
        },
        {
            "name": "list-models",
            "file": "examples/beginner/02_list_models.py",
            "description": "List built-in models with status and license.",
        },
        {
            "name": "download",
            "file": "examples/beginner/03_download_first_model.py",
            "description": "Download the recommended detection model.",
        },
        {
            "name": "detect",
            "file": "examples/beginner/04_detect_image.py",
            "description": "Run detection on a sample image and save the annotated output.",
        },
        {
            "name": "segment",
            "file": "examples/beginner/05_segment_image.py",
            "description": "Run segmentation on a sample image.",
        },
        {
            "name": "classify",
            "file": "examples/beginner/06_classify_image.py",
            "description": "Run classification on a sample image.",
        },
        {
            "name": "open-vocab",
            "file": "examples/beginner/07_open_vocab_detect.py",
            "description": "Run open-vocabulary detection with text prompts.",
        },
        {
            "name": "api",
            "file": "examples/beginner/08_start_api.py",
            "description": "Start the local HTTP API server.",
        },
    ]


@example_app.command("check-device")
def ex_check_device() -> None:
    doctor(json_=False)


@example_app.command("detect")
def ex_detect() -> None:
    img = _ensure_sample_image()
    out = Path("outputs/detect_demo.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    predict("mock-detect", img, save=out, prompt=None, auto_pull=False, json_=False)


@example_app.command("segment")
def ex_segment() -> None:
    img = _ensure_sample_image()
    out = Path("outputs/segment_demo.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    predict("mock-segment", img, save=out, prompt=None, auto_pull=False, json_=False)


@example_app.command("classify")
def ex_classify() -> None:
    img = _ensure_sample_image()
    predict("mock-classify", img, save=None, prompt=None, auto_pull=False, json_=False)


@example_app.command("open-vocab")
def ex_open_vocab() -> None:
    img = _ensure_sample_image()
    out = Path("outputs/open_vocab_demo.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    predict("mock-open-vocab", img, save=out, prompt="cat,dog,car", auto_pull=False, json_=False)


@example_app.command("api")
def ex_api() -> None:
    console.print("Starting local API at http://127.0.0.1:8080 — press Ctrl+C to stop.")
    serve(host=None, port=None, public=False, reload=False)


def _ensure_sample_image() -> Path:
    """Return a path to a sample image, generating one if needed."""
    repo_img = Path("examples/images/simple_shapes.jpg")
    if repo_img.exists():
        return repo_img
    settings = get_settings()
    cache_img = Path(settings.cache.cache_dir) / "samples" / "simple_shapes.jpg"
    if cache_img.exists():
        return cache_img
    cache_img.parent.mkdir(parents=True, exist_ok=True)
    _make_sample(cache_img)
    return cache_img


def _make_sample(path: Path) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 80, 260, 280], outline=(20, 60, 120), width=4, fill=(80, 130, 200))
    draw.ellipse([320, 100, 540, 320], outline=(120, 20, 60), width=4, fill=(220, 100, 120))
    draw.polygon([(150, 380), (250, 380), (200, 460)], outline=(40, 100, 40), fill=(120, 200, 120))
    img.save(path, "JPEG", quality=90)


# ---------- serve ----------


@app.command(help="Start the local HTTP server.")
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    public: bool = typer.Option(
        False, "--public", help="Set public_mode=true (does not change bind address)."
    ),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    overrides: dict = {}
    if host:
        overrides["server"] = {"host": host}
    if port:
        overrides.setdefault("server", {})["port"] = port
    if public:
        overrides.setdefault("server", {})["public_mode"] = True
    if overrides:
        reload_settings(**overrides)

    settings = get_settings()
    try:
        import uvicorn  # type: ignore
    except ImportError:
        _die(
            "uvicorn is not installed. Install the server extra: pip install 'visionservex[server]'",
            json_mode=False,
            code="MISSING_DEP",
        )
        return

    for w in settings.public_safety_warnings():
        console.print(f"[yellow]warning:[/yellow] {w}")
    if settings.server.host == "0.0.0.0":
        console.print(
            "[yellow]warning:[/yellow] binding to 0.0.0.0; only do this behind a trusted network or proxy."
        )
    console.print(f"Starting VisionServeX on http://{settings.server.host}:{settings.server.port}")
    uvicorn.run(
        "visionservex.server.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        reload=reload,
    )


# ---------- config ----------


@config_app.command("show", help="Print the effective configuration.")
def config_show(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.utils.logging import log_safe_dict

    settings = get_settings().model_dump()
    safe = log_safe_dict(settings)
    _emit(safe, json_mode=json_)


@config_app.command("set", help="Set a configuration value (writes to .env in current directory).")
def config_set(
    key: str = typer.Argument(..., help="Env-style key such as server__host"),
    value: str = typer.Argument(..., help="String value"),
) -> None:
    env_key = f"VISIONSERVEX_{key.upper()}"
    env_path = Path(".env")
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    found = False
    for i, line in enumerate(lines):
        if line.startswith(env_key + "="):
            lines[i] = f"{env_key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{env_key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"wrote {env_key}={value} to {env_path}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()


# ---------- top-level convenience aliases ----------


@app.command(
    "benchmark-matrix", help="Benchmark models across devices (see also: benchmark sub-commands)."
)
def benchmark_matrix_alias(
    models: str = typer.Option("mock-detect", "--models"),
    devices: str = typer.Option("cpu", "--devices"),
    runs: int = typer.Option(5, "--runs"),
    warmup: int = typer.Option(2, "--warmup"),
    input_path: Path = typer.Option(Path("examples/images/street.jpg"), "--input"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Shortcut: runs `visionservex benchmark benchmark-matrix`."""
    from visionservex.cli.benchmark_commands import benchmark_matrix

    benchmark_matrix(
        models=models,
        devices=devices,
        runs=runs,
        warmup=warmup,
        input_path=input_path,
        out=out,
        json_=json_,
    )


@app.command("parallel-test", help="Test concurrent inference throughput.")
def parallel_test_alias(
    model_id: str,
    input_path: Path,
    concurrency: int = typer.Option(2, "--concurrency"),
    runs: int = typer.Option(5, "--runs"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Shortcut: runs `visionservex benchmark parallel-test`."""
    from visionservex.cli.benchmark_commands import parallel_test

    parallel_test(
        model_id=model_id,
        input_path=input_path,
        concurrency=concurrency,
        runs=runs,
        device=device,
        json_=json_,
    )


@app.command("downloads-audit", help="Audit download metadata for all registry models.")
def downloads_audit_alias(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Shortcut: runs `visionservex downloads audit`."""
    from visionservex.cli.downloads_commands import audit

    audit(verbose=verbose, json_=json_)


@app.command("mps", help="Apple MPS smoke test.")
def mps_smoke_test(
    models: str = typer.Option("mock-detect", "--models"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Smoke-test models on Apple MPS (macOS only)."""
    from visionservex.cli.gpu_commands import smoke_test

    smoke_test(models=models, device="mps", json_=json_)


@app.command(
    "pull-suite",
    help="Download a curated set of models (beginner|gpu-demo|server-demo|detection|segmentation|classification).",
)
def pull_suite_alias(
    suite_name: str = typer.Argument(..., help="Suite name."),
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Download all models in a named suite. Shortcut for `visionservex suite pull`."""
    from visionservex.cli.suite_commands import suite_pull

    suite_pull(suite_name=suite_name, yes=yes, json_=json_)

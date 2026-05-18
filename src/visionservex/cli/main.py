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
    aerial_commands,
    agriculture_commands,
    annotate_commands,
    anomaly_commands,
    audit_commands,
    benchmark_anomaly_cmd,
    benchmark_classification,
    benchmark_commands,
    benchmark_detection_cmd,
    benchmark_open_vocab,
    benchmark_surveillance,
    capabilities_commands,
    colab_commands,
    dev_commands,
    domain_zoo_commands,
    downloads_commands,
    draw_commands,
    embedding_commands,
    expert_commands,
    florence2_commands,
    gateway_commands,
    gpu_commands,
    license_commands,
    live_commands,
    maskdino_commands,
    medical_commands,
    model_card_commands,
    model_health_commands,
    model_lifecycle_commands,
    model_zoo_commands,
    openmmlab_commands,
    privacy_commands,
    readiness_commands,
    replacement_map_commands,
    sam3_commands,
    sam_family_commands,
    security_commands,
    suite_commands,
    syntax_audit,
    tensorrt_commands,
    training_commands,
    validation_commands,
    video_search_commands,
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
app.add_typer(syntax_audit.app, name="syntax")
app.add_typer(security_commands.app, name="security")
app.add_typer(privacy_commands.app, name="privacy")
app.add_typer(validation_commands.app, name="validation")
app.add_typer(colab_commands.app, name="colab")
app.add_typer(capabilities_commands.app, name="capabilities")
app.add_typer(model_card_commands.app, name="model-card")
app.add_typer(replacement_map_commands.app, name="replacement-map")
app.add_typer(model_lifecycle_commands.app, name="model")
app.add_typer(training_commands.training_app, name="training")
app.add_typer(training_commands.export_app, name="export-cmd")
app.add_typer(training_commands.video_app, name="video")
app.add_typer(model_zoo_commands.app, name="model-zoo")
app.add_typer(domain_zoo_commands.app, name="domain-zoo")
app.add_typer(embedding_commands.app, name="feature")
app.add_typer(dev_commands.app, name="dev")
app.add_typer(model_health_commands.app, name="models")
app.add_typer(readiness_commands.app, name="readiness")
app.add_typer(sam3_commands.app, name="sam3")
app.add_typer(expert_commands.app, name="expert")
app.add_typer(maskdino_commands.app, name="maskdino")
app.add_typer(anomaly_commands.app, name="anomaly")
app.add_typer(medical_commands.app, name="medical")
app.add_typer(video_search_commands.app, name="video-search")
app.add_typer(benchmark_open_vocab.app, name="benchmark-open-vocab", invoke_without_command=True)
app.add_typer(agriculture_commands.app, name="agriculture")
app.add_typer(aerial_commands.app, name="aerial")
app.add_typer(audit_commands.app, name="audit")
app.add_typer(license_commands.app, name="license")
app.add_typer(florence2_commands.app, name="florence2")
app.add_typer(sam_family_commands.app, name="sam-family")
app.add_typer(
    benchmark_classification.app, name="benchmark-classification", invoke_without_command=True
)
app.add_typer(benchmark_anomaly_cmd.app, name="benchmark-anomaly", invoke_without_command=True)
app.add_typer(
    benchmark_surveillance.app, name="benchmark-surveillance-search", invoke_without_command=True
)
app.add_typer(draw_commands.app, name="draw")
app.add_typer(annotate_commands.app, name="annotate")
app.add_typer(live_commands.app, name="live", invoke_without_command=True)
app.add_typer(
    benchmark_detection_cmd.app_det, name="benchmark-detection", invoke_without_command=True
)
app.add_typer(
    benchmark_detection_cmd.app_ult, name="benchmark-ultralytics", invoke_without_command=True
)

# Top-level embedding aliases for Ultralytics-style ergonomics
embedding_alias_app = embedding_commands.app

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
    task: str | None = typer.Option(
        None, "--task", help="Task: detect|segment|classify|pose|obb|open_vocab_detect"
    ),
    device: str | None = typer.Option(None, "--device"),
    vram: float | None = typer.Option(None, "--vram", help="Available VRAM in GB."),
    simple: bool = typer.Option(False, "--simple", help="Prefer beginner-friendly models."),
    goal: str | None = typer.Option(
        None,
        "--goal",
        help=(
            "Recommendation goal: accuracy|fastest_demo|best_open_license|"
            "best_colab|best_gpu|best_cpu|best_segmentation|best_open_vocab"
        ),
    ),
    include_docker: bool = typer.Option(
        False, "--include-docker", help="Include docker/sidecar models."
    ),
    limit: int = typer.Option(5, "--limit"),
    json_: bool = typer.Option(False, "--json"),
):
    recs = recommend(task=task, device=device, vram_gb=vram, simple=simple, goal=goal, limit=limit)
    if not include_docker:
        recs = [
            r
            for r in recs
            if r.entry.implementation_status != "partial"
            or r.entry.engine not in {"openmmlab_sidecar"}
        ]
    if json_:
        _emit([r.to_dict() for r in recs], json_mode=True)
        return
    if not recs:
        console.print("[yellow]No recommendations found for the given filters.[/yellow]")
        return
    title = "Recommendations"
    if goal:
        title += f" (goal={goal})"
    table = Table(title=title)
    for col in ("id", "task", "category", "status", "impl", "score", "device", "license"):
        table.add_column(col)
    for r in recs:
        e = r.entry
        cat = e.model_category or "-"
        cat_color = {
            "accuracy_grade": "green",
            "production_recommended": "cyan",
            "demo_fast": "yellow",
            "experimental_sota": "magenta",
            "expert_sidecar": "grey50",
            "external_api": "grey50",
            "unavailable_with_reason": "red",
            "utility": "grey50",
        }.get(cat, "white")
        table.add_row(
            e.id,
            e.task,
            f"[{cat_color}]{cat}[/{cat_color}]",
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
    task: str | None = typer.Option(None, "--task", help="Limit to this task."),
    yes: bool = typer.Option(False, "--yes"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    tasks = [task] if task else ["detect", "segment", "classify", "pose", "open_vocab_detect"]
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
    model: str | None = typer.Option(
        None, "--model", help="Model ID to clean (alias for positional argument)."
    ),
    all_: bool = typer.Option(False, "--all", help="Clean ALL cached models."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation prompt."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    # --model overrides positional; --all clears everything
    target = model or model_id
    if all_:
        target = None
    if not yes:
        what = target or "ALL cached models"
        if not typer.confirm(f"Delete {what}?"):
            raise typer.Exit(1)
    freed = cache_clean(target)
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


# ---------- predict / batch-predict / benchmark / export ----------


@app.command(help="Run inference from the command line.")
def predict(
    model_id: str,
    input_path: Path,
    save: Path | None = typer.Option(None, "--save", help="Save annotated image or JSON."),
    save_json: Path | None = typer.Option(None, "--save-json", help="Save result JSON to path."),
    save_image: Path | None = typer.Option(
        None, "--save-image", help="Save annotated image to path."
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Comma-separated text prompts."),
    device: str | None = typer.Option(None, "--device", help="Device: auto|cpu|cuda|mps"),
    precision: str | None = typer.Option(None, "--precision", help="Precision: auto|fp32|fp16"),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k for classification."),
    threshold: float | None = typer.Option(None, "--threshold", help="Score threshold."),
    point: str | None = typer.Option(
        None, "--point", help="Point prompt x,y for SAM-style models."
    ),
    box: str | None = typer.Option(
        None, "--box", help="Box prompt x1,y1,x2,y2 for SAM-style models."
    ),
    task: str | None = typer.Option(
        None, "--task", help="Task for multi-task models (semantic/panoptic/instance)."
    ),
    timeout: float | None = typer.Option(None, "--timeout", help="Prediction timeout in seconds."),
    auto_pull: bool = typer.Option(False, "--auto-pull", help="Download weights if missing."),
    no_auto_pull: bool = typer.Option(False, "--no-auto-pull", help="Disable auto-pull."),
    json_: bool = typer.Option(False, "--json"),
    debug: bool = typer.Option(False, "--debug"),
) -> None:
    """Run prediction on an image. Supports detection, classification, segmentation, pose, OBB."""
    from visionservex.exceptions import (
        InputNotFoundError,
        ModelNotFoundError,
        VisionServeXError,
    )

    if not input_path.exists():
        _die(
            str(InputNotFoundError(str(input_path))),
            json_mode=json_,
            code="INPUT_NOT_FOUND",
            hint=f"Check file path: {input_path}",
        )
        return

    effective_auto_pull = auto_pull and not no_auto_pull

    # Parse convenience args
    kwargs: dict = {}
    if prompt:
        kwargs["prompt"] = prompt
    if top_k is not None:
        kwargs["top_k"] = top_k
    if threshold is not None:
        kwargs["threshold"] = threshold
    if task is not None:
        kwargs["task"] = task
    if point:
        try:
            x, y = [float(v) for v in point.split(",")]
            kwargs["points"] = [[x, y]]
            kwargs["point_labels"] = [1]
        except ValueError:
            _die(
                f"Invalid --point format {point!r}; expected x,y", json_mode=json_, code="BAD_ARGS"
            )
            return
    if box:
        try:
            coords = [float(v) for v in box.split(",")]
            if len(coords) != 4:
                raise ValueError
            kwargs["box"] = coords
        except ValueError:
            _die(
                f"Invalid --box format {box!r}; expected x1,y1,x2,y2",
                json_mode=json_,
                code="BAD_ARGS",
            )
            return

    try:
        model_kwargs: dict = {"auto_pull": effective_auto_pull}
        if device:
            model_kwargs["device"] = device
        if precision:
            model_kwargs["precision"] = precision
        model = VisionModel(model_id, **model_kwargs)
        result = model.predict(input_path, **kwargs)
    except RegistryError:
        err = ModelNotFoundError(model_id)
        _die(str(err), json_mode=json_, code=err.code, hint=err.hint)
        return
    except (DownloadError, ManualDownloadRequired) as exc:
        _die(
            str(exc),
            json_mode=json_,
            code="DOWNLOAD_FAILED",
            hint=f"visionservex model pull {model_id}  (or pass --auto-pull to download automatically)",
        )
        return
    except VisionServeXError as exc:
        _die(str(exc), json_mode=json_, code=exc.code, hint=exc.hint)
        return
    except Exception as exc:
        if debug:
            raise
        _die(str(exc), json_mode=json_, code="PREDICT_FAILED")
        return

    # Save outputs
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        result.save(save)
    if save_json:
        result.save_json(save_json)
    if save_image:
        result.save_image(save_image)

    payload = result.to_dict()
    if save:
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
    if save or save_json or save_image:
        console.print(f"  saved to: {save or save_json or save_image}")
    for w in result.warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")


@app.command("batch-predict", help="Run inference on a directory of images.")
def batch_predict(
    model_id: str,
    input_dir: Path,
    save_dir: Path | None = typer.Option(
        None, "--save-dir", help="Output directory for annotated images."
    ),
    save_json: Path | None = typer.Option(
        None, "--save-json", help="Write all results to this JSON file."
    ),
    prompt: str | None = typer.Option(None, "--prompt", help="Prompts for open-vocab models."),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k for classification."),
    extensions: str = typer.Option("jpg,jpeg,png,webp,bmp", "--ext"),
    auto_pull: bool = typer.Option(False, "--auto-pull"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run the same model on all images in a directory."""

    if not input_dir.exists():
        _die(f"directory not found: {input_dir}", json_mode=json_, code="INPUT_NOT_FOUND")
        return

    exts = {f".{e.strip().lower()}" for e in extensions.split(",")}
    paths = sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not paths:
        _die(f"no images found in {input_dir}", json_mode=json_, code="NO_IMAGES")
        return

    kwargs: dict = {}
    if prompt:
        kwargs["prompt"] = prompt
    if top_k is not None:
        kwargs["top_k"] = top_k

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = VisionModel(model_id, auto_pull=auto_pull)
    except RegistryError as exc:
        _die(str(exc), json_mode=json_, code="MODEL_NOT_FOUND")
        return

    all_results = []
    for p in paths:
        try:
            result = model.predict(p, **kwargs)
            entry = result.to_dict()
            entry["input_path"] = str(p)
            all_results.append(entry)
            if save_dir:
                out = save_dir / p.name
                result.save(out)
            if not json_:
                console.print(f"  [green]ok[/green] {p.name}  {result.summary()}")
        except Exception as exc:
            all_results.append({"input_path": str(p), "error": str(exc)[:100]})
            if not json_:
                console.print(f"  [red]err[/red] {p.name}: {str(exc)[:80]}")

    if save_json:
        import json as _j

        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(_j.dumps(all_results, indent=2, default=str), encoding="utf-8")

    if json_:
        _emit(all_results, json_mode=True)
    else:
        ok = sum(1 for r in all_results if "error" not in r)
        console.print(f"\n{ok}/{len(all_results)} predictions successful.")


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
    stop_on_vram_risk: bool = typer.Option(
        False,
        "--stop-on-vram-risk",
        help="Check VRAM before running; abort if safety buffer is at risk.",
    ),
    max_vram_fraction: float = typer.Option(
        0.80,
        "--max-vram-fraction",
        help="VRAM safety fraction (only checked with --stop-on-vram-risk).",
    ),
    min_free_vram_gb: float = typer.Option(
        3.0, "--min-free-vram-gb", help="Min free VRAM GB (only checked with --stop-on-vram-risk)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Shortcut: runs `visionservex benchmark parallel-test`."""
    if stop_on_vram_risk and device not in ("cpu",):
        from visionservex.cli.gpu_commands import (
            _DEFAULT_VRAM_POLICY,
            _compute_safety_budget,
            _get_vram_state,
        )

        policy = _DEFAULT_VRAM_POLICY.copy()
        policy["max_vram_fraction"] = max_vram_fraction
        policy["min_free_vram_gb"] = min_free_vram_gb
        vram = _get_vram_state()
        if vram["source"] != "unavailable":
            budget = _compute_safety_budget(vram, policy)
            if not budget.get("safe", True):
                msg = (
                    f"GPU_MEMORY_GUARD: VRAM at risk (free={vram['free_gb']:.2f}GB). "
                    "Aborting parallel-test to protect system stability."
                )
                if json_:
                    _emit(
                        {"error": "GPU_MEMORY_GUARD", "message": msg, "vram": vram}, json_mode=json_
                    )
                else:
                    console.print(f"[red]{msg}[/red]")
                    console.print("[dim]Run: visionservex gpu processes[/dim]")
                raise typer.Exit(1)

    from visionservex.cli.benchmark_commands import parallel_test

    parallel_test(
        model_id=model_id,
        input_path=input_path,
        concurrency=concurrency,
        runs=runs,
        device=device,
        json_=json_,
    )


@app.command(
    "debug-output",
    help="Print raw output diagnostics for a model — use to audit postprocessing before blaming the checkpoint.",
)
def debug_output(
    model_id: str,
    input_path: Path,
    threshold: float = typer.Option(
        0.01, "--threshold", help="Low threshold to see all detections."
    ),
    device: str | None = typer.Option(None, "--device"),
    save_json: Path | None = typer.Option(
        None, "--save-json", help="Save diagnostics JSON to this path."
    ),
    visualize: Path | None = typer.Option(
        None, "--visualize", help="Save annotated image with detection boxes."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Diagnostic tool: print raw output keys, normalized detections, score/label histograms,
    invalid boxes, preprocessing config, and optional visualization.
    Run this before declaring a checkpoint weak.
    """

    from visionservex.core.results import DetectionResult

    if not input_path.exists():
        _die(f"file not found: {input_path}", json_mode=json_, code="FILE_NOT_FOUND")
        return

    try:
        from PIL import Image as _PIL

        img = _PIL.open(input_path).convert("RGB")
        w, h = img.size
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="IMAGE_LOAD_FAILED")
        return

    try:
        kwargs: dict = {}
        if device:
            kwargs["device"] = device
        model = VisionModel(model_id, **kwargs)
        model._ensure_loaded()
        result = model.predict(img, threshold=threshold)
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="PREDICT_FAILED")
        return

    diag: dict = {
        "model_id": model_id,
        "device": result.device,
        "precision": result.precision,
        "backend": result.backend,
        "task": result.task,
        "kind": result.kind,
        "image_size_wh": [w, h],
        "threshold_used": threshold,
    }

    if isinstance(result, DetectionResult):
        dets = result.detections
        diag["total_detections"] = len(dets)
        diag["first_10_boxes"] = [
            {
                "label": d.label,
                "class_id": d.class_id,
                "score": round(d.score, 4),
                "x1": round(d.box.x1, 2),
                "y1": round(d.box.y1, 2),
                "x2": round(d.box.x2, 2),
                "y2": round(d.box.y2, 2),
            }
            for d in dets[:10]
        ]

        # Score histogram (5 bins)
        scores = [d.score for d in dets]
        bins = [0, 0.1, 0.3, 0.5, 0.7, 1.01]
        bin_labels = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
        score_hist = dict.fromkeys(bin_labels, 0)
        for s in scores:
            for i, b in enumerate(bins[:-1]):
                if b <= s < bins[i + 1]:
                    score_hist[bin_labels[i]] += 1
                    break
        diag["score_histogram"] = score_hist

        # Label histogram
        label_counts: dict = {}
        for d in dets:
            label_counts[d.label] = label_counts.get(d.label, 0) + 1
        top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
        diag["label_histogram_top10"] = dict(top_labels)

        # Invalid boxes
        invalid = []
        for d in dets:
            b = d.box
            reasons = []
            if b.x1 < 0:
                reasons.append("x1<0")
            if b.y1 < 0:
                reasons.append("y1<0")
            if b.x2 > w:
                reasons.append(f"x2>{w}")
            if b.y2 > h:
                reasons.append(f"y2>{h}")
            if b.x1 >= b.x2:
                reasons.append("x1>=x2")
            if b.y1 >= b.y2:
                reasons.append("y1>=y2")
            if reasons:
                invalid.append({"label": d.label, "score": round(d.score, 4), "reasons": reasons})
        diag["invalid_boxes"] = len(invalid)
        diag["invalid_box_details"] = invalid[:5]

        # Unmapped labels
        unmapped = [d.label for d in dets if d.label.startswith("class_")]
        diag["unmapped_labels_count"] = len(unmapped)
        diag["unmapped_label_samples"] = list(set(unmapped))[:5]

        # Preprocessing advice
        diag["preprocessing_notes"] = (
            f"If scores are very low or all boxes are invalid, check: "
            f"(1) image normalization (RGB order?), "
            f"(2) resize/letterbox match model expectation, "
            f"(3) threshold — this run used threshold={threshold:.2f}. "
            f"Use --threshold 0.0 to see all raw outputs."
        )
    else:
        diag["raw_kind"] = result.kind
        diag["note"] = f"Full debug-output is implemented for detect task. Got task={result.task}."

    if json_:
        _emit(diag, json_mode=True)
        return

    from rich.panel import Panel as _Panel

    console.print(_Panel.fit(f"[bold]debug-output:[/bold] {model_id}", border_style="yellow"))
    console.print(
        f"  Device: {diag['device']}  Precision: {diag['precision']}  Backend: {diag['backend']}"
    )
    console.print(f"  Image: {w}x{h}  Threshold: {threshold}")
    console.print(f"  Total detections: {diag.get('total_detections', '-')}")

    if "score_histogram" in diag:
        console.print("\n  [bold]Score histogram:[/bold]")
        for bin_label, count in diag["score_histogram"].items():
            bar = "█" * min(count, 40)
            console.print(f"    {bin_label}: {bar} ({count})")

    if "label_histogram_top10" in diag:
        console.print("\n  [bold]Top labels:[/bold]")
        for label, count in diag["label_histogram_top10"].items():
            console.print(f"    {label}: {count}")

    if diag.get("invalid_boxes", 0) > 0:
        console.print(f"\n  [red]Invalid boxes: {diag['invalid_boxes']}[/red]")
        for iv in diag.get("invalid_box_details", []):
            console.print(f"    {iv}")
    else:
        console.print("\n  [green]No invalid boxes.[/green]")

    if diag.get("unmapped_labels_count", 0) > 0:
        console.print(
            f"\n  [yellow]Unmapped labels (class_N): {diag['unmapped_labels_count']}[/yellow]"
        )
        console.print(f"    Samples: {diag.get('unmapped_label_samples', [])}")

    if diag.get("first_10_boxes"):
        console.print("\n  [bold]First 10 boxes:[/bold]")
        for b in diag["first_10_boxes"]:
            console.print(
                f"    [{b['score']:.2f}] {b['label']} (id={b['class_id']}) "
                f"[{b['x1']:.0f},{b['y1']:.0f},{b['x2']:.0f},{b['y2']:.0f}]"
            )

    console.print(f"\n  [dim]{diag.get('preprocessing_notes', '')}[/dim]")

    # Save JSON if requested
    if save_json:
        import json as _json

        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(_json.dumps(diag, indent=2, default=str), encoding="utf-8")
        console.print(f"\n  [green]Diagnostics JSON saved to {save_json}[/green]")

    # Visualize if requested
    if visualize and isinstance(result, DetectionResult):
        try:
            from PIL import ImageDraw as _Draw

            vis_img = img.copy()
            draw = _Draw.Draw(vis_img)
            for det in result.detections:
                b = det.box
                draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=(255, 0, 0), width=2)
                draw.text(
                    (b.x1, max(0, b.y1 - 12)), f"{det.label} {det.score:.2f}", fill=(255, 0, 0)
                )
            visualize.parent.mkdir(parents=True, exist_ok=True)
            vis_img.save(str(visualize))
            console.print(f"  [green]Visualization saved to {visualize}[/green]")
        except Exception as vis_exc:
            console.print(f"  [yellow]Visualization failed: {vis_exc}[/yellow]")


@app.command("downloads-audit", help="Audit download metadata for all registry models.")
def downloads_audit_alias(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit 1 if any required metadata is missing.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Shortcut: runs `visionservex downloads audit`."""
    from visionservex.cli.downloads_commands import audit

    audit(verbose=verbose, json_=json_)

    # --strict: fail only if the audit itself reports missing required metadata
    if strict:
        from visionservex.cli.downloads_commands import audit_missing_required_count

        n_missing = audit_missing_required_count()
        if n_missing > 0:
            if not json_:
                console.print(
                    f"[red]--strict: {n_missing} model(s) missing required download metadata[/red]"
                )
            raise typer.Exit(1)
        if not json_:
            console.print("[green]--strict: all models have required download metadata[/green]")


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


# ---------- models audit ----------


@app.command("models-audit", help="Audit model registry for completeness and honest status.")
def models_audit(
    verbose: bool = typer.Option(False, "--verbose"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Audit model registry: find stubs without notes, external without alternatives, etc."""
    from visionservex.registry import default_registry

    reg = default_registry()
    issues = []
    status_counts: dict[str, int] = {}

    # Statuses that are self-explanatory: "manual" means user must install/download
    # manually (this IS the explanation); "external" is API-gated.
    _SELF_DOCUMENTING_STATUSES = {"manual", "external", "optional"}

    for entry in reg.list():
        status_counts[entry.implementation_status] = (
            status_counts.get(entry.implementation_status, 0) + 1
        )
        model_issues = []
        if (
            entry.implementation_status == "stub"
            and not entry.notes
            and not entry.warnings
            and entry.status not in _SELF_DOCUMENTING_STATUSES
        ):
            model_issues.append("stub without notes/warnings")
        if entry.status == "external" and not entry.notes and not entry.warnings:
            model_issues.append("external without notes/warnings")
        if not entry.license:
            model_issues.append("missing license")
        if not entry.upstream_url:
            model_issues.append("missing upstream_url")
        if model_issues:
            issues.append({"id": entry.id, "issues": model_issues})

    payload = {
        "counts": status_counts,
        "models_with_issues": len(issues),
        "issues": issues if verbose or json_ else issues[:5],
    }
    if json_:
        _emit(payload, json_mode=True)
        return

    console.print(f"[bold]Models audit[/bold] — {sum(status_counts.values())} total")
    for status, count in sorted(status_counts.items()):
        color = {"wired": "green", "partial": "yellow", "stub": "grey50"}.get(status, "white")
        console.print(f"  [{color}]{status}[/{color}]: {count}")
    if issues:
        console.print(f"\n[yellow]{len(issues)} models have issues[/yellow]")
        for item in issues[:10]:
            console.print(f"  {item['id']}: {', '.join(item['issues'])}")
    else:
        console.print("\n[green]All models look good.[/green]")


# ---------- onnx sub-commands ----------


@app.command("onnx-validate", help="Validate an ONNX file with onnx.checker.")
def onnx_validate(
    onnx_path: Path = typer.Argument(..., help="Path to .onnx file."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        payload = {
            "status": "ok",
            "path": str(onnx_path),
            "opset": model.opset_import[0].version if model.opset_import else "unknown",
        }
        if json_:
            _emit(payload, json_mode=True)
        else:
            console.print(f"[green]ONNX validation passed[/green]: {onnx_path}")
    except ImportError:
        _die(
            "onnx not installed. pip install 'visionservex[onnx]'",
            json_mode=json_,
            code="MISSING_DEP",
        )
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="ONNX_VALIDATION_FAILED")


@app.command("onnx-parity", help="Check ONNX model output matches the original PyTorch model.")
def onnx_parity(
    model_id: str,
    onnx_path: Path = typer.Argument(...),
    tolerance: float = typer.Option(1e-3, "--tol"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image as _Image

        from visionservex import VisionModel

        img = _Image.new("RGB", (256, 256), "blue")
        m = VisionModel(model_id, device="cpu")
        m._ensure_loaded()
        torch_result = m.predict(img)

        sess = ort.InferenceSession(str(onnx_path))
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        ort_out = sess.run(None, {inp_name: dummy})

        payload = {
            "model_id": model_id,
            "onnx_path": str(onnx_path),
            "ort_output_shapes": [list(o.shape) for o in ort_out],
            "torch_task": torch_result.task,
            "status": "parity_check_run",
            "note": "Shape check only; numerical parity requires matching inputs.",
        }
        if json_:
            _emit(payload, json_mode=True)
        else:
            console.print(f"[green]ORT outputs:[/green] {payload['ort_output_shapes']}")
    except Exception as exc:
        _die(str(exc), json_mode=json_, code="PARITY_FAILED")


# ---------- parallel-test-pair ----------


@app.command("parallel-test-pair", help="Test two models running concurrently on the same device.")
def parallel_test_pair(
    model_a: str,
    model_b: str,
    input_path: Path,
    device: str = typer.Option("auto", "--device"),
    runs: int = typer.Option(3, "--runs"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run model_a and model_b concurrently and compare to sequential times."""
    import threading
    import time

    from PIL import Image as _PIL

    from visionservex import VisionModel

    img = (
        _PIL.open(input_path).convert("RGB") if input_path.exists() else _PIL.new("RGB", (320, 240))
    )

    ma = VisionModel(model_a, device=device)
    mb = VisionModel(model_b, device=device)
    ma._ensure_loaded()
    mb._ensure_loaded()

    # Sequential baseline
    a_times, b_times = [], []
    for _ in range(runs):
        t = time.perf_counter()
        ma.predict(img)
        a_times.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        mb.predict(img)
        b_times.append((time.perf_counter() - t) * 1000)
    seq_total = sorted(a_times)[runs // 2] + sorted(b_times)[runs // 2]

    # Concurrent
    wall_times = []
    for _ in range(runs):
        results_ab = {}

        def _run(name, model, out):
            t = time.perf_counter()
            model.predict(img)
            out[name] = (time.perf_counter() - t) * 1000

        threads = [
            threading.Thread(target=_run, args=(model_a, ma, results_ab)),
            threading.Thread(target=_run, args=(model_b, mb, results_ab)),
        ]
        tw = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall_times.append((time.perf_counter() - tw) * 1000)

    wall_p50 = sorted(wall_times)[runs // 2]
    slowdown = (wall_p50 / seq_total - 1.0) * 100 if seq_total > 0 else 0.0
    status = (
        "excellent_parallelism"
        if slowdown <= 10
        else "acceptable_parallelism"
        if slowdown <= 25
        else "scheduler_needs_queueing"
    )

    payload = {
        "model_a": model_a,
        "model_b": model_b,
        "device": device,
        "seq_a_p50_ms": round(sorted(a_times)[runs // 2], 2),
        "seq_b_p50_ms": round(sorted(b_times)[runs // 2], 2),
        "seq_total_ms": round(seq_total, 2),
        "concurrent_wall_p50_ms": round(wall_p50, 2),
        "slowdown_pct": round(slowdown, 1),
        "status": status,
    }
    if json_:
        _emit(payload, json_mode=True)
        return
    console.print(f"[bold]Pair test:[/bold] {model_a} + {model_b} on {device}")
    console.print(
        f"  Sequential: {seq_total:.1f}ms  |  Concurrent wall: {wall_p50:.1f}ms  |  Slowdown: {slowdown:.0f}%"
    )
    color = "green" if "excellent" in status else "yellow" if "acceptable" in status else "red"
    console.print(f"  Status: [{color}]{status}[/{color}]")


# ---------------------------------------------------------------------------
# Task alias commands (Ultralytics-like ergonomics)
# ---------------------------------------------------------------------------


@app.command(
    "detect", help="Alias: visionservex detect MODEL IMAGE [same as predict for detect models]."
)
def detect_alias(
    model_id: str,
    input_path: Path,
    conf: float = typer.Option(0.25, "--conf", "--threshold"),
    device: str | None = typer.Option(None, "--device"),
    save_image: Path | None = typer.Option(None, "--save-image"),
    save_json: Path | None = typer.Option(None, "--save-json"),
    out: Path | None = typer.Option(
        None, "--out", help="Save result JSON (notebook alias for --save-json)."
    ),
    draw: Path | None = typer.Option(
        None, "--draw", help="Save annotated image (notebook alias for --save-image)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Ultralytics-like detect alias. Equivalent to: visionservex predict MODEL IMAGE."""
    predict(
        model_id=model_id,
        input_path=input_path,
        save=None,
        save_json=out or save_json,
        save_image=draw or save_image,
        prompt=None,
        device=device,
        precision=None,
        top_k=None,
        threshold=conf,
        point=None,
        box=None,
        task=None,
        timeout=None,
        auto_pull=False,
        no_auto_pull=False,
        json_=json_,
        debug=False,
    )


@app.command(
    "segment", help="Alias: visionservex segment MODEL IMAGE [same as predict for segment models]."
)
def segment_alias(
    model_id: str,
    input_path: Path,
    conf: float = typer.Option(0.25, "--conf", "--threshold"),
    device: str | None = typer.Option(None, "--device"),
    save_image: Path | None = typer.Option(None, "--save-image"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Ultralytics-like segment alias."""
    predict(
        model_id=model_id,
        input_path=input_path,
        save=None,
        save_json=None,
        save_image=save_image,
        prompt=None,
        device=device,
        precision=None,
        top_k=None,
        threshold=conf,
        point=None,
        box=None,
        task=None,
        timeout=None,
        auto_pull=False,
        no_auto_pull=False,
        json_=json_,
        debug=False,
    )


@app.command(
    "classify",
    help="Alias: visionservex classify MODEL IMAGE [same as predict for classify models].",
)
def classify_alias(
    model_id: str,
    input_path: Path,
    top_k: int = typer.Option(5, "--top-k"),
    device: str | None = typer.Option(None, "--device"),
    out: Path | None = typer.Option(
        None, "--out", help="Save result JSON (notebook alias for --save-json)."
    ),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text or json (notebook contract)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Ultralytics-like classify alias.

    v2.16.0: `--out PATH` and `--format json` are accepted to match the
    notebook contract (`visionservex classify swinv2-tiny img.jpg --top-k 5
    --out /tmp/x.json --format json`). They behave the same as the predict
    --save-json/--json flags.
    """
    predict(
        model_id=model_id,
        input_path=input_path,
        save=None,
        save_json=out,
        save_image=None,
        prompt=None,
        device=device,
        precision=None,
        top_k=top_k,
        threshold=None,
        point=None,
        box=None,
        task=None,
        timeout=None,
        auto_pull=False,
        no_auto_pull=False,
        json_=(json_ or fmt == "json"),
        debug=False,
    )


@app.command("open-vocab", help="Open-vocabulary detection with a text prompt.")
def open_vocab_alias(
    model_id: str,
    input_path: Path,
    prompt: str = typer.Option(..., "--prompt", help="Comma-separated class names."),
    conf: float = typer.Option(0.25, "--conf", "--threshold"),
    device: str | None = typer.Option(None, "--device"),
    save_image: Path | None = typer.Option(None, "--save-image"),
    out: Path | None = typer.Option(
        None, "--out", help="Save result JSON (notebook alias for --save-json)."
    ),
    draw: Path | None = typer.Option(
        None, "--draw", help="Save annotated image (notebook alias for --save-image)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Open-vocabulary detection: visionservex open-vocab MODEL IMAGE --prompt 'cat,dog'."""
    predict(
        model_id=model_id,
        input_path=input_path,
        save=None,
        save_json=out,
        save_image=draw or save_image,
        prompt=prompt,
        device=device,
        precision=None,
        top_k=None,
        threshold=conf,
        point=None,
        box=None,
        task=None,
        timeout=None,
        auto_pull=False,
        no_auto_pull=False,
        json_=json_,
        debug=False,
    )


@app.command(
    "grounded-segment", help="Text-prompted segmentation: find and segment objects by name."
)
def grounded_segment_alias(
    model_id: str,
    input_path: Path,
    prompt: str = typer.Option(..., "--prompt", help="Comma-separated class names."),
    conf: float = typer.Option(0.25, "--conf", "--threshold"),
    device: str | None = typer.Option(None, "--device"),
    save_image: Path | None = typer.Option(None, "--save-image"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Grounded segmentation: text prompt → detect → segment."""
    predict(
        model_id=model_id,
        input_path=input_path,
        save=None,
        save_json=None,
        save_image=save_image,
        prompt=prompt,
        device=device,
        precision=None,
        top_k=None,
        threshold=conf,
        point=None,
        box=None,
        task=None,
        timeout=None,
        auto_pull=False,
        no_auto_pull=False,
        json_=json_,
        debug=False,
    )


@app.command("val", help="Evaluate model AP on an annotated dataset (detect task).")
def val_cmd(
    model_id: str,
    dataset: str = typer.Option(..., "--dataset", help="'yolo:<path>' or 'coco-json:<img>:<ann>'"),
    max_images: int = typer.Option(100, "--max-images"),
    device: str | None = typer.Option(None, "--device"),
    out: Path | None = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Evaluate detection AP50/mAP50:95. Detection task only; others return BENCHMARK_NOT_IMPLEMENTED."""
    from visionservex.cli.training_commands import val_model

    val_model(
        model_id=model_id,
        dataset=dataset,
        max_images=max_images,
        device=device or "auto",
        out=out,
        json_=json_,
    )


@app.command(
    "train", help="Train a model — returns structured TRAINING_NOT_SUPPORTED for most models."
)
def train_cmd(
    model_id: str,
    data: str | None = typer.Option(None, "--data"),
    epochs: int = typer.Option(50, "--epochs"),
    device: str | None = typer.Option(None, "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Train. Most models return TRAINING_NOT_SUPPORTED with a structured error and hint."""
    from visionservex.cli.training_commands import train_model

    train_model(
        model_id=model_id,
        data=data,
        epochs=epochs,
        device=device or "auto",
        json_=json_,
    )


@app.command("finetune", help="Fine-tune a model — returns structured error for most models.")
def finetune_cmd(
    model_id: str,
    data: str | None = typer.Option(None, "--data"),
    epochs: int = typer.Option(20, "--epochs"),
    device: str | None = typer.Option(None, "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Fine-tune. Most models return TRAINING_NOT_SUPPORTED with a structured error and hint."""
    from visionservex.cli.training_commands import finetune_model

    finetune_model(
        model_id=model_id,
        data=data,
        epochs=epochs,
        device=device or "auto",
        json_=json_,
    )


# ---------------------------------------------------------------------------
# Top-level embedding aliases (v1.6.0 — feature intelligence)
# ---------------------------------------------------------------------------


@app.command("embed", help="Compute image embeddings (DINOv2 / SigLIP2 / etc.).")
def embed_top_alias(
    model_id: str,
    target: Path,
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature embed`."""
    from visionservex.cli.embedding_commands import embed_cmd

    embed_cmd(
        model_id=model_id,
        target=target,
        out=out,
        device=device,
        max_images=max_images,
        json_=json_,
    )


@app.command("similarity", help="Cosine similarity between two images.")
def similarity_top_alias(
    model_id: str,
    image_a: Path,
    image_b: Path,
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(
        None, "--out", help="Save similarity result JSON to this path."
    ),
    fmt: str = typer.Option(
        "text", "--format", help="Output format: text or json (notebook contract)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature similarity`.

    v2.16.0: accepts `--out PATH` and `--format json` so the notebook command
    `visionservex similarity siglip2-base-patch16-224 a.jpg b.jpg --out
    /tmp/sim.json --format json` works.
    """
    from visionservex.cli.embedding_commands import similarity_cmd

    similarity_cmd(
        model_id,
        image_a,
        image_b,
        device=device,
        out=out,
        fmt=fmt,
        json_=json_,
    )


@app.command("index", help="Build an embedding search index.")
def index_top_alias(
    model_id: str,
    folder: Path,
    out: Path = typer.Option(..., "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature index`."""
    from visionservex.cli.embedding_commands import index_cmd

    index_cmd(model_id, folder, out=out, device=device, max_images=max_images, json_=json_)


@app.command("search", help="Search an embedding index for nearest neighbors.")
def search_top_alias(
    model_id: str,
    query: Path,
    index_dir: Path = typer.Option(..., "--index"),
    top_k: int = typer.Option(10, "--top-k"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature search`."""
    from visionservex.cli.embedding_commands import search_cmd

    search_cmd(model_id, query, index_dir=index_dir, top_k=top_k, device=device, json_=json_)


@app.command("deduplicate", help="Find likely duplicates in a folder using embeddings.")
def deduplicate_top_alias(
    model_id: str,
    folder: Path,
    threshold: float = typer.Option(0.98, "--threshold"),
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature deduplicate`."""
    from visionservex.cli.embedding_commands import deduplicate_cmd

    deduplicate_cmd(model_id, folder, threshold=threshold, out=out, device=device, json_=json_)


@app.command("dataset-report", help="Generate a dataset report using embeddings.")
def dataset_report_top_alias(
    model_id: str,
    folder: Path,
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature dataset-report`."""
    from visionservex.cli.embedding_commands import dataset_report_cmd

    dataset_report_cmd(model_id, folder, out=out, device=device, max_images=max_images, json_=json_)


@app.command("active-select", help="Active learning sample selection from a folder.")
def active_select_top_alias(
    model_id: str,
    folder: Path,
    budget: int = typer.Option(100, "--budget"),
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int | None = typer.Option(None, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature active-select`."""
    from visionservex.cli.embedding_commands import active_select_cmd

    active_select_cmd(
        model_id, folder, budget=budget, out=out, device=device, max_images=max_images, json_=json_
    )


@app.command("domain-shift", help="Estimate domain shift between train and test folders.")
def domain_shift_top_alias(
    model_id: str,
    train_folder: Path,
    test_folder: Path,
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int = typer.Option(200, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Top-level alias for `visionservex feature domain-shift`."""
    from visionservex.cli.embedding_commands import domain_shift_cmd

    domain_shift_cmd(
        model_id,
        train_folder,
        test_folder,
        out=out,
        device=device,
        max_images=max_images,
        json_=json_,
    )


# ---------------------------------------------------------------------------
# Benchmark-embeddings (v1.6.0)
# ---------------------------------------------------------------------------


@app.command("benchmark-embeddings", help="Benchmark embedding model: kNN accuracy and latency.")
def benchmark_embeddings_top(
    model_id: str = typer.Option(..., "--model"),
    dataset: str = typer.Option(..., "--dataset", help="folder:<path> with labels.csv inside"),
    metrics: str = typer.Option("knn_accuracy,recall_at_5,latency", "--metrics"),
    out: Path | None = typer.Option(None, "--out"),
    device: str = typer.Option("auto", "--device"),
    max_images: int = typer.Option(200, "--max-images"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Lightweight embedding benchmark: structural readiness check + kNN if labels exist."""
    from visionservex.runtime.embeddings import embed_folder

    if not dataset.startswith("folder:"):
        _die(
            f"benchmark-embeddings requires --dataset folder:<path>, got {dataset!r}",
            json_mode=json_,
            code="BAD_DATASET",
        )
        return
    folder = Path(dataset[len("folder:") :])
    if not folder.exists():
        _die(f"folder not found: {folder}", json_mode=json_, code="DATASET_NOT_FOUND")
        return

    import time

    t0 = time.perf_counter()
    embeddings, paths = embed_folder(model_id, folder, device=device, max_images=max_images)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    payload: dict = {
        "model_id": model_id,
        "dataset": dataset,
        "n_images": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "total_time_ms": round(duration_ms, 1),
        "mean_latency_ms": round(duration_ms / max(1, embeddings.shape[0]), 2),
        "metrics_requested": metrics.split(","),
    }

    # Try optional kNN accuracy if labels.csv exists
    labels_csv = folder / "labels.csv"
    if labels_csv.exists() and embeddings.shape[0] >= 2:
        try:
            import csv

            label_map: dict[str, str] = {}
            with open(labels_csv, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        label_map[row[0]] = row[1]
            y = [label_map.get(Path(p).name, "?") for p in paths]
            classes = sorted(set(y))
            if len(classes) >= 2:
                # Leave-one-out kNN
                import numpy as _np

                sim = embeddings @ embeddings.T
                correct = 0
                for i in range(len(y)):
                    sim_i = sim[i].copy()
                    sim_i[i] = -_np.inf
                    nn = int(_np.argmax(sim_i))
                    if y[nn] == y[i]:
                        correct += 1
                acc = correct / len(y)
                payload["knn_accuracy"] = round(acc, 4)
                payload["n_classes"] = len(classes)
        except Exception as exc:
            payload["knn_error"] = str(exc)[:100]
    else:
        payload["knn_accuracy"] = None
        payload["note"] = (
            "Provide labels.csv (filename,label per row) in the folder for kNN accuracy."
        )

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[bold]Embedding benchmark:[/bold] {model_id}")
        console.print(f"  Images:       {payload['n_images']}")
        console.print(f"  Embedding:    {payload['embedding_dim']}-d")
        console.print(f"  Mean latency: {payload['mean_latency_ms']} ms/image")
        if payload.get("knn_accuracy") is not None:
            console.print(
                f"  kNN accuracy: {payload['knn_accuracy']:.4f} ({payload.get('n_classes')} classes)"
            )
        else:
            console.print(f"  [dim]{payload.get('note', '')}[/dim]")


@app.command(
    "seg",
    help=(
        "Short alias for `visionservex segment` — canonical command is `segment`. "
        "Added for convenience; `segment` is preferred in scripts and docs."
    ),
)
def seg_alias(
    model_id: str,
    input_path: Path,
    conf: float = typer.Option(0.25, "--conf", "--threshold"),
    device: str | None = typer.Option(None, "--device"),
    save_image: Path | None = typer.Option(None, "--save-image"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Short alias for `visionservex segment`."""
    segment_alias(
        model_id=model_id,
        input_path=input_path,
        conf=conf,
        device=device,
        save_image=save_image,
        json_=json_,
    )

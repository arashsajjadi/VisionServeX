# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Local model gateway management commands.

The gateway is the VisionServeX HTTP server branded as a local model gateway.
All `gateway` commands map to the same underlying FastAPI app.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(help="Local model gateway management.")
console = Console()

_PROFILES: dict[str, dict] = {
    "laptop": {
        "VISIONSERVEX_SERVER__HOST": "127.0.0.1",
        "VISIONSERVEX_RUNTIME__MAX_LOADED_MODELS": "2",
        "VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY": "1",
        "VISIONSERVEX_RUNTIME__QUEUE_SIZE": "8",
        "VISIONSERVEX_MODELS__AUTO_PULL": "true",
        "VISIONSERVEX_MODELS__AUTO_PULL_POLICY": "easy_only",
    },
    "gpu-workstation": {
        "VISIONSERVEX_SERVER__HOST": "127.0.0.1",
        "VISIONSERVEX_RUNTIME__MAX_LOADED_MODELS": "4",
        "VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY": "2",
        "VISIONSERVEX_RUNTIME__QUEUE_SIZE": "32",
        "VISIONSERVEX_MODELS__AUTO_PULL": "true",
        "VISIONSERVEX_MODELS__AUTO_PULL_POLICY": "registry_allowed",
    },
    "cpu-safe": {
        "VISIONSERVEX_SERVER__HOST": "127.0.0.1",
        "VISIONSERVEX_RUNTIME__MAX_LOADED_MODELS": "1",
        "VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY": "1",
        "VISIONSERVEX_RUNTIME__QUEUE_SIZE": "4",
        "VISIONSERVEX_MODELS__AUTO_PULL": "false",
    },
    "public-tunnel-safe": {
        "VISIONSERVEX_SERVER__HOST": "127.0.0.1",
        "VISIONSERVEX_AUTH__ENABLED": "true",
        "VISIONSERVEX_RUNTIME__MAX_LOADED_MODELS": "2",
        "VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY": "1",
        "VISIONSERVEX_RUNTIME__QUEUE_SIZE": "16",
        "VISIONSERVEX_MODELS__AUTO_PULL": "false",
    },
}


@app.command("start", help="Start the local model gateway.")
def start(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    profile: str | None = typer.Option(
        None, "--profile", help="Config profile: laptop|gpu-workstation|cpu-safe|public-tunnel-safe"
    ),
    preload: str | None = typer.Option(
        None, "--preload", help="Comma-separated model IDs to warm up on startup."
    ),
    auto_pull: bool = typer.Option(False, "--auto-pull", help="Enable auto-pull on first request."),
    auth: bool = typer.Option(False, "--auth", help="Enable API key authentication."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to YAML gateway config file."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Start the VisionServeX local model gateway.

    This is a local-first API server for computer vision inference.
    Bind to 127.0.0.1 by default (change with --host or --profile).
    """
    import os

    from visionservex.config import reload_settings
    from visionservex.utils.logging import configure_logging

    configure_logging("INFO")

    # Apply profile env vars
    if profile and profile in _PROFILES:
        for k, v in _PROFILES[profile].items():
            os.environ.setdefault(k, v)
        if not json_:
            console.print(f"[dim]Profile {profile!r} applied.[/dim]")
    elif profile:
        console.print(
            f"[yellow]Unknown profile {profile!r}. Available: {', '.join(_PROFILES)}[/yellow]"
        )

    # Apply config file if provided
    if config_file and config_file.exists():
        os.environ["VISIONSERVEX_CONFIG_FILE"] = str(config_file)
    # Apply flags
    if auto_pull:
        os.environ.setdefault("VISIONSERVEX_MODELS__AUTO_PULL", "true")
    if auth:
        os.environ.setdefault("VISIONSERVEX_AUTH__ENABLED", "true")

    settings = reload_settings(**{"server": {"host": host, "port": port}})

    for w in settings.public_safety_warnings():
        console.print(f"[yellow]warning:[/yellow] {w}")

    if not json_:
        console.print(
            Panel.fit(
                f"[bold]VisionServeX Local Gateway[/bold]\n"
                f"http://{host}:{port}  |  docs at http://{host}:{port}/docs",
                border_style="green",
            )
        )
        console.print(f"  Models endpoint:  http://{host}:{port}/models")
        console.print(f"  Health endpoint:  http://{host}:{port}/health")
        console.print(f"  Metrics:          http://{host}:{port}/metrics")
        console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed.[/red] Run: pip install 'visionservex[server]'")
        raise typer.Exit(1)

    uvicorn.run(
        "visionservex.server.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="warning",
    )


@app.command("status", help="Show gateway status (requires gateway to be running).")
def status(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        import httpx

        r = httpx.get(f"{url}/gateway/status", timeout=3.0)
        data = r.json()
    except Exception as exc:
        if json_:
            typer.echo(json.dumps({"error": str(exc), "running": False}))
        else:
            console.print(f"[red]Gateway not reachable at {url}[/red]: {exc}")
            console.print("Start with: [cyan]visionservex gateway start[/cyan]")
        return

    if json_:
        typer.echo(json.dumps(data, indent=2, default=str))
        return

    console.print(f"[green]Gateway running[/green] at {url}")
    if isinstance(data, dict):
        for k, v in data.items():
            console.print(f"  {k}: {v}")


@app.command("doctor", help="Diagnose gateway configuration.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.config import get_settings
    from visionservex.runtime.device import best_device
    from visionservex.utils.system import probe_dependencies

    settings = get_settings()
    best = best_device()
    deps = probe_dependencies()
    warnings = settings.public_safety_warnings()

    payload = {
        "bind": f"{settings.server.host}:{settings.server.port}",
        "public_mode": settings.server.public_mode,
        "auth_enabled": settings.auth.enabled,
        "auto_pull": settings.models.auto_pull,
        "auto_pull_policy": settings.models.auto_pull_policy,
        "best_device": best.to_dict(),
        "max_loaded_models": settings.runtime.max_loaded_models,
        "queue_size": settings.runtime.queue_size,
        "server_dep": deps.get("fastapi", {}).get("installed", False),
        "warnings": warnings,
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print("[bold]Gateway Doctor[/bold]")
    table = Table(show_header=False, box=None)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("Bind address", payload["bind"])
    table.add_row("Best device", f"{best.name} — {best.detail[:60]}")
    table.add_row("Auto-pull", str(payload["auto_pull"]))
    table.add_row("Max loaded models", str(payload["max_loaded_models"]))
    table.add_row(
        "FastAPI installed", "[green]yes[/green]" if payload["server_dep"] else "[red]no[/red]"
    )
    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")
    if not payload["server_dep"]:
        console.print("\nInstall server: [cyan]pip install 'visionservex[server]'[/cyan]")


@app.command("profile", help="Show or apply a configuration profile.")
def profile_cmd(
    name: str = typer.Argument(
        ..., help="Profile name: laptop|gpu-workstation|cpu-safe|public-tunnel-safe"
    ),
    show: bool = typer.Option(True, "--show/--apply", help="Show env vars or export them."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if name not in _PROFILES:
        console.print(f"[red]Unknown profile:[/red] {name!r}. Available: {', '.join(_PROFILES)}")
        raise typer.Exit(1)
    env = _PROFILES[name]
    if json_:
        typer.echo(json.dumps(env, indent=2))
        return
    console.print(f"[bold]Profile: {name}[/bold]")
    for k, v in env.items():
        console.print(f"  export {k}={v}")
    console.print(
        f'\nApply: eval $(visionservex gateway profile {name} --json | jq -r \'to_entries[] | "export " + .key + "=" + .value\')'
    )


@app.command("preload", help="Warm up models in the loaded cache.")
def preload_cmd(
    model_ids: str = typer.Argument(..., help="Comma-separated model IDs to preload."),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex import VisionModel

    ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    results = []
    for mid in ids:
        try:
            m = VisionModel(mid, device=device)
            m.warmup()
            results.append({"model_id": mid, "status": "ok", "device": m.device})
            if not json_:
                console.print(f"  [green]ok[/green] {mid} on {m.device}")
        except Exception as exc:
            results.append({"model_id": mid, "status": "error", "error": str(exc)[:100]})
            if not json_:
                console.print(f"  [red]err[/red] {mid}: {str(exc)[:80]}")
    if json_:
        typer.echo(json.dumps(results, indent=2, default=str))


@app.command("client-example", help="Show Python client usage examples.")
def client_example(json_: bool = typer.Option(False, "--json")) -> None:
    examples = {
        "basic": """from visionservex import Client
client = Client("http://127.0.0.1:8080")

# Detection
result = client.detect("dfine-n", "image.jpg")
print(result)

# Classification
result = client.classify("swinv2-tiny", "image.jpg")
for label, score in result.results[:5]:
    print(label, score)

# Text-prompted segmentation
result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
print(f"{len(result.results)} segments")
""",
        "curl": """# Health
curl http://127.0.0.1:8080/health

# List models
curl http://127.0.0.1:8080/models | jq '.models[].id'

# Detection
curl -F "image=@image.jpg" -F "model_id=dfine-n" http://127.0.0.1:8080/detect

# GPU smoke
visionservex gpu smoke-test --models dfine-n,swinv2-tiny --device cuda
""",
    }
    if json_:
        typer.echo(json.dumps(examples, indent=2))
        return
    console.print("[bold]Python Client Examples[/bold]")
    console.print(examples["basic"])
    console.print("[bold]curl Examples[/bold]")
    console.print(examples["curl"])


@app.command("openapi", help="Print or open the OpenAPI spec URL.")
def openapi(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    open_browser: bool = typer.Option(False, "--open"),
) -> None:
    spec_url = f"{url}/openapi.json"
    docs_url = f"{url}/docs"
    console.print(f"OpenAPI spec:  {spec_url}")
    console.print(f"Swagger UI:    {docs_url}")
    console.print(f"ReDoc:         {url}/redoc")
    if open_browser:
        import webbrowser

        webbrowser.open(docs_url)


@app.command("loaded-models", help="Show currently loaded models (requires running gateway).")
def loaded_models(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        import httpx

        r = httpx.get(f"{url}/gateway/status", timeout=3.0)
        data = r.json()
        models = data.get("loaded_models", [])
    except Exception as exc:
        if json_:
            typer.echo(json.dumps({"error": str(exc), "loaded_models": []}))
        else:
            console.print(f"[red]Cannot reach gateway at {url}[/red]: {exc}")
        return
    if json_:
        typer.echo(json.dumps(models, indent=2, default=str))
        return
    if not models:
        console.print("[dim]No models currently loaded.[/dim]")
    else:
        from rich.table import Table

        table = Table(title="Loaded models")
        for col in ("id", "task", "device", "engine"):
            table.add_column(col)
        for m in models:
            table.add_row(
                m.get("model_id", "?"),
                m.get("task", "?"),
                m.get("device", "?"),
                m.get("engine", "?"),
            )
        console.print(table)


@app.command("memory", help="Show GPU/CPU memory usage (requires running gateway or local probe).")
def memory(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.runtime.device import available_devices

    payload: dict = {"devices": []}
    for d in available_devices():
        if d.available and d.total_vram_gb:
            payload["devices"].append(
                {
                    "name": d.name,
                    "total_vram_gb": d.total_vram_gb,
                    "free_vram_gb": d.free_vram_gb,
                    "detail": d.detail,
                }
            )
    # Also try to get loaded model info from gateway
    try:
        import httpx

        r = httpx.get(f"{url}/gateway/status", timeout=2.0)
        payload["loaded_models"] = r.json().get("loaded_models", [])
    except Exception:
        payload["loaded_models"] = []

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    for d in payload["devices"]:
        console.print(f"{d['name']:8s}: {d['free_vram_gb']:.1f}/{d['total_vram_gb']:.1f} GB free")
    if payload["loaded_models"]:
        console.print(f"Loaded models: {len(payload['loaded_models'])}")


@app.command("stop", help="Stop the gateway (send SIGTERM to the server process).")
def stop(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
) -> None:
    """Graceful stop via /shutdown endpoint (if implemented) or inform user."""
    try:
        import httpx

        httpx.post(f"{url}/shutdown", timeout=3.0)
        console.print(f"Gateway stop signal sent to {url}")
    except Exception:
        console.print(
            f"[yellow]Could not reach {url}.[/yellow]\n"
            "The gateway runs in the foreground — press [bold]Ctrl+C[/bold] in the terminal where it is running."
        )


__all__ = ["app"]


@app.command("health", help="Quick health check against a running gateway.")
def health(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    try:
        import httpx

        r = httpx.get(f"{url}/health", timeout=3.0)
        data = r.json()
        if json_:
            typer.echo(json.dumps(data, indent=2, default=str))
        else:
            status = data.get("status", "unknown")
            color = "green" if status == "ok" else "red"
            console.print(f"Gateway [{color}]{status}[/{color}] at {url}")
    except Exception as exc:
        if json_:
            typer.echo(json.dumps({"error": str(exc), "running": False}))
        else:
            console.print(f"[red]Gateway not reachable at {url}[/red]")


@app.command("logs", help="Tail gateway logs (runs in foreground — use stdout redirect).")
def logs(lines: int = typer.Option(50, "--lines", "-n")) -> None:
    console.print(
        "[yellow]Note:[/yellow] VisionServeX logs to stdout by default.\n"
        "Capture logs with: [cyan]visionservex gateway start 2>&1 | tee gateway.log[/cyan]"
    )


@app.command("config", help="Show active gateway configuration.")
def show_config(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.config import get_settings
    from visionservex.utils.logging import log_safe_dict

    s = get_settings()
    safe = log_safe_dict(s.model_dump())
    if json_:
        typer.echo(json.dumps(safe, indent=2, default=str))
    else:
        console.print("[bold]Active gateway configuration[/bold] (secrets redacted):")
        for section, val in safe.items():
            if isinstance(val, dict):
                console.print(f"  [cyan]{section}[/cyan]: {val}")


@app.command("profile-list", help="List available gateway profiles.")
def profile_list(json_: bool = typer.Option(False, "--json")) -> None:
    if json_:
        typer.echo(json.dumps(list(_PROFILES.keys()), indent=2))
    else:
        for name in _PROFILES:
            console.print(f"  [cyan]{name}[/cyan]")


@app.command("token", help="Generate a local API key for development.")
def token_create() -> None:
    import secrets

    key = secrets.token_urlsafe(48)
    console.print(f"Generated API key: [bold]{key}[/bold]")
    console.print("  [cyan]export VISIONSERVEX_AUTH__ENABLED=true[/cyan]")
    console.print(f"  [cyan]export VISIONSERVEX_AUTH__API_KEY={key}[/cyan]")

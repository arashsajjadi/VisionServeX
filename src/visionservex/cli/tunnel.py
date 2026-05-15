"""CLI subcommands for Cloudflare Tunnel.

These are thin wrappers around the ``visionservex.tunnel.cloudflare`` module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from visionservex.config import get_settings
from visionservex.tunnel.cloudflare import (
    CloudflaredNotFound,
    cloudflared_doctor,
    generate_config,
    public_checklist,
)

app = typer.Typer(help="Cloudflare Tunnel integration (external `cloudflared` binary).")
console = Console()


def _json_or_text(payload, *, json_mode: bool, text: str | None = None) -> None:
    if json_mode:
        typer.echo(json.dumps(payload, indent=2, default=str))
    elif text is not None:
        console.print(text)
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("doctor", help="Check that `cloudflared` is installed and tells you how to install it if not.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    report = cloudflared_doctor()
    _json_or_text(report, json_mode=json_)


@app.command("login", help="Run `cloudflared tunnel login`.")
def login() -> None:
    _ensure_cloudflared()
    subprocess.run(["cloudflared", "tunnel", "login"], check=False)


@app.command("create", help="Create a named tunnel.")
def create(name: Optional[str] = typer.Argument(None)) -> None:
    _ensure_cloudflared()
    settings = get_settings()
    tunnel_name = name or settings.tunnel.tunnel_name
    subprocess.run(["cloudflared", "tunnel", "create", tunnel_name], check=False)


@app.command("route", help="Route a DNS name to the tunnel.")
def route(tunnel_name: str, hostname: str) -> None:
    _ensure_cloudflared()
    subprocess.run(
        ["cloudflared", "tunnel", "route", "dns", tunnel_name, hostname],
        check=False,
    )


@app.command("config", help="Generate a safe ingress config for the local API.")
def config(
    hostname: str = typer.Argument(..., help="Public hostname, e.g. api.example.com"),
    tunnel_name: Optional[str] = typer.Option(None, "--tunnel-name"),
    output: Optional[Path] = typer.Option(None, "--out"),
    print_only: bool = typer.Option(False, "--print"),
) -> None:
    settings = get_settings()
    tn = tunnel_name or settings.tunnel.tunnel_name
    payload = generate_config(tunnel_name=tn, hostname=hostname)
    if print_only or output is None:
        console.print(payload)
    if output:
        output.write_text(payload, encoding="utf-8")
        console.print(f"wrote {output}")


@app.command("run", help="Run the tunnel using a generated or user-supplied config file.")
def run(
    config_file: Path = typer.Argument(..., exists=True, readable=True),
    confirm_public: bool = typer.Option(
        False,
        "--i-understand-this-is-public",
        help="Required flag confirming you understand this exposes the API publicly.",
    ),
) -> None:
    settings = get_settings()
    if not confirm_public:
        console.print(
            "[red]refusing to run the tunnel without explicit confirmation.[/red]\n"
            "Pass --i-understand-this-is-public after reviewing the checklist below."
        )
        for item in public_checklist(auth_enabled=settings.auth.enabled):
            console.print(f"  [yellow]?[/yellow] {item}")
        raise typer.Exit(2)
    if not settings.auth.enabled:
        console.print(
            "[red]refusing to run the tunnel because authentication is disabled.[/red] "
            "Set VISIONSERVEX_AUTH__ENABLED=true and VISIONSERVEX_AUTH__API_KEY=... first."
        )
        raise typer.Exit(2)
    _ensure_cloudflared()
    subprocess.run(["cloudflared", "tunnel", "--config", str(config_file), "run"], check=False)


def _ensure_cloudflared() -> None:
    if shutil.which("cloudflared") is None:
        raise CloudflaredNotFound(
            "`cloudflared` is not installed. See docs/cloudflare_tunnel.md for install instructions."
        )


__all__ = ["app"]

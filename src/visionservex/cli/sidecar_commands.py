# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.23.0: `visionservex sidecar list / doctor / create / exec` CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="v2.23.0: universal sidecar manager (DEIMv2, RT-DETRv4, ...).",
    no_args_is_help=True,
)
console = Console()


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        color = {
            "ok": "green",
            "expected_blocker": "yellow",
            "failed": "red",
        }.get(payload.get("status", ""), "white")
        console.print(f"[{color}]{payload.get('code', '')}[/{color}]: {payload.get('message', '')}")


@app.command("list")
def list_cmd(
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """List all known sidecar specs."""
    from visionservex.sidecars import SidecarManager

    specs = {k: v.to_dict() for k, v in SidecarManager.list_specs().items()}
    payload = {"status": "ok", "code": "OK", "n_specs": len(specs), "specs": specs}
    _emit(payload, out=out, fmt=fmt)


@app.command("doctor")
def doctor_cmd(
    name: str = typer.Argument(..., help="Sidecar spec name (deimv2 | rtdetrv4 | ...)."),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Diagnose a sidecar environment (does not modify anything)."""
    from visionservex.sidecars import SidecarManager

    payload = SidecarManager().doctor(name)
    _emit(payload, out=out, fmt=fmt)


@app.command("create")
def create_cmd(
    name: str = typer.Argument(..., help="Sidecar spec name."),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Default: plan only."),
    timeout_s: int = typer.Option(
        1800, "--timeout-s", help="Per-command timeout (default 1800s = 30 min)."
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Plan (default) or execute creation of a sidecar conda environment."""
    from visionservex.sidecars import SidecarConfig, SidecarManager

    cfg = SidecarConfig(timeout_s=timeout_s)
    payload = SidecarManager().create(name, dry_run=dry_run, config=cfg)
    _emit(payload, out=out, fmt=fmt)


@app.command("exec")
def exec_cmd(
    name: str = typer.Argument(..., help="Sidecar spec name."),
    command: list[str] = typer.Argument(..., help="Command (everything after the sidecar name)."),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    expect_json: bool = typer.Option(
        True,
        "--expect-json/--no-expect-json",
        help="Parse stdout as JSON (default).",
    ),
) -> None:
    """Run a command inside a sidecar env; return structured JSON result."""
    from visionservex.sidecars import SidecarManager

    result = SidecarManager().exec(name, list(command), expect_json=expect_json)
    payload = result.to_dict()
    payload.setdefault("message", "")
    _emit(payload, out=out, fmt=fmt)


__all__ = ["app"]

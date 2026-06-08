# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""LocateAnything-3B CLI commands — VisionServeX v3.6.

NVIDIA LocateAnything-3B is released under the NVIDIA License for
non-commercial use only. VisionServeX does not ship or mirror the weights.
Use is BYOT/user-local-cache only.

Commands:
  list        List all LocateAnything-3B model IDs with license status.
  status      Return structured status for a given model ID.
  run         Run LocateAnything-3B grounded detection (--accept-noncommercial required).
  install     Print sidecar install instructions.
  explain     Print full explain() dict for a model ID.

IMPORTANT: Every command that executes inference requires --accept-noncommercial.
"""

from __future__ import annotations

import json
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="locate-anything",
    help=(
        "LocateAnything-3B commands (NVIDIA License — non-commercial only, BYOT). "
        "All inference commands require --accept-noncommercial."
    ),
    no_args_is_help=True,
)
console = Console()

_WARNING = (
    "WARNING: LocateAnything-3B pretrained weights are released under the NVIDIA License "
    "for non-commercial use only. Do not use this model for commercial products, paid SaaS, "
    "client work, production annotation, or redistribution unless you have written commercial "
    "permission from NVIDIA. VisionServeX does not ship or mirror the weights. "
    "Use is BYOT/user-local-cache only."
)

_MODEL_IDS = [
    "locate-anything-3b",
    "locate-anything-3b-v2",
    "locate-anything-3b-grounded",
    "locate-anything-3b-coco",
    "locate-anything-3b-lvis",
    "locate-anything-3b-objects365",
    "locate-anything-3b-open-vocab",
    "locate-anything-3b-caption",
    "locate-anything-3b-video",
    "locate-anything-3b-ft",
]

_SIDECAR_INSTALL = (
    "git clone https://github.com/NVlabs/Eagle.git eagle && cd eagle/Embodied && pip install -e ."
)


def _emit(payload: dict, *, out: str | None, fmt: str) -> None:
    if out:
        from pathlib import Path

        Path(out).write_text(json.dumps(payload, indent=2, default=str))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("list")
def cmd_list(
    fmt: str = typer.Option("table", "--format", help="Output format: table or json."),
    out: str | None = typer.Option(None, "--out", help="Write JSON to file."),
) -> None:
    """List all LocateAnything-3B model IDs with license status."""
    rows = [
        {
            "model_id": m,
            "license": "NVIDIA License (non-commercial only)",
            "default_safe": False,
            "commercial_safe": False,
            "state": "legal_review_required",
            "byot": True,
        }
        for m in _MODEL_IDS
    ]
    if fmt == "json" or out:
        _emit({"models": rows}, out=out, fmt=fmt)
        return
    t = Table(title="LocateAnything-3B Models")
    t.add_column("model_id", style="cyan")
    t.add_column("license")
    t.add_column("default_safe")
    t.add_column("state")
    for r in rows:
        t.add_row(r["model_id"], r["license"], str(r["default_safe"]), r["state"])
    console.print(t)
    console.print(f"[yellow]{_WARNING}[/yellow]")


@app.command("status")
def cmd_status(
    model_id: str = typer.Argument(..., help="LocateAnything-3B model ID."),
    fmt: str = typer.Option("json", "--format", help="Output format: json."),
    out: str | None = typer.Option(None, "--out", help="Write JSON to file."),
) -> None:
    """Return structured status for a given LocateAnything-3B model ID."""
    from visionservex.vsx import VSX

    handle = VSX.locateanything(model_id)
    payload = handle.explain()
    _emit(payload, out=out, fmt=fmt)


@app.command("explain")
def cmd_explain(
    model_id: str = typer.Argument(..., help="LocateAnything-3B model ID."),
    out: str | None = typer.Option(None, "--out", help="Write JSON to file."),
) -> None:
    """Print the full explain() dict for a LocateAnything-3B model ID."""
    from visionservex.vsx import VSX

    payload = VSX.locateanything(model_id).explain()
    _emit(payload, out=out, fmt="json")


@app.command("install")
def cmd_install() -> None:
    """Print sidecar install instructions for LocateAnything-3B."""
    console.print(f"[yellow]{_WARNING}[/yellow]")
    console.print("\n[bold]Sidecar install commands:[/bold]")
    console.print(f"  {_SIDECAR_INSTALL}")
    console.print(
        "\n[bold]After install, run:[/bold]\n"
        "  visionservex locate-anything run locate-anything-3b image.jpg "
        "--text 'cat' --accept-noncommercial --out result.json"
    )


@app.command("run")
def cmd_run(
    model_id: str = typer.Argument(..., help="LocateAnything-3B model ID."),
    image: str = typer.Argument(..., help="Path to input image."),
    text: str = typer.Option(..., "--text", help="Text prompt for grounded detection."),
    accept_noncommercial: bool = typer.Option(
        False,
        "--accept-noncommercial",
        help=(
            "Acknowledge the NVIDIA non-commercial license. Required to run inference. "
            "See: visionservex locate-anything install"
        ),
    ),
    out: str | None = typer.Option(None, "--out", help="Write result JSON to file."),
    fmt: str = typer.Option("json", "--format", help="Output format."),
) -> None:
    """Run LocateAnything-3B grounded detection.

    Requires --accept-noncommercial to acknowledge the NVIDIA non-commercial license.
    Requires sidecar install: visionservex locate-anything install
    """
    print(_WARNING, file=sys.stderr)
    if not accept_noncommercial:
        payload = {
            "status": "expected_blocker",
            "code": "NONCOMMERCIAL_ACKNOWLEDGMENT_REQUIRED",
            "message": (
                "--accept-noncommercial flag is required to run LocateAnything-3B. "
                "This acknowledges the NVIDIA non-commercial license."
            ),
            "next_command": (
                f"visionservex locate-anything run {model_id} {image} "
                f"--text '{text}' --accept-noncommercial --out result.json"
            ),
        }
        _emit(payload, out=out, fmt=fmt)
        raise typer.Exit(0)

    from visionservex.vsx import VSX

    try:
        result = VSX.locateanything(model_id).locate(image, text=text, accept_noncommercial=True)
        payload = {
            "status": "ok",
            "model_id": model_id,
            "text": text,
            "result": result,
        }
    except Exception as exc:
        payload = {
            "status": "expected_blocker",
            "code": "SIDECAR_REQUIRED",
            "message": str(exc),
            "next_command": "visionservex locate-anything install",
        }
    _emit(payload, out=out, fmt=fmt)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Audit export commands — generate machine-readable manifests for notebooks."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(
    help="Audit export commands — generate machine-readable manifests for notebooks.",
    no_args_is_help=True,
)
console = Console()


@app.command("export-model-inventory")
def export_model_inventory(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export a full model inventory with notebook eligibility flags."""
    from visionservex.audit.builder import export_model_inventory as _export

    payload = _export()
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(json.dumps({"out": str(out), "n_models": payload["n_models"]}, indent=2))
    else:
        typer.echo(text)


@app.command("export-feature-inventory")
def export_feature_inventory(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export feature/capability inventory grouped by notebook section."""
    from visionservex.audit.builder import export_feature_inventory as _export

    payload = _export()
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(json.dumps({"out": str(out), "n_sections": payload["n_sections"]}, indent=2))
    else:
        typer.echo(text)


@app.command("export-command-inventory")
def export_command_inventory(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export an inventory of every public CLI command with help status."""
    from visionservex.audit.builder import export_command_inventory as _export

    payload = _export()
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(json.dumps({"out": str(out), "n_commands": len(payload)}, indent=2))
    else:
        typer.echo(text)


@app.command("export-notebook-manifest")
def export_notebook_manifest(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export the complete notebook input manifest (models + commands + groups)."""
    from visionservex.audit.builder import export_notebook_manifest as _export

    payload = _export()
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(
            json.dumps(
                {
                    "out": str(out),
                    "n_models": len(payload["models"]),
                    "n_sections": len(payload["notebook_sections"]),
                },
                indent=2,
            )
        )
    else:
        typer.echo(text)


@app.command("export-benchmark-plan")
def export_benchmark_plan(
    fmt: str = typer.Option("markdown", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export the benchmark plan as markdown."""
    from visionservex.audit.builder import export_benchmark_plan as _export

    text = _export()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        typer.echo(json.dumps({"out": str(out)}, indent=2))
    else:
        typer.echo(text)


@app.command("export-ultralytics-plan")
def export_ultralytics_plan(
    fmt: str = typer.Option("json", "--format"),
    out: Path = typer.Option(None, "--out"),
) -> None:
    """Export Ultralytics comparison plan with eligible models and caveats."""
    from visionservex.audit.builder import export_ultralytics_comparison_plan as _export

    payload = _export()
    text = json.dumps(payload, indent=2)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        n = len(payload["eligible_visionservex_models"])
        typer.echo(json.dumps({"out": str(out), "n_eligible_models": n}, indent=2))
    else:
        typer.echo(text)


@app.command("bundle")
def bundle(
    out_dir: str = typer.Option("docs/audit", "--out-dir"),
    also_tmp: bool = typer.Option(False, "--also-tmp", help="Also write copies under /tmp."),
) -> None:
    """Build all audit artifacts and write them under out-dir."""
    from visionservex.audit.builder import build_audit_bundle

    written = build_audit_bundle(out_dir)
    if also_tmp:
        import shutil

        for _name, path in written.items():
            dest = f"/tmp/{Path(path).name}"
            shutil.copy2(path, dest)

    typer.echo(json.dumps({"n_artifacts": len(written), "artifacts": written}, indent=2))


__all__ = ["app"]

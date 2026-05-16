# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model zoo CLI: sources, verify-links, export."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Source-grounded model manifest commands.")


@app.command("sources", help="List all model sources with URLs and license.")
def sources_cmd(
    runnable_only: bool = typer.Option(False, "--runnable-only"),
    domain: str | None = typer.Option(None, "--domain"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import SOURCE_MANIFEST

    entries = list(SOURCE_MANIFEST.values())
    if runnable_only:
        entries = [e for e in entries if e.runnable_in_visionservex]
    if domain:
        entries = [e for e in entries if e.domain.lower() == domain.lower()]

    if json_:
        typer.echo(json.dumps([e.to_dict() for e in entries], indent=2))
        return

    table = Table(title=f"Model sources ({len(entries)})")
    for col in ("Model ID", "Family", "Task", "Runnable", "Access", "Action", "License"):
        table.add_column(col)
    for e in entries:
        run = "[green]yes[/green]" if e.runnable_in_visionservex else "[grey50]no[/grey50]"
        access = e.access_status
        access_color = {"open": "green", "api_token": "yellow", "gated": "red"}.get(access, "white")
        action_color = {
            "add_now": "green",
            "expert_sidecar": "yellow",
            "external_api": "magenta",
            "audit_only": "grey50",
            "do_not_add": "red",
            "non_core_license_optional": "yellow",
        }.get(e.recommended_action, "white")
        table.add_row(
            e.model_id,
            e.family,
            e.task,
            run,
            f"[{access_color}]{access}[/{access_color}]",
            f"[{action_color}]{e.recommended_action}[/{action_color}]",
            f"{e.license}{' ⚠' if e.license_risk not in ('none', '') else ''}",
        )
    console.print(table)


@app.command("verify-links", help="Static verification of manifest (no network calls).")
def verify_links_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.model_zoo import verify_manifest

    report = verify_manifest()
    if json_:
        typer.echo(json.dumps(report, indent=2))
        return
    counts = report["counts"]
    console.print("[bold]Manifest verification[/bold]")
    console.print(f"  Total entries:       {counts['total']}")
    console.print(f"  Runnable:            [green]{counts['runnable']}[/green]")
    console.print(f"  Expert sidecar:      [yellow]{counts.get('expert_sidecar', 0)}[/yellow]")
    console.print(f"  External API:        [magenta]{counts.get('external_api', 0)}[/magenta]")
    console.print(f"  Audit only:          [grey50]{counts.get('audit_only', 0)}[/grey50]")
    console.print(
        f"  Non-core license:    [yellow]{counts.get('non_core_license_optional', 0)}[/yellow]"
    )
    console.print(f"  Do not add:          [red]{counts.get('do_not_add', 0)}[/red]")
    if report["issues"]:
        console.print("\n[yellow]Issues:[/yellow]")
        for issue in report["issues"]:
            console.print(f"  {issue['model_id']}: {issue['issue']}")
    else:
        console.print("\n[green]No structural issues found.[/green]")


@app.command("export", help="Export manifest to JSON or markdown.")
def export_cmd(
    format_: str = typer.Option("json", "--format", help="json | markdown"),
    out: Path = typer.Option(Path("docs/model_zoo_manifest.json"), "--out"),
) -> None:
    from visionservex.model_zoo import SOURCE_MANIFEST

    out.parent.mkdir(parents=True, exist_ok=True)
    if format_ == "json":
        payload = {mid: src.to_dict() for mid, src in SOURCE_MANIFEST.items()}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elif format_ == "markdown":
        lines = ["# VisionServeX Model Zoo Manifest", ""]
        lines.append("| Model ID | Family | Task | Runnable | Access | License | Action |")
        lines.append("|----------|--------|------|----------|--------|---------|--------|")
        for _mid, src in sorted(SOURCE_MANIFEST.items()):
            run = "✓" if src.runnable_in_visionservex else "—"
            lines.append(
                f"| `{src.model_id}` | {src.family} | {src.task} | {run} | "
                f"{src.access_status} | {src.license} | {src.recommended_action} |"
            )
        lines.append("")
        lines.append("## Sources")
        for _mid, src in sorted(SOURCE_MANIFEST.items()):
            lines.append(f"### `{src.model_id}`")
            if src.official_repo:
                lines.append(f"- Official: <{src.official_repo}>")
            if src.hf_repo:
                lines.append(f"- HF: `{src.hf_repo}`")
            if src.paper_url:
                lines.append(f"- Paper: <{src.paper_url}>")
            if src.known_blockers:
                lines.append("- Blockers:")
                for b in src.known_blockers:
                    lines.append(f"  - {b}")
            lines.append("")
        out.write_text("\n".join(lines), encoding="utf-8")
    else:
        console.print(f"[red]unknown format: {format_}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Exported manifest to {out}[/green]")


@app.command("show", help="Show source detail for one model.")
def show_cmd(
    model_id: str,
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import get_model_source

    src = get_model_source(model_id)
    if src is None:
        console.print(f"[red]not in manifest:[/red] {model_id}")
        raise typer.Exit(1)
    if json_:
        typer.echo(json.dumps(src.to_dict(), indent=2))
        return
    console.print(f"[bold]{src.model_id}[/bold]  ({src.family}, {src.task})")
    if src.official_repo:
        console.print(f"  Repo:        {src.official_repo}")
    if src.official_docs:
        console.print(f"  Docs:        {src.official_docs}")
    if src.hf_repo:
        console.print(f"  HF:          {src.hf_repo}")
    if src.paper_url:
        console.print(f"  Paper:       {src.paper_url}")
    console.print(f"  License:     {src.license} ({src.license_risk})")
    console.print(f"  Install:     {src.install_command}")
    console.print(f"  Runnable:    {src.runnable_in_visionservex}")
    console.print(f"  Action:      {src.recommended_action}")
    if src.known_blockers:
        console.print("  Blockers:")
        for b in src.known_blockers:
            console.print(f"    - {b}")
    if src.notes:
        console.print(f"  Notes:       {src.notes}")


__all__ = ["app"]

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


@app.command("gap-report")
def gap_report(
    format_: str = typer.Option("markdown", "--format"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a gap report from SOURCE_MANIFEST grouped by recommended_action."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    groups: dict[str, list] = {
        "runnable": [],
        "optional_extra": [],
        "expert_sidecar": [],
        "external_api": [],
        "do_not_add": [],
        "audit_only": [],
        "unavailable": [],
        "non_core_license_optional": [],
    }
    for entry in SOURCE_MANIFEST.values():
        action = entry.recommended_action
        if entry.runnable_in_visionservex and action == "add_now":
            groups["runnable"].append(entry)
        elif action == "expert_sidecar":
            groups["expert_sidecar"].append(entry)
        elif action == "external_api":
            groups["external_api"].append(entry)
        elif action == "do_not_add":
            groups["do_not_add"].append(entry)
        elif action == "non_core_license_optional":
            groups["non_core_license_optional"].append(entry)
        elif action == "audit_only":
            if entry.known_blockers:
                groups["unavailable"].append(entry)
            else:
                groups["audit_only"].append(entry)
        else:
            groups["audit_only"].append(entry)

    payload = {grp: [e.to_dict() for e in entries] for grp, entries in groups.items()}
    counts = {grp: len(entries) for grp, entries in groups.items()}
    payload["_counts"] = counts  # type: ignore[assignment]

    if json_:
        print(json.dumps(payload, indent=2))
        return

    if format_ == "json":
        text = json.dumps(payload, indent=2)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]Gap report written to {out}[/green]")
        else:
            print(text)
        return

    lines = ["# VisionServeX Model Zoo Gap Report", ""]
    section_labels = {
        "runnable": "Runnable (wired, usable now)",
        "optional_extra": "Optional extras (require extra install)",
        "expert_sidecar": "Expert sidecars (OpenMMLab, Detectron2, etc.)",
        "external_api": "External / gated APIs",
        "non_core_license_optional": "Non-core license (optional)",
        "do_not_add": "Excluded (do_not_add with reason)",
        "audit_only": "Audit only (no blockers yet)",
        "unavailable": "Unresolved blockers",
    }
    for grp, label in section_labels.items():
        entries = groups[grp]
        lines.append(f"## {label} ({len(entries)})")
        lines.append("")
        if entries:
            lines.append("| Model ID | Family | Task | License | Blockers |")
            lines.append("|----------|--------|------|---------|---------|")
            for e in entries:
                blockers = "; ".join(e.known_blockers[:2]) if e.known_blockers else "-"
                lines.append(
                    f"| `{e.model_id}` | {e.family} | {e.task} | {e.license} | {blockers} |"
                )
        else:
            lines.append("_None_")
        lines.append("")

    text = "\n".join(lines)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Gap report written to {out}[/green]")
    else:
        console.print(text)


@app.command("matrix")
def matrix(
    format_: str = typer.Option("markdown", "--format"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
    family: str = typer.Option("", "--family", help="Filter by family"),
    domain: str = typer.Option("", "--domain", help="Filter by domain"),
) -> None:
    """Generate a full model matrix from SOURCE_MANIFEST."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    entries = list(SOURCE_MANIFEST.values())
    if family:
        entries = [e for e in entries if e.family.lower() == family.lower()]
    if domain:
        entries = [e for e in entries if e.domain.lower() == domain.lower()]

    rows = []
    for e in entries:
        rows.append(
            {
                "model_id": e.model_id,
                "family": e.family,
                "task": e.task,
                "status": "runnable" if e.runnable_in_visionservex else e.recommended_action,
                "license": e.license,
                "install": e.install_command,
                "source_url": e.official_repo or e.hf_repo or "",
                "blockers": e.known_blockers,
            }
        )

    if json_:
        print(json.dumps(rows, indent=2))
        return

    if format_ == "json":
        text = json.dumps(rows, indent=2)
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            console.print(f"[green]Matrix written to {out}[/green]")
        else:
            print(text)
        return

    lines = ["# VisionServeX Model Matrix", ""]
    lines.append("| Model ID | Family | Task | Status | License | Install | Source | Blockers |")
    lines.append("|----------|--------|------|--------|---------|---------|--------|---------|")
    for row in rows:
        blockers = "; ".join(row["blockers"][:1]) if row["blockers"] else "-"
        lines.append(
            f"| `{row['model_id']}` | {row['family']} | {row['task']} | "
            f"{row['status']} | {row['license']} | `{row['install']}` | "
            f"{row['source_url'] or '-'} | {blockers} |"
        )
    lines.append("")

    text = "\n".join(lines)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Matrix written to {out}[/green]")
    else:
        console.print(text)


@app.command("blockers")
def blockers_cmd(
    family: str = typer.Option("", "--family"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show known blockers for a model family."""
    from visionservex.model_zoo import SOURCE_MANIFEST

    entries = list(SOURCE_MANIFEST.values())
    if family:
        entries = [e for e in entries if e.family.lower() == family.lower()]

    blocked = [e for e in entries if e.known_blockers]
    rows = []
    for e in blocked:
        rows.append(
            {
                "model_id": e.model_id,
                "family": e.family,
                "recommended_action": e.recommended_action,
                "known_blockers": e.known_blockers,
                "install": e.install_command,
            }
        )

    if json_:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        label = f" for family {family!r}" if family else ""
        console.print(f"[green]No blockers found{label}.[/green]")
        return

    table = Table(title=f"Known blockers ({len(rows)})", show_header=True)
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Family", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Blockers")
    for row in rows:
        blockers_text = "\n".join(f"- {b}" for b in row["known_blockers"])
        table.add_row(
            row["model_id"],
            row["family"],
            row["recommended_action"],
            blockers_text,
        )
    console.print(table)


__all__ = ["app"]

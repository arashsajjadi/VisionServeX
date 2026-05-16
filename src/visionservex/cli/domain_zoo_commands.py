# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Domain-zoo CLI: list domains, recommend pipelines per goal."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Domain-specialized model recommendations.")


@app.command("list", help="List all available domains.")
def list_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    from visionservex.model_zoo import DOMAIN_ZOO, list_domains

    if json_:
        payload = {d: list(DOMAIN_ZOO[d].keys()) for d in list_domains()}
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print("[bold]Available domains:[/bold]")
    for d in list_domains():
        goals = list(DOMAIN_ZOO[d].keys())
        console.print(f"  [cyan]{d}[/cyan]: {', '.join(goals)}")


@app.command("recommend", help="Recommend a pipeline for a domain + goal.")
def recommend_cmd(
    domain: str = typer.Option(..., "--domain"),
    goal: str | None = typer.Option(None, "--goal"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.model_zoo import recommend_for_domain

    recipes = recommend_for_domain(domain, goal)
    if not recipes:
        console.print(f"[yellow]No recipes for domain={domain!r} goal={goal!r}[/yellow]")
        from visionservex.model_zoo import list_domains

        console.print(f"  Available: {', '.join(list_domains())}")
        raise typer.Exit(1)

    if json_:
        typer.echo(json.dumps([r.to_dict() for r in recipes], indent=2))
        return

    for recipe in recipes:
        from rich.panel import Panel

        runnable = (
            "[green]✓ runnable today[/green]"
            if recipe.runnable_today
            else "[yellow]⚠ roadmap / expert-sidecar[/yellow]"
        )
        console.print(
            Panel.fit(
                f"[bold]{recipe.domain}[/bold] / {recipe.goal}\n"
                f"{recipe.description}\n\n"
                f"Status: {runnable}",
                border_style="cyan",
            )
        )
        console.print("\n[bold]Pipeline:[/bold]")
        for step in recipe.pipeline:
            console.print(f"  {step}")
        console.print("\n[bold]Recommended models:[/bold]")
        for m in recipe.recommended_models:
            console.print(f"  • {m}")
        if recipe.install_commands:
            console.print("\n[bold]Install:[/bold]")
            for c in recipe.install_commands:
                console.print(f"  [cyan]$[/cyan] {c}")
        if recipe.quick_commands:
            console.print("\n[bold]Quick commands:[/bold]")
            for c in recipe.quick_commands:
                console.print(f"  [cyan]$[/cyan] {c}")
        if recipe.expected_hardware:
            console.print(f"\n[bold]Hardware:[/bold] {recipe.expected_hardware}")
        if recipe.limitations:
            console.print("\n[bold]Limitations:[/bold]")
            for lim in recipe.limitations:
                console.print(f"  [yellow]⚠[/yellow] {lim}")
        if recipe.license_notes:
            console.print("\n[bold]License notes:[/bold]")
            for note in recipe.license_notes:
                console.print(f"  [dim]{note}[/dim]")
        console.print()


@app.command("yolo26-competitors", help="Show YOLO26-X competitor recommendations.")
def yolo26_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="yolo26-competitors", goal=None, json_=json_)


@app.command("sam-family", help="Show SAM-family recommendations.")
def sam_family_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="sam-family", goal=None, json_=json_)


@app.command("medical", help="Show medical-domain recommendations (extras).")
def medical_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="medical", goal=None, json_=json_)


@app.command("agriculture", help="Show agriculture-domain recommendations.")
def agriculture_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="agriculture", goal=None, json_=json_)


@app.command("aerial", help="Show aerial / OBB recommendations.")
def aerial_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="aerial", goal=None, json_=json_)


@app.command("industrial", help="Show industrial anomaly recommendations.")
def industrial_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="industrial", goal=None, json_=json_)


@app.command("surveillance", help="Show surveillance / video-search recommendations.")
def surveillance_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="surveillance", goal=None, json_=json_)


@app.command("feature-intelligence", help="Show DINOv2 / feature-backbone recommendations.")
def feature_intelligence_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="feature-intelligence", goal=None, json_=json_)


@app.command("promptable", help="Show open-vocabulary / promptable recommendations.")
def promptable_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    recommend_cmd(domain="promptable", goal=None, json_=json_)


@app.command(
    "export",
    help="Export domain zoo to markdown.",
)
def export_cmd(
    out: Path = typer.Option(Path("docs/domain_zoo.md"), "--out"),
) -> None:
    from visionservex.model_zoo import DOMAIN_ZOO

    lines = ["# VisionServeX Domain Zoo", ""]
    lines.append(
        "Curated pipelines per vertical. Each recipe is honest: status, install, "
        "hardware, license, and known limitations are documented.\n"
    )
    for domain in sorted(DOMAIN_ZOO):
        lines.append(f"## {domain}\n")
        for goal_key, recipe in DOMAIN_ZOO[domain].items():
            lines.append(f"### {goal_key}")
            run = "✓ runnable today" if recipe.runnable_today else "⚠ roadmap / expert"
            lines.append(f"_{recipe.description}_  ")
            lines.append(f"**Status:** {run}\n")
            lines.append("**Pipeline:**")
            for step in recipe.pipeline:
                lines.append(f"1. {step}")
            lines.append("\n**Recommended models:**")
            for m in recipe.recommended_models:
                lines.append(f"- `{m}`")
            if recipe.install_commands:
                lines.append("\n**Install:**")
                for c in recipe.install_commands:
                    lines.append(f"```\n{c}\n```")
            if recipe.limitations:
                lines.append("\n**Limitations:**")
                for lim in recipe.limitations:
                    lines.append(f"- {lim}")
            lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Exported domain zoo to {out}[/green]")


__all__ = ["app"]

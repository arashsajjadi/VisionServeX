# SPDX-License-Identifier: Apache-2.0
"""`visionservex readiness` — v2.9 readiness factor table + release verdict."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from visionservex.readiness import READINESS_ROWS, compute_readiness_table, is_row_release_ready

app = typer.Typer(
    help="v2.9 readiness factor table — every row carries evidence + verdict.",
    no_args_is_help=True,
)
console = Console()


@app.command("table")
def table_cmd(
    out: Path = typer.Option(None, "--out", help="JSON output path."),
    md: Path = typer.Option(None, "--md", help="Markdown output path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Emit the full readiness table."""
    rows = compute_readiness_table()
    payload = {
        "rows": rows,
        "n": len(rows),
        "n_release_ready": sum(1 for r in rows if r["release_ready"]),
        "all_ready": all(r["release_ready"] for r in rows),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if md is not None:
        lines = [
            "| Factor | v2.8 | Functional | Operational | Certainty | Verdict |",
            "|--------|-----:|----------:|-----------:|---------:|:-------:|",
        ]
        for r in rows:
            verdict = "✅ ready" if r["release_ready"] else "❌ below 90"
            lines.append(
                f"| {r['factor']} | {r['v2_8']} | {r['functional']} | "
                f"{r['operational']} | {r['blocker_certainty']} | {verdict} |"
            )
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text("\n".join(lines) + "\n")
    if json_:
        print(json.dumps(payload, indent=2))
        return
    table = Table(title=f"v2.9 readiness ({payload['n_release_ready']}/{payload['n']} ready)")
    for col, justify in (
        ("Factor", "left"),
        ("v2.8", "right"),
        ("Func", "right"),
        ("Op", "right"),
        ("Cert", "right"),
        ("Verdict", "center"),
    ):
        table.add_column(col, justify=justify)
    for r in rows:
        verdict = "[green]ready[/green]" if r["release_ready"] else "[red]<90[/red]"
        table.add_row(
            r["factor"],
            str(r["v2_8"]),
            str(r["functional"]),
            str(r["operational"]),
            str(r["blocker_certainty"]),
            verdict,
        )
    console.print(table)


@app.command("verdict")
def verdict_cmd(json_: bool = typer.Option(False, "--json")) -> None:
    """Print a single-line release verdict for v2.9."""
    rows = list(READINESS_ROWS)
    not_ready = [r.factor for r in rows if not is_row_release_ready(r)]
    payload = {
        "n_rows": len(rows),
        "n_ready": len(rows) - len(not_ready),
        "all_ready": not not_ready,
        "not_ready": not_ready,
        "verdict": "RELEASE_OK" if not not_ready else "RELEASE_BLOCKED",
    }
    if json_:
        print(json.dumps(payload, indent=2))
        return
    if payload["all_ready"]:
        console.print(f"[green]RELEASE_OK[/green] — {payload['n_ready']}/{payload['n_rows']} rows")
    else:
        console.print(f"[red]RELEASE_BLOCKED[/red] — pending: {', '.join(not_ready)}")


__all__ = ["app"]

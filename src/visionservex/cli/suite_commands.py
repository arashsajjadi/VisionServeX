# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""pull-suite and scheduler commands."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

console = Console()

suite_app = typer.Typer(help="Pull curated model suites.")
scheduler_app = typer.Typer(help="Scheduler policy inspection and tuning.")

# Curated suites: name → list of model IDs
_SUITES: dict[str, list[str]] = {
    "beginner": ["dfine-n", "swinv2-tiny", "sam2-hiera-tiny"],
    "gpu-demo": [
        "rfdetr-nano",
        "dfine-n",
        "swinv2-tiny",
        "sam2-hiera-tiny",
        "grounding-dino-tiny",
        "grounded-sam2",
    ],
    "server-demo": ["dfine-n", "sam2-hiera-tiny", "grounded-sam2"],
    "detection": ["rfdetr-nano", "rfdetr-small", "dfine-n", "dfine-s", "grounding-dino-tiny"],
    "segmentation": ["rfdetr-seg-nano", "sam-vit-base", "sam2-hiera-tiny", "grounded-sam"],
    "classification": ["swinv2-tiny", "swinv2-small"],
}

# Model-aware scheduler concurrency policies
# Based on observed GPU parallel benchmark results:
_MODEL_POLICIES: dict[str, dict] = {
    "rfdetr-nano": {
        "policy": "gpu_exclusive",
        "max_concurrency": 1,
        "note": "rfdetr package not thread-safe",
    },
    "rfdetr-small": {"policy": "gpu_exclusive", "max_concurrency": 1},
    "dfine-n": {
        "policy": "queue_recommended",
        "max_concurrency": 1,
        "note": "215% slowdown at concurrency=2",
    },
    "dfine-s": {"policy": "queue_recommended", "max_concurrency": 1},
    "swinv2-tiny": {
        "policy": "acceptable_parallelism",
        "max_concurrency": 2,
        "note": "80% slowdown at concurrency=2",
    },
    "swinv2-small": {"policy": "acceptable_parallelism", "max_concurrency": 2},
    "sam2-hiera-tiny": {
        "policy": "gpu_exclusive",
        "max_concurrency": 1,
        "note": "large memory, exclusive preferred",
    },
    "sam-vit-base": {"policy": "gpu_exclusive", "max_concurrency": 1},
    "grounding-dino-tiny": {
        "policy": "queue_recommended",
        "max_concurrency": 1,
        "note": "text encoding overhead",
    },
    "grounded-sam2": {"policy": "gpu_exclusive", "max_concurrency": 1, "note": "composed pipeline"},
    "grounded-sam": {"policy": "gpu_exclusive", "max_concurrency": 1},
    "oneformer-swin-large": {
        "policy": "gpu_exclusive",
        "max_concurrency": 1,
        "note": "very large model",
    },
}


@suite_app.command("list", help="List available model suites.")
def suite_list(json_: bool = typer.Option(False, "--json")) -> None:
    if json_:
        typer.echo(json.dumps(dict(_SUITES.items()), indent=2))
        return
    table = Table(title="Model suites")
    table.add_column("Suite")
    table.add_column("Models")
    for name, models in _SUITES.items():
        table.add_row(name, ", ".join(models))
    console.print(table)
    console.print("\nPull a suite: [cyan]visionservex pull-suite beginner --yes[/cyan]")


@suite_app.command("pull", help="Pull all models in a named suite.")
def suite_pull(
    suite_name: str = typer.Argument(..., help="Suite name (see suite list)."),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.cli.benchmark_commands import _pull_with_progress
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.downloads import (
        DownloadError,
        ManualDownloadRequired,
        download,
    )

    if suite_name not in _SUITES:
        msg = f"Unknown suite {suite_name!r}. Available: {', '.join(_SUITES)}"
        if json_:
            typer.echo(json.dumps({"error": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    model_ids = _SUITES[suite_name]
    reg = default_registry()

    if not yes and not json_:
        console.print(f"Suite [bold]{suite_name}[/bold] ({len(model_ids)} models):")
        for mid in model_ids:
            try:
                e = reg.get(mid)
                sz = f" ~{e.size_bytes / 1e9:.1f}GB" if e.size_bytes else ""
                console.print(f"  {mid}{sz}")
            except RegistryError:
                console.print(f"  {mid} [red](not in registry)[/red]")
        if not typer.confirm("Proceed?"):
            raise typer.Exit(1)

    results = []
    for mid in model_ids:
        entry = {"model_id": mid}
        try:
            e = reg.get(mid)
            path = download(e) if json_ else _pull_with_progress(e)
            entry.update({"status": "ok", "path": str(path)})
        except (DownloadError, ManualDownloadRequired) as exc:
            entry.update({"status": "skip", "reason": str(exc)[:100]})
            if not json_:
                console.print(f"  [yellow]skip[/yellow] {mid}: {str(exc)[:80]}")
        except Exception as exc:
            entry.update({"status": "error", "error": str(exc)[:100]})
            if not json_:
                console.print(f"  [red]error[/red] {mid}: {str(exc)[:80]}")
        results.append(entry)

    if json_:
        typer.echo(json.dumps(results, indent=2, default=str))
    else:
        ok = sum(1 for r in results if r.get("status") == "ok")
        console.print(f"\n{ok}/{len(results)} models ready.")


@scheduler_app.command("profile", help="Show model-aware concurrency policies.")
def scheduler_profile(
    model: str | None = typer.Option(None, "--model"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if model:
        policy = _MODEL_POLICIES.get(model, {"policy": "default", "max_concurrency": 2})
        if json_:
            typer.echo(json.dumps({model: policy}, indent=2))
        else:
            console.print(
                f"[bold]{model}[/bold]: policy={policy['policy']} max_concurrency={policy.get('max_concurrency', 2)}"
            )
            if "note" in policy:
                console.print(f"  [dim]{policy['note']}[/dim]")
        return

    if json_:
        typer.echo(json.dumps(_MODEL_POLICIES, indent=2))
        return

    table = Table(title="Model scheduler policies")
    table.add_column("Model")
    table.add_column("Policy")
    table.add_column("Max concurrency")
    table.add_column("Note")
    for mid, p in _MODEL_POLICIES.items():
        color = {
            "gpu_exclusive": "yellow",
            "queue_recommended": "cyan",
            "acceptable_parallelism": "green",
        }.get(p["policy"], "white")
        table.add_row(
            mid,
            f"[{color}]{p['policy']}[/{color}]",
            str(p.get("max_concurrency", 2)),
            p.get("note", ""),
        )
    console.print(table)


@scheduler_app.command("recommend", help="Recommend concurrency policy for a model.")
def scheduler_recommend(
    model_id: str = typer.Option(..., "--model"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    policy = _MODEL_POLICIES.get(model_id, {"policy": "default", "max_concurrency": 2})
    payload = {
        "model_id": model_id,
        "device": device,
        "recommended_policy": policy.get("policy", "default"),
        "recommended_max_concurrency": policy.get("max_concurrency", 2),
        "note": policy.get("note", "No specific benchmark data available; using default."),
        "env_suggestion": f"VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY={policy.get('max_concurrency', 2)}",
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]Scheduler recommendation for {model_id}[/bold]")
    console.print(f"  Policy:              {payload['recommended_policy']}")
    console.print(f"  Max concurrency:     {payload['recommended_max_concurrency']}")
    console.print(f"  Note:                {payload['note']}")
    console.print(f"  Config:              {payload['env_suggestion']}")


def get_model_max_concurrency(model_id: str) -> int:
    """Return the recommended max concurrency for a model."""
    return _MODEL_POLICIES.get(model_id, {}).get("max_concurrency", 2)


__all__ = ["_MODEL_POLICIES", "_SUITES", "get_model_max_concurrency", "scheduler_app", "suite_app"]

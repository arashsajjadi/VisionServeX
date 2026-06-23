# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""License-policy CLI commands for the `visionservex models` group.

Adds: list (with --commercial-safe / --research filters), explain, policy,
assert-commercial-safe. The commercial-safe-by-default policy is the source of
truth; restricted models fail closed.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def register(app: typer.Typer) -> None:
    """Attach the policy commands to the given (models) typer app."""

    @app.command("list")
    def list_models_cmd(
        commercial_safe: bool = typer.Option(
            False, "--commercial-safe", help="Only commercial-safe models."
        ),
        research: bool = typer.Option(False, "--research", help="Only research-only models."),
        byo: bool = typer.Option(False, "--byo", help="Only BYO-license/checkpoint models."),
        legal_review: bool = typer.Option(
            False, "--legal-review", help="Only legal-review (not commercial-safe) models."
        ),
        json_: bool = typer.Option(False, "--json"),
    ) -> None:
        """List models with their commercial-safety status."""
        from visionservex.policy import (
            get_model_policy,
            list_byo_models,
            list_commercial_safe_models,
            list_legal_review_models,
            list_models,
            list_research_models,
        )

        if commercial_safe:
            ids = list_commercial_safe_models()
        elif research:
            ids = list_research_models()
        elif byo:
            ids = list_byo_models()
        elif legal_review:
            ids = list_legal_review_models()
        else:
            ids = list_models()
        rows = [
            {
                "model_id": m,
                "commercial_status": (p := get_model_policy(m)).commercial_status,
                "tier": p.default_package_tier,
                "commercial_safe": p.is_commercial_safe,
                "acknowledgement_required": p.requires_acknowledgement,
            }
            for m in ids
        ]
        if json_:
            print(json.dumps(rows, indent=2))
            return
        table = Table(title=f"Models ({len(rows)})", show_header=True)
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Commercial status")
        table.add_column("Tier")
        table.add_column("Safe")
        for r in rows:
            safe = "[green]yes[/green]" if r["commercial_safe"] else "[red]no[/red]"
            table.add_row(r["model_id"], r["commercial_status"], r["tier"], safe)
        console.print(table)

    @app.command("coverage")
    def coverage_cmd(json_: bool = typer.Option(False, "--json")) -> None:
        """Policy-coverage report: curated vs registry-derived (commercial-safe)."""
        from visionservex.policy import policy_coverage

        rep = policy_coverage()
        if json_:
            print(json.dumps(rep, indent=2))
            return
        console.print(
            f"total={rep['total_models']}  commercial_safe={rep['commercial_safe_total']} "
            f"(curated={rep['commercial_safe_curated']}, "
            f"registry_derived={rep['commercial_safe_registry_derived']}, "
            f"curated_pct={rep['commercial_safe_curated_pct']}%)"
        )
        console.print(f"by_status: {rep['by_commercial_status']}")
        console.print(
            f"registry-derived commercial-safe: {rep['registry_derived_commercial_safe_ids']}"
        )

    @app.command("explain")
    def explain_cmd(model_id: str = typer.Argument(...)) -> None:
        """Explain a model's license + commercial-safety in plain language."""
        from visionservex.policy import explain_model_license

        console.print(explain_model_license(model_id))

    @app.command("policy")
    def policy_cmd(
        model_id: str = typer.Argument(...),
        json_: bool = typer.Option(False, "--json"),
    ) -> None:
        """Print the full machine-readable policy for a model."""
        from visionservex.policy import get_model_policy

        payload = get_model_policy(model_id).to_dict()
        if json_:
            print(json.dumps(payload, indent=2))
        else:
            console.print_json(json.dumps(payload))

    @app.command("assert-commercial-safe")
    def assert_commercial_safe_cmd(
        model_id: str = typer.Argument(...),
        json_: bool = typer.Option(False, "--json"),
    ) -> None:
        """Exit 0 if the model is commercial-safe; non-zero (with reason) otherwise."""
        from visionservex.exceptions import ModelNotCommercialSafeError
        from visionservex.policy import assert_commercial_safe, get_model_policy

        try:
            assert_commercial_safe(model_id)
        except ModelNotCommercialSafeError as exc:
            payload = {
                "model_id": model_id,
                "commercial_safe": False,
                "code": exc.code,
                "commercial_status": get_model_policy(model_id).commercial_status,
            }
            print(json.dumps(payload, indent=2)) if json_ else console.print(
                f"[red]NOT commercial-safe[/red]: {model_id} "
                f"({payload['commercial_status']}) [{exc.code}]"
            )
            raise typer.Exit(3) from exc
        payload = {"model_id": model_id, "commercial_safe": True}
        print(json.dumps(payload, indent=2)) if json_ else console.print(
            f"[green]commercial-safe[/green]: {model_id}"
        )


__all__ = ["register"]

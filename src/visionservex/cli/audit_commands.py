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


@app.command("validate")
def validate_cmd(
    audit_dir: Path = typer.Option(Path("docs/audit"), "--audit-dir"),
    out: Path = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate all audit artifacts for schema consistency and notebook-safety."""
    import csv as _csv

    issues: list[str] = []
    passed: list[str] = []

    # Check JSON validity
    for fname in audit_dir.glob("*.json"):
        try:
            json.loads(fname.read_text())
            passed.append(f"JSON valid: {fname.name}")
        except json.JSONDecodeError as exc:
            issues.append(f"INVALID_JSON: {fname.name} — {exc}")

    # Required keys in notebook manifest
    manifest_path = audit_dir / "visionservex_notebook_input_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        required_keys = [
            "package",
            "families",
            "models",
            "commands",
            "benchmark_groups",
            "ultralytics_comparison",
            "expected_blockers",
            "sidecars",
            "optional_extras",
            "license_risks",
            "notebook_sections",
        ]
        for key in required_keys:
            if key not in manifest:
                issues.append(f"MANIFEST_MISSING_KEY: {key}")
            else:
                passed.append(f"manifest_key: {key}")

        # Model-level checks
        embedding_families = {"dinov2", "clip", "siglip", "siglip2"}
        for m in manifest.get("models", []):
            if not m.get("notebook_section"):
                issues.append(f"MODEL_MISSING_SECTION: {m.get('model_id')}")
            if m.get("eligible_for_ultralytics_comparison"):
                fam = m.get("family", "")
                if m.get("task") not in ("detect",) or fam in embedding_families:
                    issues.append(
                        f"INVALID_UC_ELIGIBLE: {m.get('model_id')} "
                        f"task={m.get('task')} family={fam}"
                    )

        # Expected blocker codes
        for b in manifest.get("expected_blockers", []):
            if not b.get("code"):
                issues.append(f"BLOCKER_MISSING_CODE: {b}")

    # CSV column check
    csv_path = audit_dir / "visionservex_model_test_matrix.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        for col in ("model_id", "family", "eligible_for_ultralytics_comparison", "smoke_command"):
            if col not in fieldnames:
                issues.append(f"CSV_MISSING_COLUMN: {col}")
            else:
                passed.append(f"csv_col: {col}")

    payload = {
        "audit_dir": str(audit_dir),
        "n_passed": len(passed),
        "n_issues": len(issues),
        "issues": issues,
        "verdict": "VALID" if not issues else "INVALID",
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    color = "green" if not issues else "red"
    console.print(
        f"[{color}]{payload['verdict']}[/{color}] — {len(passed)} passed, {len(issues)} issues"
    )
    for iss in issues:
        console.print(f"  ✗ {iss}")

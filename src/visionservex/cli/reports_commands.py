# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: `visionservex reports {audit-truth, validate-final-states}` CLIs.

These commands operate over a reports directory (e.g. the v31 notebook
output) and produce structured truth-audit verdicts. The notebook reads
these audit artifacts and refuses to write a "v3_ready" verdict if any
of the required-zero counts is non-zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from visionservex.reporting.status_vocab import (
    ALLOWED_FINAL_STATES,
    FORBIDDEN_FINAL_STATES,
)
from visionservex.reporting.truth_audit import audit_reports_dir

app = typer.Typer(
    help="v2.28.0: reporting truth audit + final-state validation.",
    no_args_is_help=True,
)
console = Console()


def _emit(payload: dict, *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = "green" if payload.get("status") == "ok" else "red"
    console.print(f"[{color}]{payload.get('code', '?')}[/{color}]")
    for k in (
        "raw_nan_count_final",
        "not_wired_count_final",
        "failed_runtime_parseable_blocker_count",
        "stale_marker_count",
        "empty_status_count",
        "missing_blocker_code_count",
        "missing_source_status_count",
    ):
        console.print(f"  {k}: {payload.get(k, 0)}")


@app.command("audit-truth")
def audit_truth_cmd(
    reports_dir: Path = typer.Option(..., "--reports-dir"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit non-zero when any required-zero count > 0.",
    ),
) -> None:
    """v2.28.0: scan a reports directory for raw NaN / NOT_WIRED / stale text."""
    if not reports_dir.exists():
        _emit(
            {
                "status": "failed",
                "code": "INPUT_NOT_FOUND",
                "message": f"--reports-dir {reports_dir} not found.",
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)
    payload = audit_reports_dir(reports_dir)
    _emit(payload, out=out, fmt=fmt)
    if strict and payload.get("status") != "ok":
        raise typer.Exit(3)


@app.command("validate-final-states")
def validate_final_states_cmd(
    reports_dir: Path = typer.Option(..., "--reports-dir"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """v2.28.0: validate that all final_state cells use the canonical vocabulary.

    Reads ``canonical_model_state_v228.csv`` if present and flags any
    final_state value outside :data:`ALLOWED_FINAL_STATES`.
    """
    import csv as _csv

    candidate_paths = [
        reports_dir / "canonical_model_state_v228.csv",
        reports_dir / "canonical_model_state_v228.json",
    ]
    found = None
    for p in candidate_paths:
        if p.exists():
            found = p
            break
    payload = {
        "status": "ok",
        "code": "OK",
        "reports_dir": str(reports_dir),
        "n_rows": 0,
        "n_allowed": 0,
        "n_forbidden": 0,
        "offending_rows": [],
        "allowed_vocab_size": len(ALLOWED_FINAL_STATES),
    }
    if found is None:
        payload["status"] = "expected_blocker"
        payload["code"] = "CANONICAL_MODEL_STATE_NOT_FOUND"
        payload["message"] = "canonical_model_state_v228.{csv,json} not present in the reports dir."
        _emit(payload, out=out, fmt=fmt)
        return

    rows: list[dict] = []
    if found.suffix == ".csv":
        rows = list(_csv.DictReader(found.open()))
    else:
        d = json.loads(found.read_text())
        rows = d.get("rows") if isinstance(d, dict) else []
    payload["n_rows"] = len(rows)
    for r in rows:
        fs = (r.get("final_state") or "").strip()
        if fs in ALLOWED_FINAL_STATES:
            payload["n_allowed"] += 1
        else:
            payload["n_forbidden"] += 1
            payload["offending_rows"].append(
                {
                    "model_id": r.get("model_id"),
                    "final_state": fs,
                    "in_forbidden_set": fs in FORBIDDEN_FINAL_STATES,
                }
            )
    if payload["n_forbidden"] > 0:
        payload["status"] = "expected_blocker"
        payload["code"] = "FORBIDDEN_FINAL_STATE_PRESENT"
    _emit(payload, out=out, fmt=fmt)


@app.command("information-ledger")
def information_ledger_cmd(
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option(..., "--out"),
) -> None:
    """v2.28.0: machine-readable ledger of information still required."""
    import csv as _csv
    from pathlib import Path as _P

    from visionservex.reporting.information_ledger import build_information_ledger

    rows = build_information_ledger()
    _P(out).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        fields = list(rows[0].keys()) if rows else []
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))
        return
    _P(out).write_text(
        json.dumps({"status": "ok", "code": "OK", "n_rows": len(rows), "rows": rows}, indent=2)
    )
    typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))


@app.command("reconcile-model-states")
def reconcile_model_states_cmd(
    registry: Path = typer.Option(
        None,
        "--registry",
        help="Path to notebook/shared/model_registry.yaml (informational).",
    ),
    task_reports: Path = typer.Option(
        Path("notebook"),
        "--task-reports",
        help="Root to walk for per-task reports (default: ./notebook).",
    ),
    resolution: Path = typer.Option(
        Path("reports/v238_49_blocked_resolution_matrix.json"),
        "--resolution",
        help="Latest 49-row resolution matrix.",
    ),
    notebook_call_ledger: Path = typer.Option(
        Path("notebook/99_final_report/reports/notebook_model_call_ledger.json"),
        "--notebook-call-ledger",
        help="Notebook call ledger JSON.",
    ),
    out_json: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.json"),
        "--out-json",
    ),
    out_csv: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.csv"),
        "--out-csv",
    ),
    final_winners: Path = typer.Option(
        Path("notebook/99_final_report/reports/final_winners.json"),
        "--final-winners",
    ),
    fail_on_stale: bool = typer.Option(False, "--fail-on-stale"),
    fail_on_missing_notebook_calls: bool = typer.Option(False, "--fail-on-missing-notebook-calls"),
    write_provenance: bool = typer.Option(False, "--write-provenance"),
    historical_fallback_ledger: Path | None = typer.Option(
        None,
        "--historical-fallback-ledger",
        help=(
            "v2.46: path to a previous model_coverage_ledger.csv. Defaults "
            "to the existing --out-csv so reruns don't lose benchmark_passed "
            "rows whose per-model evidence was cleaned."
        ),
    ),
) -> None:
    """v2.39.0: reconcile registry + task reports + matrix + ledger into a canonical ledger."""
    from visionservex.reporting.v239_reconciler import (
        fail_on_missing_notebook_calls as _fail_missing,
    )
    from visionservex.reporting.v239_reconciler import (
        fail_on_stale as _fail_stale,
    )
    from visionservex.reporting.v239_reconciler import (
        reconcile,
        write_outputs,
    )

    fallback_path = historical_fallback_ledger
    if fallback_path is None and out_csv.exists():
        fallback_path = out_csv
    payload = reconcile(
        registry_path=registry,
        task_reports_root=task_reports,
        resolution_matrix_path=resolution if resolution.exists() else None,
        notebook_call_ledger_path=notebook_call_ledger if notebook_call_ledger.exists() else None,
        historical_fallback_ledger=fallback_path,
    )
    write_outputs(
        payload,
        out_json=out_json,
        out_csv=out_csv,
        final_winners=final_winners,
        write_provenance=write_provenance,
    )
    stale_issues = _fail_stale(payload)
    missing_issues = _fail_missing(payload)
    summary = {
        "status": "ok"
        if (not stale_issues or not fail_on_stale)
        and (not missing_issues or not fail_on_missing_notebook_calls)
        else "failed",
        "out_json": str(out_json),
        "out_csv": str(out_csv),
        "final_winners": str(final_winners),
        "total_rows": payload.get("total", 0),
        "stale_issues_count": len(stale_issues),
        "stale_issues_sample": stale_issues[:10],
        "missing_notebook_calls_count": len(missing_issues),
        "missing_notebook_calls_sample": missing_issues[:10],
        "n_called_in_notebook": payload.get("n_called_in_notebook", 0),
    }
    typer.echo(json.dumps(summary, indent=2))
    if fail_on_stale and stale_issues:
        raise typer.Exit(3)
    if fail_on_missing_notebook_calls and missing_issues:
        raise typer.Exit(4)


@app.command("audit-stale-final-tables")
def audit_stale_final_tables_cmd(
    notebook_root: Path = typer.Option(Path("notebook"), "--notebook-root"),
    reports_root: Path | None = typer.Option(None, "--reports-root"),
    target_models_file: Path | None = typer.Option(
        None,
        "--target-models",
        help="Optional file with one model_id per line. Defaults to DEFAULT_TARGET_MODELS_49.",
    ),
    out: Path = typer.Option(..., "--out"),
    fmt: str = typer.Option("json", "--format"),
    fail_on_stale: bool = typer.Option(False, "--fail-on-stale"),
) -> None:
    """v2.39.0: scan every CSV/JSON/MD/notebook for stale 49-target rows."""
    from visionservex.reporting.v239_stale_audit import (
        DEFAULT_TARGET_MODELS_49,
        audit_stale_final_tables,
    )

    target_models: list[str] = list(DEFAULT_TARGET_MODELS_49)
    if target_models_file and target_models_file.exists():
        target_models = [
            line.strip()
            for line in target_models_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    payload = audit_stale_final_tables(
        notebook_root=notebook_root,
        reports_root=reports_root,
        target_models=target_models,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(
            f"[{'green' if payload['status'] == 'ok' else 'red'}]stale={payload['total_issues']}[/]"
        )
        for k, v in payload["counts"].items():
            console.print(f"  {k}: {v}")
    if fail_on_stale and payload["status"] != "ok":
        raise typer.Exit(5)


@app.command("verify-generated-artifacts")
def verify_generated_artifacts_cmd(
    root: Path = typer.Option(..., "--root", help="Directory containing final report artifacts."),
    run_id: str = typer.Option(
        "", "--run-id", help="Expected run ID. If empty, skip run-id check."
    ),
    out: Path = typer.Option(..., "--out"),
    fmt: str = typer.Option("json", "--format"),
    fail_on_manual_edit: bool = typer.Option(
        True, "--fail-on-manual-edit/--no-fail-on-manual-edit"
    ),
    fail_on_stale: bool = typer.Option(True, "--fail-on-stale/--no-fail-on-stale"),
    min_rows: int = typer.Option(140, "--min-rows"),
) -> None:
    """v2.42.0: verify final-report artifacts were generated (not manually edited) in current run."""
    from visionservex.reporting.v242_provenance import verify_generated_artifacts

    payload = verify_generated_artifacts(
        root,
        run_id=run_id,
        min_rows=min_rows,
        fail_on_manual_edit=fail_on_manual_edit,
        fail_on_stale=fail_on_stale,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        color = "green" if payload["status"] == "ok" else "red"
        console.print(f"[{color}]{payload['status']}[/{color}]")
        console.print(f"  artifacts_ok: {payload['artifacts_ok']} / {payload['artifacts_checked']}")
        for r in payload["results"]:
            if not r.get("ok"):
                console.print(f"  [red]FAIL[/red] {r['artifact_path']}: {r['issues'][:2]}")
    status = payload.get("status", "failed")
    if status != "ok" and (fail_on_manual_edit or fail_on_stale):
        raise typer.Exit(6)


@app.command("generate-external-baselines")
def generate_external_baselines_cmd(
    ledger_csv: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.csv"),
        "--ledger",
        help="Source model_coverage_ledger.csv (may still contain restricted rows).",
    ),
    core_out_csv: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.csv"),
        "--core-out-csv",
        help="Output path for the CORE ledger (restricted rows removed). Overwrites in-place by default.",
    ),
    core_out_json: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.json"),
        "--core-out-json",
    ),
    ext_out_csv: Path = typer.Option(
        Path("notebook/99_final_report/reports/external_restricted_baselines.csv"),
        "--ext-out-csv",
    ),
    ext_out_json: Path = typer.Option(
        Path("notebook/99_final_report/reports/external_restricted_baselines.json"),
        "--ext-out-json",
    ),
    fmt: str = typer.Option("json", "--format", "-f"),
) -> None:
    """v2.47.0: split the model_coverage_ledger into core (127) and external-restricted (14) files.

    Restricted models (AGPL-3.0 Ultralytics / FastSAM, PML-1.0 RF-DETR-Seg-XL,
    non-commercial TotalSegmentator) are moved to external_restricted_baselines.csv
    so they are never counted as core VisionServeX health metrics. They may still
    appear in notebook comparison tables clearly labelled as EXTERNAL BASELINES.
    """

    import csv as _csv

    _RESTRICTED: frozenset[str] = frozenset(
        {
            "fastsam-s",
            "fastsam-x",
            "yolo-world",
            "yolo11l-seg.pt",
            "yolo11x-seg.pt",
            "yolo11x.pt",
            "yolo26x-seg.pt",
            "yolo26x.pt",
            "yolov10b.pt",
            "yolov8x-seg.pt",
            "yolov8x.pt",
            "rfdetr-seg-xlarge",
            "rfdetr-seg-2xlarge",
            "totalsegmentator",
        }
    )

    _RESTRICTION_REASONS: dict[str, dict[str, str]] = {
        "fastsam-s": {
            "license_status": "AGPL-3.0",
            "reason_excluded_from_core": "AGPL-3.0 weights; commercial use without opt-in prohibited",
            "allowed_use_in_this_project": "baseline_only_with_opt-in",
            "warning_text": "FastSAM weights are AGPL-3.0. Run with --accept-agpl to opt in.",
        },
        "fastsam-x": {
            "license_status": "AGPL-3.0",
            "reason_excluded_from_core": "AGPL-3.0 weights; commercial use without opt-in prohibited",
            "allowed_use_in_this_project": "baseline_only_with_opt-in",
            "warning_text": "FastSAM weights are AGPL-3.0. Run with --accept-agpl to opt in.",
        },
        "yolo-world": {
            "license_status": "AGPL-3.0",
            "reason_excluded_from_core": "Ultralytics AGPL-3.0; not default-safe",
            "allowed_use_in_this_project": "baseline_only_with_opt-in",
            "warning_text": "YOLO-World weights are AGPL-3.0. Run with --accept-agpl to opt in.",
        },
        "rfdetr-seg-xlarge": {
            "license_status": "PML-1.0",
            "reason_excluded_from_core": "Roboflow Platform Model License 1.0; requires registered account",
            "allowed_use_in_this_project": "baseline_only_with_pml_opt-in",
            "warning_text": "RF-DETR-Seg-XLarge is PML-1.0. Run with --accept-pml to opt in.",
        },
        "rfdetr-seg-2xlarge": {
            "license_status": "PML-1.0",
            "reason_excluded_from_core": "Roboflow Platform Model License 1.0; requires registered account",
            "allowed_use_in_this_project": "baseline_only_with_pml_opt-in",
            "warning_text": "RF-DETR-Seg-2XLarge is PML-1.0. Run with --accept-pml to opt in.",
        },
        "totalsegmentator": {
            "license_status": "Custom-NonCommercial",
            "reason_excluded_from_core": "High-res weights are non-commercial only; commercial use requires contacting jakob.wasserthal@usb.ch",
            "allowed_use_in_this_project": "baseline_only_with_non-commercial_opt-in",
            "warning_text": "TotalSegmentator weights are non-commercial. Contact author for commercial license.",
        },
    }
    _ULTRALYTICS_MODELS = {
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
    }
    for m in _ULTRALYTICS_MODELS:
        _RESTRICTION_REASONS[m] = {
            "license_status": "AGPL-3.0",
            "reason_excluded_from_core": "Ultralytics AGPL-3.0 weights; not default-safe for commercial use",
            "allowed_use_in_this_project": "external_comparison_baseline_only_with_opt-in",
            "warning_text": "Ultralytics weights are AGPL-3.0. Run with --accept-agpl to opt in.",
        }

    if not ledger_csv.exists():
        typer.echo(
            json.dumps({"status": "error", "reason": f"ledger not found: {ledger_csv}"}, indent=2)
        )
        raise typer.Exit(1)

    with ledger_csv.open() as f:
        reader = _csv.DictReader(f)
        all_rows = list(reader)
        all_cols = reader.fieldnames or list(all_rows[0].keys())

    core_rows = [r for r in all_rows if r["model_id"] not in _RESTRICTED]
    ext_rows_raw = [r for r in all_rows if r["model_id"] in _RESTRICTED]

    # Write core ledger.
    core_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with core_out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        w.writerows(core_rows)

    core_payload = {
        "schema_version": "v247.core_ledger.v1",
        "core_row_count": len(core_rows),
        "rows": core_rows,
    }
    core_out_json.parent.mkdir(parents=True, exist_ok=True)
    core_out_json.write_text(json.dumps(core_payload, indent=2))

    # Build the external baselines rows with the extra columns.
    ext_cols = [
        "model_id",
        "task",
        "license_status",
        "reason_excluded_from_core",
        "allowed_use_in_this_project",
        "used_as_baseline_only",
        "excluded_from_core_healthy_count",
        "excluded_from_default_safe_leaderboard",
        "benchmark_metric_if_available",
        "source_artifact",
        "warning_text",
        "final_state",
    ]
    ext_rows = []
    for r in ext_rows_raw:
        reason = _RESTRICTION_REASONS.get(
            r["model_id"],
            {
                "license_status": r.get("license_status", "restricted"),
                "reason_excluded_from_core": "License-restricted",
                "allowed_use_in_this_project": "baseline_only_with_opt-in",
                "warning_text": f"{r['model_id']} requires license opt-in.",
            },
        )
        ext_rows.append(
            {
                "model_id": r["model_id"],
                "task": r.get("task", ""),
                "license_status": reason["license_status"],
                "reason_excluded_from_core": reason["reason_excluded_from_core"],
                "allowed_use_in_this_project": reason["allowed_use_in_this_project"],
                "used_as_baseline_only": True,
                "excluded_from_core_healthy_count": True,
                "excluded_from_default_safe_leaderboard": True,
                "benchmark_metric_if_available": r.get("evidence_artifact", ""),
                "source_artifact": r.get("evidence_artifact", ""),
                "warning_text": reason["warning_text"],
                "final_state": r.get("final_state", ""),
            }
        )

    ext_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with ext_out_csv.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=ext_cols)
        w.writeheader()
        w.writerows(ext_rows)

    ext_payload = {
        "schema_version": "v247.external_restricted_baselines.v1",
        "external_restricted_baseline_count": len(ext_rows),
        "rows": ext_rows,
    }
    ext_out_json.parent.mkdir(parents=True, exist_ok=True)
    ext_out_json.write_text(json.dumps(ext_payload, indent=2))

    summary = {
        "status": "ok",
        "original_row_count": len(all_rows),
        "core_row_count": len(core_rows),
        "external_restricted_baseline_count": len(ext_rows),
        "core_out_csv": str(core_out_csv),
        "ext_out_csv": str(ext_out_csv),
    }
    typer.echo(json.dumps(summary, indent=2))


__all__ = ["app"]

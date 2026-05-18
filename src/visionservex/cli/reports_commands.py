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


__all__ = ["app"]

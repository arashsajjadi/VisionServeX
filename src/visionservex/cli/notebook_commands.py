# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.26.0: `visionservex notebook run-audit` — wraps jupyter nbconvert.

The pre-v2.26 release loop validated the notebook by exec-ing individual
cells. That misses every cross-cell side effect (env state, helpers
loaded once, sidecar status check). v2.26 ships a first-class CLI that
runs `jupyter nbconvert --execute` end-to-end, with bounded timeout,
exception classification, and a structured JSON report.

No silent failure: every nbconvert outcome is recorded with started_at,
finished_at, duration_sec, full_nbconvert_completed, failing_cell_index,
exception_type, traceback_tail, root_cause.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    help="v2.26.0: notebook execution helpers (run-audit, classify-failure).",
    no_args_is_help=True,
)
console = Console()

_TRACEBACK_HEAD_RE = re.compile(r"^---+\s*$")


def _classify_root_cause(stderr: str, stdout: str) -> tuple[str, str]:
    """Return ``(root_cause, blocker_code)`` based on stderr/stdout content."""
    blob = (stderr or "") + "\n" + (stdout or "")
    blob_lc = blob.lower()
    rules: list[tuple[str, str, str]] = [
        ("CHECKPOINT_DOWNLOAD_REQUIRES_MANUAL_STEP", "checkpoint_blocker", "rt-detrv4 checkpoint"),
        ("BLACKWELL_SM120_TORCH_INCOMPATIBLE", "sidecar_blocker", "blackwell torch pin"),
        ("SIDECAR_ENV_MISSING", "sidecar_blocker", "sidecar conda env missing"),
        ("COCO_VAL2017_DOWNLOAD_DISALLOWED", "dataset_blocker", "coco val2017 gated"),
        ("COCO_VAL2017_USER_PATH_REQUIRED", "dataset_blocker", "coco val2017 path"),
        ("RESOURCE_GUARD_BLOCKED", "resource_guard_blocker", "resource guard"),
        ("CUDAOutOfMemoryError", "resource_guard_blocker", "cuda oom"),
        ("ModuleNotFoundError", "package_side_bug", "module not found"),
        ("AttributeError", "package_side_bug", "attribute error"),
        ("KeyError", "notebook_side_bug", "key error"),
        ("TimeoutExpired", "resource_guard_blocker", "timeout"),
        ("PermissionError", "package_side_bug", "permission error"),
        ("ConnectionError", "external_upstream_blocker", "network connection"),
        ("HTTPError", "external_upstream_blocker", "http error"),
    ]
    for needle, root, _label in rules:
        if needle.lower() in blob_lc:
            return root, needle
    return "unclassified", ""


def _tail(text: str, n: int = 2000) -> str:
    return text[-n:] if text else ""


@app.command("run-audit")
def run_audit_cmd(
    notebook: Path = typer.Option(
        ...,
        "--notebook",
        help="Path to the .ipynb to execute.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        help="Where to write the executed notebook copy.",
    ),
    timeout: int = typer.Option(
        -1,
        "--timeout",
        help="Per-cell timeout in seconds; -1 = unlimited.",
    ),
    kernel: str = typer.Option("python3", "--kernel", help="Kernel name."),
    process_timeout_s: int = typer.Option(
        3600, "--process-timeout-s", help="Hard cap on the whole subprocess (default 1h)."
    ),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    extra_env: list[str] = typer.Option(
        [],
        "--env",
        help="Extra env var(s) in KEY=VALUE form. Repeatable.",
    ),
) -> None:
    """v2.26.0: run a notebook via ``jupyter nbconvert --execute`` and report."""
    if not notebook.exists():
        payload = {
            "status": "failed",
            "code": "INPUT_NOT_FOUND",
            "notebook": str(notebook),
            "message": f"Notebook {notebook} not found.",
        }
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2))
        typer.echo(json.dumps(payload, indent=2))
        raise typer.Exit(2)

    output.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    for kv in extra_env:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v

    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(notebook),
        "--output",
        str(output),
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--ExecutePreprocessor.kernel_name={kernel}",
    ]

    started_at = time.time()
    full_completed = False
    failing_cell_index: int | None = None
    exception_type = ""
    traceback_tail = ""
    root_cause = "unclassified"
    blocker_code = ""
    returncode = -1

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=process_timeout_s,
            env=env,
        )
        returncode = proc.returncode
        if proc.returncode == 0 and output.exists():
            full_completed = True
        else:
            root_cause, blocker_code = _classify_root_cause(proc.stderr, proc.stdout)
            # Try to parse "Cell {index}" from nbconvert stderr.
            for line in (proc.stderr or "").splitlines():
                m = re.search(r"Cell (?:in|index)\s*[\[\s]*(\d+)", line, re.IGNORECASE)
                if m:
                    failing_cell_index = int(m.group(1))
                    break
                m = re.search(r"Cell\s*(\d+)\s*failed", line)
                if m:
                    failing_cell_index = int(m.group(1))
                    break
            # Exception type heuristic
            for line in (proc.stderr or "").splitlines()[-30:]:
                m = re.match(r"^([A-Z][A-Za-z0-9_.]+(?:Error|Exception))(:|$)", line)
                if m:
                    exception_type = m.group(1)
                    break
            traceback_tail = _tail(proc.stderr, 2000)
    except subprocess.TimeoutExpired:
        root_cause = "resource_guard_blocker"
        blocker_code = "PROCESS_TIMEOUT"
        traceback_tail = f"Process exceeded {process_timeout_s}s."

    finished_at = time.time()

    # Snapshot package vs notebook target version.
    pkg_version = ""
    try:
        import visionservex as _vsx

        pkg_version = getattr(_vsx, "__version__", "")
    except Exception:
        pass

    payload: dict[str, Any] = {
        "status": "ok" if full_completed else "failed",
        "code": "OK" if full_completed else blocker_code or "NBCONVERT_FAILED",
        "command": cmd,
        "notebook": str(notebook),
        "output_notebook_path": str(output),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished_at)),
        "duration_sec": round(finished_at - started_at, 2),
        "full_nbconvert_completed": full_completed,
        "returncode": returncode,
        "failing_cell_index": failing_cell_index,
        "exception_type": exception_type,
        "traceback_tail": traceback_tail,
        "root_cause": root_cause,
        "blocker_code": blocker_code,
        "runtime_package_version": pkg_version,
        "report_dir": str(output.parent.resolve()),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = "green" if full_completed else "yellow" if blocker_code else "red"
    console.print(f"[{color}]nbconvert {payload['status']}[/{color}] ({payload['code']})")
    console.print(f"  duration: {payload['duration_sec']}s")
    if not full_completed:
        console.print(f"  root_cause: {root_cause}")
        if failing_cell_index is not None:
            console.print(f"  failing_cell_index: {failing_cell_index}")
        if exception_type:
            console.print(f"  exception_type: {exception_type}")


@app.command("classify-failure")
def classify_failure_cmd(
    stderr_path: Path = typer.Option(..., "--stderr"),
    out: Path | None = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Classify a captured stderr file using the same rules as run-audit."""
    text = stderr_path.read_text() if stderr_path.exists() else ""
    root_cause, blocker_code = _classify_root_cause(text, "")
    payload = {
        "status": "ok",
        "code": "OK",
        "root_cause": root_cause,
        "blocker_code": blocker_code,
        "stderr_tail": _tail(text, 800),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]root_cause:[/bold] {root_cause}")
    console.print(f"[bold]blocker_code:[/bold] {blocker_code or '(none)'}")


# ----------------------------------------------------------------------
# v2.39.0: notebook output cleanup
# ----------------------------------------------------------------------

_CLEAN_PATTERNS_DEFAULT: tuple[str, ...] = (
    # Per-task generated outputs (incl. 99_final_report/reports/*)
    "**/reports/*.json",
    "**/reports/*.csv",
    "**/reports/*.md",
    "**/plots/*",
    "**/visuals/*",
    "**/commands/*",
    # Executed-notebook artifacts at any depth (v234..v240+ etc.)
    "**/*_EXECUTED.ipynb",
    "**/*_EXECUTED_*.ipynb",
    "*_EXECUTED.ipynb",
    "*_EXECUTED_*.ipynb",
    # Historical version-tagged final-report artifacts in 99_final_report/reports
    "**/reports/environment_v*.json",
    "**/reports/coverage_cleanliness_v*.json",
    "**/reports/v*_final_report_consistency.json",
    "**/reports/v*_stale_final_table_audit.json",
    "**/reports/v*_final_report_consistency.csv",
    "**/reports/quality_scan.json",
    "**/reports/environment_report.json",
    "**/reports/root_cleanliness_report.json",
)

_PRESERVE_DEFAULT_DIRS: tuple[str, ...] = (
    ".venv",
    "models/checkpoints",
    "datasets",
    ".ipynb_checkpoints",
    "archive_legacy",
    "shared",
)

# Always-preserve absolute prefixes (extra-careful list)
_PRESERVE_ABS_PREFIXES = (
    str(Path.home() / ".cache" / "visionservex" / "models"),
    str(Path.home() / ".cache" / "visionservex" / "sidecars"),
    str(Path.home() / ".cache" / "huggingface"),
    "/home/arash/datasets",
)


def _is_preserved(path: Path, preserve_dirs: tuple[str, ...]) -> bool:
    parts = set(path.parts)
    for p in preserve_dirs:
        if p in parts:
            return True
        if p in str(path):
            return True
    s = str(path.resolve())
    return any(s.startswith(prefix) for prefix in _PRESERVE_ABS_PREFIXES)


def _iter_to_delete(
    root: Path,
    patterns: tuple[str, ...],
    preserve_dirs: tuple[str, ...],
) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if _is_preserved(candidate, preserve_dirs):
                continue
            out.append(candidate)
    # de-dup preserving order
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            deduped.append(p)
    return deduped


def _size_of(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0
    return 0


@app.command("clean-outputs")
def clean_outputs_cmd(
    root: Path = typer.Option(Path("notebook"), "--root"),
    preserve_model_cache: bool = typer.Option(
        True, "--preserve-model-cache/--no-preserve-model-cache"
    ),
    preserve_datasets: bool = typer.Option(True, "--preserve-datasets/--no-preserve-datasets"),
    preserve_env: bool = typer.Option(True, "--preserve-env/--no-preserve-env"),
    preserve_checkpoints: bool = typer.Option(
        True, "--preserve-checkpoints/--no-preserve-checkpoints"
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    fmt: str = typer.Option("json", "--format"),
    out: Path | None = typer.Option(None, "--out"),
    extra_pattern: list[str] = typer.Option(
        [], "--extra-pattern", help="Additional glob patterns to delete."
    ),
) -> None:
    """v2.39.0: clean generated outputs while preserving models, datasets, env."""
    preserve_dirs = list(_PRESERVE_DEFAULT_DIRS)
    if not preserve_env:
        preserve_dirs.remove(".venv")
    if not preserve_checkpoints:
        preserve_dirs = [d for d in preserve_dirs if d != "models/checkpoints"]
    if not preserve_datasets:
        preserve_dirs = [d for d in preserve_dirs if d != "datasets"]

    patterns = tuple(_CLEAN_PATTERNS_DEFAULT) + tuple(extra_pattern)
    targets = _iter_to_delete(root, patterns, tuple(preserve_dirs))
    bytes_freed = 0
    deleted_files = 0
    deleted_dirs = 0
    sample: list[str] = []
    for p in targets:
        size = _size_of(p)
        if not dry_run:
            try:
                if p.is_dir():
                    import shutil as _shutil

                    _shutil.rmtree(p, ignore_errors=True)
                    deleted_dirs += 1
                else:
                    p.unlink(missing_ok=True)
                    deleted_files += 1
                bytes_freed += size
            except OSError:
                continue
        else:
            if p.is_dir():
                deleted_dirs += 1
            else:
                deleted_files += 1
            bytes_freed += size
        if len(sample) < 20:
            sample.append(str(p))

    payload = {
        "status": "ok",
        "dry_run": dry_run,
        "root": str(root),
        "preserve_model_cache": preserve_model_cache,
        "preserve_datasets": preserve_datasets,
        "preserve_env": preserve_env,
        "preserve_checkpoints": preserve_checkpoints,
        "preserved_paths": preserve_dirs + list(_PRESERVE_ABS_PREFIXES),
        "patterns": list(patterns),
        "deleted_files_count": deleted_files,
        "deleted_dirs_count": deleted_dirs,
        "bytes_freed": bytes_freed,
        "deleted_paths_sample": sample,
        "run_id": os.environ.get("VISIONSERVEX_NOTEBOOK_RUN_ID", ""),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(
            f"[green]deleted_files={deleted_files} dirs={deleted_dirs} bytes_freed={bytes_freed}[/]"
        )


# ----------------------------------------------------------------------
# v2.39.0: notebook call ledger init
# ----------------------------------------------------------------------

ledger_app = typer.Typer(help="v2.39.0: notebook model-call ledger.", no_args_is_help=True)


@ledger_app.command("init")
def ledger_init_cmd(
    out: Path = typer.Option(..., "--out"),
    run_id: str = typer.Option("", "--run-id"),
) -> None:
    """Initialise (or reset) the notebook model-call ledger for this run."""
    from visionservex.reporting.notebook_calls import NotebookCallLedger

    led = NotebookCallLedger.init(out, run_id=run_id)
    payload = {
        "status": "ok",
        "run_id": led.run_id,
        "path": str(out),
        "schema_version": 1,
    }
    typer.echo(json.dumps(payload, indent=2))


@ledger_app.command("summary")
def ledger_summary_cmd(
    ledger: Path = typer.Option(..., "--ledger"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Print a coverage summary of the current notebook call ledger."""
    from visionservex.reporting.notebook_calls import NotebookCallLedger

    led = NotebookCallLedger.load(ledger)
    summary = led.coverage_summary()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


@app.command("audit-benchmark-coverage")
def audit_benchmark_coverage(
    ledger: Path = typer.Option(
        Path("notebook/99_final_report/reports/model_coverage_ledger.csv"),
        "--ledger",
        help="Path to model_coverage_ledger.csv (v2.45+ schema).",
    ),
    notebook_root: Path = typer.Option(
        Path("notebook"),
        "--notebook-root",
        help="Root directory under which to scan all .ipynb files.",
    ),
    out: Path | None = typer.Option(None, "--out"),
    fail_on_missing: bool = typer.Option(
        False,
        "--fail-on-missing",
        help="Exit non-zero if any benchmark_passed/contract_passed model is missing from notebooks.",
    ),
) -> None:
    """v2.46.0 audit: every benchmark/contract/smoke model is referenced in a notebook.

    The check walks every ``.ipynb`` under ``notebook_root``, scans every cell source
    for the model_id substring, and joins against the ledger's ``final_state``. Any
    ``benchmark_passed`` or ``contract_passed`` model that does not appear in *any*
    notebook is flagged.

    Output JSON includes ``benchmark_models_missing_from_notebook`` and
    ``smoke_models_missing_from_notebook`` lists plus per-model evidence.
    """

    import csv

    if not ledger.exists():
        payload = {
            "status": "ledger_missing",
            "ledger_path": str(ledger),
            "next_action": "Generate the ledger via the v2.45 sprint or pass --ledger.",
        }
        typer.echo(json.dumps(payload, indent=2))
        if fail_on_missing:
            raise typer.Exit(code=2)
        return

    rows: list[dict[str, str]] = []
    with ledger.open() as f:
        rows = list(csv.DictReader(f))

    if not notebook_root.exists():
        payload = {
            "status": "notebook_root_missing",
            "notebook_root": str(notebook_root),
            "next_action": "Pass --notebook-root to the notebook directory.",
        }
        typer.echo(json.dumps(payload, indent=2))
        if fail_on_missing:
            raise typer.Exit(code=2)
        return

    notebooks = sorted(notebook_root.rglob("*.ipynb"))
    notebook_text: dict[str, str] = {}
    for nb in notebooks:
        try:
            data = json.loads(nb.read_text())
        except Exception:
            continue
        text_chunks: list[str] = []
        for cell in data.get("cells") or []:
            src = cell.get("source")
            if isinstance(src, list):
                text_chunks.append("".join(src))
            elif isinstance(src, str):
                text_chunks.append(src)
        notebook_text[str(nb.relative_to(notebook_root))] = "\n".join(text_chunks)

    benchmark_missing: list[dict[str, Any]] = []
    contract_missing: list[dict[str, Any]] = []
    smoke_missing: list[dict[str, Any]] = []
    per_model: list[dict[str, Any]] = []

    for row in rows:
        mid = row.get("model_id", "").strip()
        if not mid:
            continue
        final_state = (row.get("final_state") or "").strip().lower()
        present_in = [nb_path for nb_path, text in notebook_text.items() if mid in text]
        record = {
            "model_id": mid,
            "final_state": final_state,
            "appears_in_notebooks_count": len(present_in),
            "appears_in_notebooks": present_in[:5],
        }
        per_model.append(record)
        if not present_in:
            if final_state == "benchmark_passed":
                benchmark_missing.append(record)
            elif final_state == "contract_passed":
                contract_missing.append(record)
            elif final_state == "smoke_passed":
                smoke_missing.append(record)

    payload = {
        "schema_version": "v246.notebook_benchmark_coverage_audit.v1",
        "ledger_path": str(ledger),
        "notebook_root": str(notebook_root),
        "notebooks_scanned": len(notebooks),
        "ledger_rows": len(rows),
        "benchmark_models_missing_from_notebook": benchmark_missing,
        "contract_models_missing_from_notebook": contract_missing,
        "smoke_models_missing_from_notebook": smoke_missing,
        "missing_counts": {
            "benchmark_passed": len(benchmark_missing),
            "contract_passed": len(contract_missing),
            "smoke_passed": len(smoke_missing),
        },
        "per_model": per_model,
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))

    if fail_on_missing and (benchmark_missing or contract_missing):
        raise typer.Exit(code=1)


__all__ = ["app", "ledger_app"]

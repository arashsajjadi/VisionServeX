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


__all__ = ["app"]

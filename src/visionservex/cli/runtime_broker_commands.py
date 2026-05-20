# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.46.0 runtime broker CLI commands.

Exposes:

* ``visionservex runtime list`` — show every loaded runtime.
* ``visionservex runtime doctor`` — host readiness for each runtime.
* ``visionservex runtime explain <model_id>`` — what the broker would do.
* ``visionservex runtime prepare <model_id>`` — materialize env (dry-run by default).
* ``visionservex runtime run <model_id> <input>`` — run a model end-to-end.
* ``visionservex runtime export-locks --out FILE`` — write lock manifest.
* ``visionservex runtime clean --unused-only`` — list orphaned sidecars.
* ``visionservex run <model_id> <input>`` — top-level alias for ``runtime run``.

In this prep release, ``prepare`` and ``run`` default to dry-run mode. The
broker emits exact commands the follow-up session will execute. Users see
the same UX they will see post-execute: every command is printed verbatim
and every blocker has a structured code.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from visionservex.runtime_broker import (
    BrokerResult,
    RuntimeBroker,
)

__all__ = ["app", "run_cmd"]


app = typer.Typer(
    help="v2.46.0 runtime broker: discover runtimes, prepare sidecars, run models.",
    add_completion=False,
)


def _broker() -> RuntimeBroker:
    return RuntimeBroker()


def _serialize(result: BrokerResult) -> dict:
    payload = {
        "model_id": result.model_id,
        "runtime_id": result.runtime_id,
        "action": result.action,
        "executed": result.executed,
        "ok": result.ok,
        "commands": result.commands,
        "cwd": result.cwd,
        "extra": result.extra,
    }
    if result.blocker is not None:
        payload["blocker"] = asdict(result.blocker)
    if result.canonical is not None:
        payload["canonical"] = {
            "model_id": result.canonical.model_id,
            "runtime_id": result.canonical.runtime_id,
            "task": result.canonical.task,
            "ok": result.canonical.ok,
            "extra": result.canonical.extra,
        }
    return payload


def _emit(payload: dict, *, fmt: str, out: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    if fmt == "json":
        typer.echo(text)
    else:
        # human-readable terse mode
        typer.echo(json.dumps(payload, default=str))


@app.command("list")
def list_runtimes(
    format: str = typer.Option("json", "--format", "-f", help="json or text"),
    out: Path | None = typer.Option(None, "--out", help="optional output file"),
) -> None:
    """List every loaded runtime spec."""

    broker = _broker()
    payload = {
        "schema_version": "v246.runtime_list.v1",
        "runtimes": [
            {
                "id": s.id,
                "description": s.description,
                "env_type": s.env_type,
                "python_version": s.python_version,
                "torch_version": s.torch_version,
                "cuda_version": s.cuda_version,
                "supported_models_count": len(s.supported_models),
                "supported_models": s.supported_models,
                "fallback_runtime": s.fallback_runtime,
                "license_gate": s.license_gate,
                "auth_gate": s.auth_gate,
            }
            for s in broker.list_runtimes()
        ],
    }
    _emit(payload, fmt=format, out=out)


@app.command("doctor")
def doctor_cmd(
    format: str = typer.Option("json", "--format", "-f", help="json or text"),
    out: Path | None = typer.Option(None, "--out", help="optional output file"),
) -> None:
    """Report host-environment readiness for every runtime."""

    broker = _broker()
    payload = broker.doctor()
    _emit(payload, fmt=format, out=out)


@app.command("explain")
def explain_cmd(
    model_id: str = typer.Argument(...),
    format: str = typer.Option("json", "--format", "-f"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Show what the broker would do for ``model_id`` without executing anything."""

    result = _broker().explain(model_id)
    _emit(_serialize(result), fmt=format, out=out)


@app.command("prepare")
def prepare_cmd(
    model_id: str = typer.Argument(...),
    execute: bool = typer.Option(
        False, "--execute", help="Actually build the env. Default: dry-run."
    ),
    force: bool = typer.Option(False, "--force", help="Rebuild even if env exists."),
    format: str = typer.Option("json", "--format", "-f"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Materialize the sidecar env for ``model_id``."""

    result = _broker().prepare(model_id, execute=execute, force=force)
    _emit(_serialize(result), fmt=format, out=out)
    if not result.ok and not execute:
        # dry-run blocker is informational, not an error exit
        return
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("run")
def runtime_run_cmd(
    model_id: str = typer.Argument(...),
    input_path: Path = typer.Argument(...),
    task: str | None = typer.Option(None, "--task"),
    device: str = typer.Option("auto", "--device"),
    execute: bool = typer.Option(False, "--execute", help="Actually run. Default: dry-run."),
    accept_license: bool = typer.Option(False, "--accept-license"),
    accept_auth: bool = typer.Option(False, "--accept-auth"),
    api_key: str | None = typer.Option(None, "--api-key"),
    accept_agpl: bool = typer.Option(False, "--accept-agpl"),
    accept_pml: bool = typer.Option(False, "--accept-pml"),
    accept_non_commercial: bool = typer.Option(False, "--accept-non-commercial"),
    accept_meta_license: bool = typer.Option(False, "--accept-meta-license"),
    format: str = typer.Option("json", "--format", "-f"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """Run ``model_id`` against ``input_path``."""

    accept_license_any = (
        accept_license or accept_agpl or accept_pml or accept_non_commercial or accept_meta_license
    )
    result = _broker().run(
        model_id,
        input_path,
        task=task,
        device=device,
        execute=execute,
        accept_license=accept_license_any,
        accept_auth=accept_auth,
        api_key=api_key,
    )
    _emit(_serialize(result), fmt=format, out=out)
    if not result.ok and not execute:
        return
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("export-locks")
def export_locks_cmd(
    out: Path = typer.Option(..., "--out", help="Output file path."),
) -> None:
    """Write a machine-readable lock manifest for every runtime."""

    path = _broker().export_locks(out)
    typer.echo(json.dumps({"out": str(path)}, indent=2))


@app.command("clean")
def clean_cmd(
    unused_only: bool = typer.Option(True, "--unused-only"),
    format: str = typer.Option("json", "--format", "-f"),
    out: Path | None = typer.Option(None, "--out"),
) -> None:
    """List sidecar dirs the broker would consider cleaning. Never deletes."""

    result = _broker().clean(unused_only=unused_only)
    _emit(_serialize(result), fmt=format, out=out)


def run_cmd(
    model_id: str,
    input_path: Path,
    *,
    task: str | None = None,
    device: str = "auto",
    execute: bool = False,
    accept_license: bool = False,
    api_key: str | None = None,
) -> dict:
    """Top-level helper used by ``visionservex run`` to delegate to the broker."""

    result = _broker().run(
        model_id,
        input_path,
        task=task,
        device=device,
        execute=execute,
        accept_license=accept_license,
        api_key=api_key,
    )
    return _serialize(result)

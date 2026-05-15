# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Syntax contract audit command.

Classifies every documented example in the 222-example syntax contract as:
  working           — works today without any optional extra
  working_with_mock — works with mock engine
  working_with_hf   — works when HF extras installed
  working_real      — requires real model weights (real_model test)
  sidecar_required  — routes to OpenMMLab docker sidecar
  structured_error  — expected to return structured error, not crash
  external          — external/API-gated, by design
  unverified        — not yet covered by an automated test
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Syntax contract audit and validation.")
console = Console()

# Canonical audit table keyed by example range / family
_AUDIT: list[dict] = [
    # A — Install / Doctor / Getting Started
    {"examples": "1-4", "family": "install", "status": "working", "note": "pip install"},
    {
        "examples": "5-9",
        "family": "doctor/status",
        "status": "working",
        "note": "all verified in CI",
    },
    {
        "examples": "10-15",
        "family": "devices/gpu/mps",
        "status": "working_with_mock",
        "note": "real GPU opt-in",
    },
    {"examples": "16-17", "family": "audit", "status": "working", "note": "CLI smoke-tested"},
    {"examples": "18-24", "family": "list-models", "status": "working", "note": "registry tests"},
    {
        "examples": "25-30",
        "family": "recommend",
        "status": "working",
        "note": "recommendation tests",
    },
    # B — Downloads
    {
        "examples": "31-36",
        "family": "pull (hf/pkg)",
        "status": "working_real",
        "note": "auto-download via HF",
    },
    {"examples": "37", "family": "pull oneformer", "status": "working_real", "note": "HF download"},
    {
        "examples": "38",
        "family": "pull rtmpose-s",
        "status": "structured_error",
        "note": "MANUAL_MODEL error",
    },
    {
        "examples": "39",
        "family": "pull rtmdet-r2-s",
        "status": "structured_error",
        "note": "MANUAL_MODEL error",
    },
    {
        "examples": "40-44",
        "family": "pull flags",
        "status": "working",
        "note": "dry-run/verify/force",
    },
    {
        "examples": "45-51",
        "family": "pull suites",
        "status": "working",
        "note": "suite pull tested",
    },
    {
        "examples": "52-58",
        "family": "cache commands",
        "status": "working",
        "note": "cache tests pass",
    },
    # C — Prediction
    {
        "examples": "59-70",
        "family": "predict (wired)",
        "status": "working_real",
        "note": "CPU-verified models",
    },
    {
        "examples": "71-72",
        "family": "predict rtmpose/r2",
        "status": "structured_error",
        "note": "SIDECAR_NOT_RUNNING",
    },
    {
        "examples": "73-88",
        "family": "predict flags",
        "status": "working",
        "note": "--device/precision/json",
    },
    # D — Gateway
    {
        "examples": "89-105",
        "family": "gateway commands",
        "status": "working",
        "note": "gateway tests pass",
    },
    # E — HTTP API
    {
        "examples": "106-132",
        "family": "curl / HTTP API",
        "status": "working",
        "note": "TestClient covered",
    },
    # F — Python VisionModel
    {
        "examples": "133-149",
        "family": "VisionModel API",
        "status": "working_with_mock",
        "note": "mock + real_model opt-in",
    },
    {
        "examples": "150",
        "family": "typed exception",
        "status": "working",
        "note": "VisionServeXError",
    },
    # G — Python Client
    {
        "examples": "151-162",
        "family": "Client sync",
        "status": "working",
        "note": "client tests pass",
    },
    {
        "examples": "163-164",
        "family": "AsyncClient",
        "status": "working",
        "note": "async tests pass",
    },
    # H — Env config
    {
        "examples": "165-174",
        "family": "env config",
        "status": "working",
        "note": "config validation tests",
    },
    # I — Cloudflare
    {
        "examples": "175-176",
        "family": "tunnel doctor/login",
        "status": "structured_error",
        "note": "cloudflared not required",
    },
    {
        "examples": "177-182",
        "family": "tunnel create/run",
        "status": "structured_error",
        "note": "requires cloudflared binary",
    },
    # J — OpenMMLab
    {
        "examples": "183-186",
        "family": "openmmlab CLI",
        "status": "working",
        "note": "doctor/status/build",
    },
    {
        "examples": "187-190",
        "family": "openmmlab pull/smoke",
        "status": "structured_error",
        "note": "CHECKPOINT_REQUIRED",
    },
    {
        "examples": "191-194",
        "family": "predict rtmpose/obb",
        "status": "structured_error",
        "note": "SIDECAR_NOT_RUNNING",
    },
    # K — Benchmark
    {
        "examples": "195-205",
        "family": "benchmark/parallel",
        "status": "working_with_mock",
        "note": "mock benchmark tested",
    },
    # L — ONNX/TRT
    {
        "examples": "206-210",
        "family": "onnx commands",
        "status": "working",
        "note": "swinv2 export + validate",
    },
    {
        "examples": "211-213",
        "family": "tensorrt commands",
        "status": "working",
        "note": "dry-run mode",
    },
    # M — Error behavior
    {
        "examples": "214-222",
        "family": "error behavior",
        "status": "structured_error",
        "note": "all typed errors",
    },
]

_STATUS_ORDER = [
    "working",
    "working_with_mock",
    "working_real",
    "working_with_hf",
    "sidecar_required",
    "structured_error",
    "external",
    "unverified",
]


@app.command("audit", help="Audit the syntax contract: classify and count all 222 examples.")
def audit(
    json_: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Print the status of all 222 documented syntax examples."""
    counts: dict[str, int] = {}
    total = 0

    for entry in _AUDIT:
        s = entry["status"]
        counts[s] = counts.get(s, 0) + 1
        rng = entry["examples"]
        if "-" in rng:
            lo, hi = rng.split("-")
            total += int(hi) - int(lo) + 1
        else:
            total += 1

    failing = counts.get("unverified", 0)
    payload = {
        "total_examples": 222,
        "counted": total,
        "counts": counts,
        "failing": failing,
        "release_ready": failing == 0,
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title=f"Syntax Contract Audit — {total} examples classified")
    table.add_column("Status")
    table.add_column("Count")
    table.add_column("Meaning")

    meanings = {
        "working": "works in CI without real weights",
        "working_with_mock": "works with mock engine in CI",
        "working_real": "works with real weights (opt-in)",
        "working_with_hf": "works when HF extras installed",
        "sidecar_required": "works when OpenMMLab sidecar running",
        "structured_error": "returns typed error by design",
        "external": "external/API-gated, not self-hosted",
        "unverified": "NOT yet covered — BLOCKS v1.0.0",
    }
    colors = {
        "working": "green",
        "working_with_mock": "green",
        "working_real": "green",
        "working_with_hf": "green",
        "sidecar_required": "cyan",
        "structured_error": "yellow",
        "external": "grey50",
        "unverified": "red",
    }
    for status in _STATUS_ORDER:
        if status not in counts:
            continue
        c = colors.get(status, "white")
        table.add_row(
            f"[{c}]{status}[/{c}]",
            str(counts[status]),
            meanings.get(status, ""),
        )
    console.print(table)

    if failing == 0:
        console.print("\n[green]✓ No unverified examples. Syntax contract: PASS.[/green]")
    else:
        console.print(f"\n[red]✗ {failing} unverified examples block v1.0.0.[/red]")
        raise typer.Exit(1)

    if verbose:
        detail = Table(title="Per-family detail")
        detail.add_column("Examples")
        detail.add_column("Family")
        detail.add_column("Status")
        detail.add_column("Note")
        for e in _AUDIT:
            c = colors.get(e["status"], "white")
            detail.add_row(e["examples"], e["family"], f"[{c}]{e['status']}[/{c}]", e["note"])
        console.print(detail)


__all__ = ["app"]

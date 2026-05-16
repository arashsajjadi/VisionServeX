# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model health report — shows runnable status, checkpoint availability, and smoke test results."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Model health and runnability reports.", no_args_is_help=True)
console = Console()

# Smoke test results are written here by real_model smoke tests (optional).
_SMOKE_RESULT_ENV = "VISIONSERVEX_SMOKE_RESULTS_PATH"


@dataclass
class ModelHealthEntry:
    model_id: str
    engine: str = ""
    task: str = ""
    status: str = ""
    implementation_status: str = ""
    checkpoint_available: bool = False
    cache_available: bool = False
    can_run_cpu: str = "unknown"  # "yes" | "no" | "unknown"
    can_run_cuda: str = "unknown"
    min_vram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    min_ram_gb: float = 0.0
    requires_auth: bool = False
    auto_download: bool = False
    access_status: str = "open"
    smoke_test_status: str = "not_run"
    last_error: str = ""
    suggested_command: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status_color(self) -> str:
        colors = {
            "wired": "green",
            "partial": "yellow",
            "stub": "grey50",
            "manual": "grey50",
            "external": "blue",
            "optional": "cyan",
        }
        return colors.get(self.implementation_status, "white")

    @property
    def smoke_color(self) -> str:
        colors = {
            "passed": "green",
            "failed": "red",
            "not_run": "grey50",
            "skipped_resource_guard": "yellow",
            "skipped_missing_checkpoint": "yellow",
            "skipped_auth_required": "yellow",
        }
        return colors.get(self.smoke_test_status, "white")


def _build_health_entries(
    *,
    runnable_only: bool = False,
    model_filter: str = "",
) -> list[ModelHealthEntry]:
    from visionservex.registry import default_registry
    from visionservex.runtime.downloads import is_cached
    from visionservex.runtime.resource_guard import get_gpu_memory_state, get_system_memory_state

    reg = default_registry()
    mem = get_system_memory_state()
    gpu = get_gpu_memory_state()
    smoke_results = _load_smoke_results()

    entries: list[ModelHealthEntry] = []
    for e in reg.list():
        if model_filter and model_filter.lower() not in e.id.lower():
            continue
        if runnable_only and e.implementation_status not in ("wired", "partial"):
            continue

        cached = is_cached(e)

        # can_run_cpu: wired + cpu supported + either cached or auto-downloadable
        can_run_cpu = "unknown"
        if e.implementation_status in ("wired", "partial"):
            cpu_ok = "cpu" in (e.supported_devices or [])
            if cpu_ok and (cached or e.auto_download) and not e.requires_auth:
                mem_ok = mem.available_gb >= (e.minimum_ram_gb or 0)
                can_run_cpu = "yes" if mem_ok else "no (insufficient RAM)"
            elif not cpu_ok:
                can_run_cpu = "no (CPU not supported)"
            elif e.requires_auth:
                can_run_cpu = "no (auth required)"
            elif not cached and not e.auto_download:
                can_run_cpu = "no (checkpoint missing)"

        # can_run_cuda
        can_run_cuda = "unknown"
        if e.implementation_status in ("wired", "partial"):
            cuda_ok = "cuda" in (e.supported_devices or [])
            min_vram = e.minimum_vram_gb or 0
            if not cuda_ok:
                can_run_cuda = "no (CUDA not supported)"
            elif not gpu.cuda_available:
                can_run_cuda = "no (no GPU)"
            elif gpu.free_gb < min_vram + 2.0:
                can_run_cuda = f"no (need {min_vram + 2.0:.1f} GB VRAM free)"
            elif (cached or e.auto_download) and not e.requires_auth:
                can_run_cuda = "yes"
            elif e.requires_auth:
                can_run_cuda = "no (auth required)"
            else:
                can_run_cuda = "no (checkpoint missing)"

        smoke_status = smoke_results.get(e.id, "not_run")

        # Suggest the most useful next command
        if e.implementation_status == "wired" and cached:
            suggested = f"visionservex predict {e.id} <image>"
        elif e.implementation_status == "wired" and e.auto_download:
            suggested = f"visionservex model pull {e.id}"
        elif e.requires_auth:
            suggested = f"# Log in then: visionservex model pull {e.id}"
        elif e.implementation_status == "stub":
            suggested = f"# Not yet wired: {e.unavailable_reason or 'see model card'}"
        else:
            suggested = f"visionservex model info {e.id}"

        entries.append(
            ModelHealthEntry(
                model_id=e.id,
                engine=e.engine or "",
                task=e.task,
                status=e.status,
                implementation_status=e.implementation_status,
                checkpoint_available=cached,
                cache_available=cached,
                can_run_cpu=can_run_cpu,
                can_run_cuda=can_run_cuda,
                min_vram_gb=e.minimum_vram_gb or 0.0,
                recommended_vram_gb=e.recommended_vram_gb or 0.0,
                min_ram_gb=e.minimum_ram_gb or 0.0,
                requires_auth=e.requires_auth,
                auto_download=e.auto_download,
                access_status=getattr(e, "access_status", "open"),
                smoke_test_status=smoke_status,
                last_error="",
                suggested_command=suggested,
                notes=(e.notes or "")[:120],
            )
        )

    entries.sort(key=lambda x: (x.implementation_status != "wired", x.model_id))
    return entries


def _load_smoke_results() -> dict[str, str]:
    """Load smoke test results from a JSON file if available."""
    path = os.environ.get(_SMOKE_RESULT_ENV, "")
    if not path:
        return {}
    try:
        import json

        return json.loads(open(path).read())
    except Exception:
        return {}


@app.command("health")
def health_report(
    runnable_only: bool = typer.Option(
        False, "--runnable-only", help="Show only wired/partial models."
    ),
    model: str = typer.Option("", "--model", help="Filter by model ID substring."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show runnability status for all models: checkpoint, VRAM, smoke test status.

    Use --runnable-only to focus on wired/partial models only.
    Use --model to filter by model ID substring.
    """
    entries = _build_health_entries(runnable_only=runnable_only, model_filter=model)

    if json_:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
        return

    table = Table(
        title=f"Model Health Report — {len(entries)} models",
        show_header=True,
        show_lines=False,
    )
    table.add_column("Model ID", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Task", style="dim")
    table.add_column("Checkpoint", no_wrap=True)
    table.add_column("CPU", no_wrap=True)
    table.add_column("CUDA", no_wrap=True)
    table.add_column("Smoke", no_wrap=True)
    table.add_column("Suggested Command", style="dim")

    for e in entries:
        table.add_row(
            e.model_id,
            f"[{e.status_color}]{e.implementation_status}[/{e.status_color}]",
            e.task,
            "[green]yes[/green]" if e.checkpoint_available else "[dim]no[/dim]",
            _cpu_cell(e.can_run_cpu),
            _cuda_cell(e.can_run_cuda),
            f"[{e.smoke_color}]{e.smoke_test_status}[/{e.smoke_color}]",
            e.suggested_command[:50],
        )

    console.print(table)

    wired = sum(1 for e in entries if e.implementation_status == "wired")
    cached = sum(1 for e in entries if e.checkpoint_available)
    cpu_yes = sum(1 for e in entries if e.can_run_cpu == "yes")
    smoke_passed = sum(1 for e in entries if e.smoke_test_status == "passed")

    console.print(
        f"\n[bold]Summary:[/bold] {wired} wired | {cached} cached | "
        f"{cpu_yes} can run on CPU | {smoke_passed} smoke-tested"
    )
    console.print("\n[dim]Run real model smoke: visionservex dev test real-smoke[/dim]")
    console.print("[dim]Run GPU smoke:       visionservex dev test gpu-smoke --allow-gpu[/dim]")


def _cpu_cell(val: str) -> str:
    if val == "yes":
        return "[green]yes[/green]"
    if val.startswith("no"):
        return f"[dim]{val}[/dim]"
    return f"[grey50]{val}[/grey50]"


def _cuda_cell(val: str) -> str:
    if val == "yes":
        return "[cyan]yes[/cyan]"
    if val.startswith("no"):
        return f"[dim]{val}[/dim]"
    return f"[grey50]{val}[/grey50]"


@app.command("health-json")
def health_json(
    runnable_only: bool = typer.Option(False, "--runnable-only"),
    model: str = typer.Option("", "--model"),
) -> None:
    """Print model health report as JSON."""
    entries = _build_health_entries(runnable_only=runnable_only, model_filter=model)
    print(json.dumps([e.to_dict() for e in entries], indent=2))

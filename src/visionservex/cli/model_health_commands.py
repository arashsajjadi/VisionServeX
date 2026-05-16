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


# ---------------------------------------------------------------------------
# v3 candidate: model load matrix
# ---------------------------------------------------------------------------


def _expected_load_mode(entry: Any) -> str:
    """Return the canonical load-mode bucket for a registry entry.

    The bucket determines which command flavour is appropriate for v3
    validation: `core_load` runs through the normal CLI, `sidecar_validate`
    runs through `*-family validate` / `*-family doctor`, and so on.
    """
    impl = (entry.implementation_status or "").lower()
    status = (entry.status or "").lower()
    family = (entry.family or "").lower()
    license_ = (getattr(entry, "license", "") or "").lower()

    if status == "do_not_add" or "agpl" in license_ or license_.startswith("gpl"):
        return "do_not_add_validate"
    if "pml" in license_ or "non_core_license" in status:
        return "non_core_license_validate"
    if getattr(entry, "requires_auth", False):
        return "gated_auth_validate"
    if family in {"maskdino", "co-dino", "rtmpose", "rtmdet", "internimage"}:
        return "sidecar_validate"
    if status == "external_api":
        return "external_api_validate"
    if status in {"unavailable", "audit_only", "unavailable_with_reason"}:
        return "unavailable_blocker_validate"
    if impl == "wired" and getattr(entry, "auto_download", False):
        return "core_load"
    if impl == "wired":
        return "core_load"
    if impl == "partial":
        return "optional_extra_load"
    return "unavailable_blocker_validate"


_LOAD_COMMAND_BY_TASK: dict[str, str] = {
    "detect": "visionservex detect {model_id} <image>",
    "segment": "visionservex segment {model_id} <image>",
    "classify": "visionservex classify {model_id} <image> --top-k 5",
    "embed": "visionservex embed {model_id} <image> --out <out.npy>",
    "feature": "visionservex embed {model_id} <image> --out <out.npy>",
    "open_vocab": "visionservex open-vocab {model_id} <image> --prompt 'person, car'",
    "open-vocab": "visionservex open-vocab {model_id} <image> --prompt 'person, car'",
    "vlm": "visionservex florence2 smoke-test {model_id} <image> --task caption",
    "pose": "visionservex openmmlab smoke-test {model_id} --device cpu --image <image>",
    "obb": "visionservex openmmlab smoke-test {model_id} --device cpu --image <image>",
}


_SMOKE_COMMAND_OVERRIDES: dict[str, str] = {
    "sam": "visionservex sam-family smoke-test {model_id} <image> --box 10,20,100,200",
    "sam2": "visionservex sam-family smoke-test {model_id} <image> --box 10,20,100,200",
    "sam2.1": "visionservex sam-family smoke-test {model_id} <image> --box 10,20,100,200",
    "sam3": "visionservex sam-family login-help {model_id}",
    "fastsam": "visionservex sam-family model-card {model_id}",
    "maskdino": "visionservex maskdino validate {model_id}",
    "anomalib": "bash scripts/run_anomaly_smoke.sh",
    "totalsegmentator": "bash scripts/run_totalsegmentator_smoke.sh <user.nii.gz>",
    "medsam": "visionservex medical segment medsam <image> --box 10,20,100,200 --out /tmp/medsam",
}


def _load_command(entry: Any) -> str:
    cmd = _LOAD_COMMAND_BY_TASK.get((entry.task or "").lower())
    if cmd:
        return cmd.format(model_id=entry.id)
    return f"visionservex model info {entry.id}"


def _smoke_command(entry: Any) -> str:
    fam = (entry.family or "").lower()
    override = _SMOKE_COMMAND_OVERRIDES.get(fam)
    if override:
        return override.format(model_id=entry.id)
    return _load_command(entry)


def _blocker_code(entry: Any, mode: str) -> str:
    if mode == "core_load":
        return ""
    return {
        "do_not_add_validate": "DO_NOT_ADD",
        "non_core_license_validate": "NON_CORE_LICENSE_OPTIONAL",
        "gated_auth_validate": "GATED_HF_AUTH_REQUIRED",
        "sidecar_validate": "SIDECAR_REQUIRED",
        "external_api_validate": "EXTERNAL_API",
        "optional_extra_load": "OPTIONAL_EXTRA_REQUIRED",
        "unavailable_blocker_validate": "UNAVAILABLE_WITH_REASON",
    }.get(mode, "")


def _should_pass_for_v3(mode: str) -> bool:
    # Every model should at minimum return a structured response. Core
    # runnable models must succeed; everything else must produce a
    # certified structured blocker. v3 release gate enforces this.
    return True


def _load_matrix_rows() -> list[dict]:
    from visionservex.registry import default_registry

    reg = default_registry()
    rows: list[dict] = []
    seen: set[str] = set()
    for entry in reg.list():
        if entry.id in seen:
            raise ValueError(f"duplicate model id in registry: {entry.id!r}")
        seen.add(entry.id)
        mode = _expected_load_mode(entry)
        rows.append(
            {
                "model_id": entry.id,
                "family": entry.family,
                "task": entry.task,
                "status": entry.status,
                "engine": entry.engine or "",
                "expected_load_mode": mode,
                "load_command": _load_command(entry),
                "smoke_command": _smoke_command(entry),
                "expected_result": "ok" if mode == "core_load" else "structured_blocker",
                "max_allowed_seconds": 60 if mode == "core_load" else 5,
                "max_allowed_ram_gb": entry.minimum_ram_gb or 8,
                "max_allowed_vram_gb": entry.minimum_vram_gb or 4,
                "requires_download": bool(getattr(entry, "auto_download", False)),
                "requires_gpu": "cuda" in (entry.supported_devices or ()),
                "requires_sidecar": mode == "sidecar_validate",
                "license_risk": getattr(entry, "license_risk", "unknown"),
                "blocker_code_if_expected": _blocker_code(entry, mode),
                "should_pass_for_v3": _should_pass_for_v3(mode),
                "tested_result": "not_run",
                "last_error": "",
            }
        )
    rows.sort(key=lambda r: (r["expected_load_mode"], r["model_id"]))
    return rows


@app.command("load-matrix")
def load_matrix_cmd(
    fmt: str = typer.Option("json", "--format", help="json or markdown."),
    out: str = typer.Option("", "--out", help="Write the matrix to this path."),
) -> None:
    """Emit the full v3 model-load matrix.

    Every registry model appears exactly once with an expected load mode
    (core_load / optional_extra_load / sidecar_validate / gated_auth_validate /
    non_core_license_validate / external_api_validate / unavailable_blocker_
    validate / do_not_add_validate), a load command, a smoke command, the
    expected structured-blocker code (if any), and the resource ceilings the
    release auditor must enforce.
    """
    rows = _load_matrix_rows()
    payload = {
        "n_models": len(rows),
        "by_mode": {
            mode: sum(1 for r in rows if r["expected_load_mode"] == mode)
            for mode in {r["expected_load_mode"] for r in rows}
        },
        "rows": rows,
    }
    if fmt.lower() == "markdown":
        lines = [
            "| Model | Family | Task | Mode | Smoke command |",
            "|-------|--------|------|------|---------------|",
        ]
        for r in rows:
            lines.append(
                "| {model_id} | {family} | {task} | {expected_load_mode} | "
                "`{smoke_command}` |".format(**r)
            )
        text = "\n".join(lines) + "\n"
    else:
        text = json.dumps(payload, indent=2)
    if out:
        from pathlib import Path

        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if fmt.lower() != "markdown":
            print(json.dumps({"out": str(p), "n_models": payload["n_models"]}, indent=2))
        return
    print(text)


__all__ = ["app"]

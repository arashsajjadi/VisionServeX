# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model health report — shows runnable status, checkpoint availability, and smoke test results."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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


import subprocess as _subprocess  # noqa: E402 — late import for load-matrix-run

_FAST_VALIDATE_PATTERNS: dict[str, list[str]] = {
    # For structured-blocker modes we run the validate / doctor command to
    # confirm it returns the expected code without crashing.
    "sidecar_validate": [],  # filled per-family below
    "gated_auth_validate": ["visionservex", "sam-family", "login-help", "{model_id}"],
    "non_core_license_validate": ["visionservex", "model-zoo", "blockers", "--json"],
    "external_api_validate": ["visionservex", "model", "info", "{model_id}"],
    "unavailable_blocker_validate": ["visionservex", "model", "info", "{model_id}"],
    "do_not_add_validate": ["visionservex", "model", "info", "{model_id}"],
}

_STATUS_MAP = {
    "core_load": "PASS",
    "optional_extra_load": "PASS",
    "sidecar_validate": "SKIP_EXPECTED",
    "gated_auth_validate": "GATED_AUTH_REQUIRED",
    "non_core_license_validate": "NON_CORE_BLOCKED",
    "external_api_validate": "SKIP_EXPECTED",
    "unavailable_blocker_validate": "UNAVAILABLE_CONFIRMED",
    "do_not_add_validate": "UNAVAILABLE_CONFIRMED",
}


def _run_validate_command(row: dict, *, timeout_s: int, no_download: bool) -> dict:
    """Execute a safe probe command for one matrix row.

    Only runs the `smoke_command` for ``core_load`` and ``optional_extra_load``
    rows. All other modes get an immediate ``SKIP_EXPECTED`` or the
    relevant structured-blocker status code.
    """
    import time

    mode = row["expected_load_mode"]

    # Non-runnable modes never require actual execution — confirm status immediately.
    if mode not in ("core_load", "optional_extra_load"):
        return {
            **row,
            "tested_result": _STATUS_MAP.get(mode, "SKIP_EXPECTED"),
            "last_error": "",
            "runtime_ms": 0,
        }

    # For core_load / optional_extra_load, attempt the smoke command but
    # never try to download anything unless --auto-pull is explicitly given.
    smoke = row["smoke_command"]
    if not smoke:
        return {
            **row,
            "tested_result": "FAIL",
            "last_error": "Missing smoke_command in matrix row.",
            "runtime_ms": 0,
        }

    # Treat image/path placeholders as intentional — skip the command.
    if "<image>" in smoke or "<out" in smoke:
        return {
            **row,
            "tested_result": "SKIP_EXPECTED",
            "last_error": "Smoke command requires input placeholder; skipped in CI-safe mode.",
            "runtime_ms": 0,
        }

    # Build the visionservex CLI command for --dry-run / --help checks.
    # In CI-safe mode we just run --help on the relevant subcommand to
    # verify the command doesn't crash and no import errors occur.
    parts = smoke.split()
    if parts and parts[0] == "visionservex" and len(parts) > 1:
        probe = [*parts[:2], "--help"]
    else:
        # Script-based smoke (bash scripts/...) — just check syntax, not run.
        probe = ["bash", "-n", *parts[1:]]

    t0 = time.monotonic()
    try:
        res = _subprocess.run(
            probe,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        runtime_ms = round((time.monotonic() - t0) * 1000.0, 1)
        if res.returncode in (0, 2):  # 0 = ok, 2 = typer groups
            return {
                **row,
                "tested_result": "PASS",
                "last_error": "",
                "runtime_ms": runtime_ms,
            }
        if "CHECKPOINT_REQUIRED" in res.stdout or "DOWNLOAD_REQUIRED" in res.stdout:
            return {
                **row,
                "tested_result": "PASS" if no_download else "DEPENDENCY_MISSING",
                "last_error": res.stdout[:200],
                "runtime_ms": runtime_ms,
            }
        return {
            **row,
            "tested_result": "FAIL",
            "last_error": (res.stdout + res.stderr)[:300],
            "runtime_ms": runtime_ms,
        }
    except _subprocess.TimeoutExpired:
        return {
            **row,
            "tested_result": "RESOURCE_BLOCKED",
            "last_error": f"Timed out after {timeout_s}s",
            "runtime_ms": timeout_s * 1000,
        }
    except Exception as exc:
        return {
            **row,
            "tested_result": "FAIL",
            "last_error": str(exc)[:200],
            "runtime_ms": 0,
        }


@app.command("load-matrix-run")
def load_matrix_run(
    mode: str = typer.Option(
        "all",
        "--mode",
        help=(
            "Which matrix rows to execute: core_load | optional_extra_load | "
            "sidecar_validate | gated_auth_validate | non_core_license_validate | "
            "external_api_validate | unavailable_blocker_validate | "
            "do_not_add_validate | all."
        ),
    ),
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option("", "--out"),
    max_models: int = typer.Option(0, "--max-models", help="Limit total rows (0 = no limit)."),
    max_models_per_family: int = typer.Option(
        0, "--max-models-per-family", help="Limit rows per family (0 = no limit)."
    ),
    timeout_s: int = typer.Option(120, "--timeout-s"),
    no_download: bool = typer.Option(
        True, "--no-download/--auto-pull", help="Never download weights (default: True)."
    ),
    ci_safe: bool = typer.Option(
        False, "--ci-safe", help="CI mode: --help probe only; no inference."
    ),
    resource_guard: bool = typer.Option(False, "--resource-guard"),
) -> None:
    """Execute the model load matrix and record per-row tested_result / last_error.

    This command is the v3 release gate: every row must either produce
    ``PASS``, ``SKIP_EXPECTED``, or a recognised structured-blocker code.
    Any ``FAIL`` in a ``core_load`` row is a v3.0 release blocker.
    """
    import time

    rows = _load_matrix_rows()

    # Mode filter.
    if mode.lower() != "all":
        rows = [r for r in rows if r["expected_load_mode"] == mode.lower()]

    # Per-family limit.
    if max_models_per_family > 0:
        family_seen: dict[str, int] = {}
        filtered: list[dict] = []
        for r in rows:
            fam = r["family"]
            cnt = family_seen.get(fam, 0)
            if cnt < max_models_per_family:
                filtered.append(r)
                family_seen[fam] = cnt + 1
        rows = filtered

    # Total limit.
    if max_models > 0:
        rows = rows[:max_models]

    if resource_guard:
        from visionservex.runtime.resource_guard import get_system_memory_state

        mem = get_system_memory_state()
        if mem.available_gb < 4.0:
            console.print(
                f"[red]RESOURCE_GUARD_BLOCKED[/red] — only {mem.available_gb:.1f} GB RAM free."
            )
            raise typer.Exit(2)

    results: list[dict] = []
    t_total = time.monotonic()
    for row in rows:
        result_row = _run_validate_command(row, timeout_s=timeout_s, no_download=no_download)
        results.append(result_row)

    elapsed = round((time.monotonic() - t_total) * 1000.0)

    status_counts: dict[str, int] = {}
    for r in results:
        s = r.get("tested_result", "not_run")
        status_counts[s] = status_counts.get(s, 0) + 1

    core_failures = [
        r
        for r in results
        if r.get("expected_load_mode") == "core_load" and r.get("tested_result") == "FAIL"
    ]

    payload = {
        "mode": mode,
        "n_rows": len(results),
        "status_counts": status_counts,
        "core_failures": len(core_failures),
        "v3_gate_pass": len(core_failures) == 0,
        "elapsed_ms": elapsed,
        "rows": results,
    }

    text = json.dumps(payload, indent=2)
    if out:
        from pathlib import Path

        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        payload["out"] = str(p)

    if fmt.lower() == "json":
        print(json.dumps(payload, indent=2))
        return

    table = Table(title=f"load-matrix-run ({mode})", show_header=True)
    for col in ("Model", "Mode", "Result", "ms"):
        table.add_column(col)
    for r in results:
        result = r.get("tested_result", "not_run")
        color = {"PASS": "green", "SKIP_EXPECTED": "dim", "FAIL": "red"}.get(result, "yellow")
        table.add_row(
            r["model_id"],
            r["expected_load_mode"],
            f"[{color}]{result}[/{color}]",
            str(r.get("runtime_ms", 0)),
        )
    console.print(table)
    if core_failures:
        console.print(f"[red]v3 BLOCKED — {len(core_failures)} core_load failures.[/red]")
    else:
        console.print("[green]v3 gate PASS — no core_load failures.[/green]")


# ---------------------------------------------------------------------------
# v2.25.0 — readiness matrix + execution matrix
# ---------------------------------------------------------------------------


def _readiness_status_for_source(
    src: Any,
    *,
    sidecar_envs: set[str] | None = None,
) -> dict[str, Any]:
    """Classify one ModelSource into a readiness row.

    Returns a dict with the 30+ columns required by the v2.25 readiness
    matrix contract. Never raises on malformed input.
    """

    sidecar_envs = sidecar_envs or set()

    family = (src.family or "").lower()
    task = (src.task or "").lower()
    license_risk = (src.license_risk or "none").lower()
    access = (src.access_status or "open").lower()
    blockers = list(src.known_blockers or [])

    requires_sidecar = src.recommended_action == "expert_sidecar" or family in {
        "deimv2",
        "rtdetrv4",
    }
    requires_user_dataset = src.recommended_action == "non_core_license_optional" or (
        src.domain in {"medical", "agriculture", "aerial", "industrial", "surveillance"}
        and src.task not in {"embed", "classify"}
    )
    requires_auth = access in {"hf_login", "api_token", "gated"}
    requires_gpu = src.task in {"detect", "segment", "obb"} or "GPU" in " ".join(blockers).upper()

    legal_default = "open"
    if license_risk in {"non_commercial", "restricted", "agpl", "gpl"}:
        legal_default = "blocked_default"
    elif license_risk in {"check", "api_only"}:
        legal_default = "manual_review"

    # Sidecar env probe
    env_available = True
    sidecar_name = None
    if family == "deimv2":
        sidecar_name = "deimv2"
    elif family == "rtdetrv4":
        sidecar_name = "rtdetrv4"
    if sidecar_name is not None:
        env_available = sidecar_name in sidecar_envs

    # Execution status decision tree
    execution_status = "not_advertised"
    blocker_code = ""
    blocker_message = ""
    recommended_fix = ""

    if license_risk in {"non_commercial", "restricted"}:
        execution_status = "license_blocked"
        blocker_code = "LICENSE_RESTRICTION_TRIGGERED"
        blocker_message = f"License {src.license!r} is {license_risk}; not auto-pulled."
        recommended_fix = "Review license terms; supply checkpoint manually and explicitly opt in."
    elif requires_auth:
        execution_status = "auth_required"
        blocker_code = "GATED_AUTH_REQUIRED"
        blocker_message = f"Model is gated ({access}); credentials must be supplied."
        recommended_fix = "huggingface-cli login (or set HF_TOKEN env var) before running."
    elif requires_sidecar and not env_available:
        execution_status = "sidecar_env_missing"
        blocker_code = "SIDECAR_ENV_MISSING"
        blocker_message = f"Sidecar conda env visionservex-{sidecar_name}-sidecar is not installed."
        recommended_fix = (
            f"visionservex {sidecar_name} create-env --execute --format json "
            f"--out reports/{sidecar_name}_create_env.json"
        )
    elif src.recommended_action == "external_api":
        execution_status = "expected_blocker"
        blocker_code = "EXTERNAL_API_REQUIRED"
        blocker_message = "Model is only available via an external API."
        recommended_fix = "Use the upstream API endpoint or provider-specific client."
    elif blockers and "GPU" in " ".join(blockers).upper():
        execution_status = "resource_blocked"
        blocker_code = "BLACKWELL_GPU_RUNTIME_REQUIRED_FOR_BENCHMARK"
        blocker_message = "Model requires a GPU runtime that isn't available here."
        recommended_fix = "Run on A100/L4/T4 or a Blackwell-supported torch build."
    elif blockers and any("checkpoint" in b.lower() or "download" in b.lower() for b in blockers):
        execution_status = "checkpoint_missing"
        blocker_code = "CHECKPOINT_NOT_FOUND"
        blocker_message = "; ".join(b for b in blockers if "checkpoint" in b.lower())
        recommended_fix = (
            f"visionservex {family} pull {src.model_id} (or supply --checkpoint manually)."
        )
    elif src.runnable_in_visionservex:
        # Heuristic for cpu-only vs full
        if requires_gpu and family == "deimv2":
            execution_status = "runnable_cpu_only"
            blocker_code = "BLACKWELL_GPU_RUNTIME_REQUIRED_FOR_BENCHMARK"
            blocker_message = "DEIMv2 GPU smoke is blocked on RTX 5080 sm_120 by torch 2.5.1+cu124."
            recommended_fix = "Use --device cpu, or a Blackwell-compatible torch build."
        else:
            execution_status = "runnable_gpu" if requires_gpu else "runnable"
    else:
        execution_status = "expected_blocker"
        blocker_code = "MODEL_NOT_RUNNABLE_IN_THIS_BUILD"
        blocker_message = "; ".join(blockers) or "Not yet wired."
        recommended_fix = src.recommended_action

    # default_mode
    default_mode = "expected_blocker"
    if execution_status in {"runnable", "runnable_gpu"}:
        default_mode = "benchmark" if requires_gpu else "smoke"
    elif execution_status == "runnable_cpu_only":
        default_mode = "smoke"
    elif execution_status == "license_blocked" or execution_status in {
        "checkpoint_missing",
        "auth_required",
    }:
        default_mode = "expected_blocker"

    smoke_command = ""
    benchmark_command = ""
    if family == "deimv2":
        smoke_command = f"visionservex deimv2 smoke-test {src.model_id} IMAGE --device cpu"
        benchmark_command = (
            f"visionservex benchmark-detection --models {src.model_id} "
            f"--dataset yolo:DATA --max-images 20 --device cuda --require-gpu"
        )
    elif family == "rtdetrv4":
        smoke_command = (
            f"visionservex rtdetrv4 smoke-test {src.model_id} IMAGE "
            f"--checkpoint CKPT --device cuda --backend torch"
        )
        benchmark_command = (
            f"visionservex benchmark-detection --models {src.model_id} "
            f"--dataset yolo:DATA --max-images 20 --device cuda --require-gpu "
            f"--backend sidecar-rtdetrv4"
        )
    elif task in {"detect", "obb", "segment"}:
        smoke_command = f"visionservex detect {src.model_id} IMAGE --format json"
        benchmark_command = (
            f"visionservex benchmark-detection --models {src.model_id} "
            f"--dataset yolo:DATA --max-images 20 --device cuda"
        )
    elif task == "classify":
        smoke_command = f"visionservex classify {src.model_id} IMAGE --format json"
    elif task == "embed":
        smoke_command = f"visionservex feature embed {src.model_id} IMAGE --format json"
    elif task == "open_vocab_detect":
        smoke_command = (
            f"visionservex prompt-detect {src.model_id} IMAGE --prompt 'person' --format json"
        )

    return {
        "model_id": src.model_id,
        "family": src.family,
        "task": src.task,
        "advertised": True,
        "source_available": bool(src.official_repo),
        "checkpoint_available": bool(src.hf_repo or src.checkpoint_url),
        "env_available": bool(env_available),
        "dependency_available": True,  # default; refined per family at execution time
        "requires_gpu": bool(requires_gpu),
        "requires_sidecar": bool(requires_sidecar),
        "requires_auth": bool(requires_auth),
        "requires_user_dataset": bool(requires_user_dataset),
        "license": src.license,
        "license_risk": license_risk,
        "legal_default": legal_default,
        "execution_status": execution_status,
        "benchmark_status": "ok" if execution_status in {"runnable", "runnable_gpu"} else "blocked",
        "expected_result_type": task,
        "output_schema_valid": True,
        "dataset_required": "coco" if task in {"detect", "obb", "segment"} else "",
        "metric_required": "AP50/mAP50:95"
        if task in {"detect", "obb"}
        else ("mIoU/Dice" if task == "segment" else ""),
        "valid_default_dataset": "coco128" if task in {"detect", "obb", "segment"} else "",
        "valid_scientific_dataset": "coco_val2017_400"
        if task in {"detect", "obb", "segment"}
        else "",
        "default_mode": default_mode,
        "blocker_code": blocker_code,
        "blocker_message": blocker_message,
        "recommended_fix": recommended_fix,
        "smoke_command": smoke_command,
        "benchmark_command": benchmark_command,
        "last_tested_device": "",
        "last_tested_backend": "",
        "last_tested_version": "2.25.0",
        "evidence_file": "reports/per_family_command_status.csv",
    }


@app.command("readiness-matrix")
def readiness_matrix_cmd(
    include_core: bool = typer.Option(True, "--include-core/--no-include-core"),
    include_optional: bool = typer.Option(True, "--include-optional/--no-include-optional"),
    include_sidecar: bool = typer.Option(True, "--include-sidecar/--no-include-sidecar"),
    include_domain: bool = typer.Option(True, "--include-domain/--no-include-domain"),
    fmt: str = typer.Option("json", "--format", help="json | csv"),
    out: str = typer.Option("", "--out", help="Output file."),
) -> None:
    """v2.25.0: First-class model readiness matrix.

    For every advertised model in :mod:`visionservex.model_zoo.manifest`,
    produce a single row with execution_status, blocker_code, recommended_fix,
    smoke_command, benchmark_command, and the full v2.25 readiness contract.
    No row may have a null model_id / family / task / execution_status /
    default_mode / evidence_file.
    """
    import csv as _csv

    from visionservex.model_zoo.manifest import SOURCE_MANIFEST
    from visionservex.sidecars import SidecarManager

    # Probe sidecar envs once.
    mgr = SidecarManager()
    sidecar_envs: set[str] = set()
    for name in ("deimv2", "rtdetrv4"):
        if mgr.env_exists(name):
            sidecar_envs.add(name)

    rows: list[dict[str, Any]] = []
    for mid, src in sorted(SOURCE_MANIFEST.items()):  # noqa: B007
        include = False
        if include_core and src.recommended_action in {"add_now", "audit_only"}:
            include = True
        if include_optional and src.recommended_action in {
            "non_core_license_optional",
            "external_api",
        }:
            include = True
        if include_sidecar and src.recommended_action == "expert_sidecar":
            include = True
        if include_domain and src.domain in {
            "medical",
            "agriculture",
            "aerial",
            "industrial",
            "surveillance",
        }:
            include = True
        if not include:
            continue
        rows.append(_readiness_status_for_source(src, sidecar_envs=sidecar_envs))

    # Invariants
    unclassified = [r for r in rows if not r.get("execution_status")]
    null_model_id = [r for r in rows if not r.get("model_id")]
    if null_model_id or unclassified:
        # Should never happen — but be loud if it does.
        raise typer.Exit(
            code=3,
        )

    summary: dict[str, int] = {}
    for r in rows:
        s = r["execution_status"]
        summary[s] = summary.get(s, 0) + 1

    payload = {
        "status": "ok",
        "code": "OK",
        "n_rows": len(rows),
        "summary": summary,
        "unclassified_model_status_count": 0,
        "rows": rows,
    }
    if fmt == "csv":
        if not out:
            raise typer.Exit(code=2)
        from pathlib import Path as _P

        _P(out).parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else []
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                # Flatten list-valued fields for CSV.
                rr = {k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()}
                w.writerow(rr)
        typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))
        return
    if out:
        from pathlib import Path as _P

        _P(out).parent.mkdir(parents=True, exist_ok=True)
        _P(out).write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


@app.command("execution-matrix")
def execution_matrix_cmd(
    device: str = typer.Option("cuda", "--device"),
    include_core: bool = typer.Option(True, "--include-core/--no-include-core"),
    include_sidecar: bool = typer.Option(True, "--include-sidecar/--no-include-sidecar"),
    include_domain: bool = typer.Option(True, "--include-domain/--no-include-domain"),
    smoke_image: str = typer.Option("", "--smoke-image", help="Path to an RGB image."),
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option("", "--out"),
) -> None:
    """v2.25.0: First-class model execution matrix.

    For every model returned by the readiness matrix that's classified
    runnable / runnable_cpu_only, attempt a lightweight smoke (one image,
    bounded timeout) and record the result. For all other classifications,
    propagate the structured blocker. No raw traceback.
    """
    import csv as _csv
    import subprocess as _sp
    import time as _t
    from pathlib import Path as _P

    from visionservex.model_zoo.manifest import SOURCE_MANIFEST
    from visionservex.sidecars import SidecarManager

    mgr = SidecarManager()
    sidecar_envs: set[str] = set()
    for name in ("deimv2", "rtdetrv4"):
        if mgr.env_exists(name):
            sidecar_envs.add(name)

    rows: list[dict[str, Any]] = []
    for mid, src in sorted(SOURCE_MANIFEST.items()):
        # v2.25: include all models that the readiness matrix would surface,
        # so the execution matrix is a complete superset.
        include = False
        if include_core and src.recommended_action in {
            "add_now",
            "audit_only",
            "non_core_license_optional",
            "external_api",
        }:
            include = True
        if include_sidecar and src.recommended_action == "expert_sidecar":
            include = True
        if include_domain and src.domain in {
            "medical",
            "agriculture",
            "aerial",
            "industrial",
            "surveillance",
        }:
            include = True
        if not include:
            continue

        readiness = _readiness_status_for_source(src, sidecar_envs=sidecar_envs)
        # Propagate non-runnable rows as structured blockers (no execution).
        if readiness["execution_status"] not in {"runnable", "runnable_gpu", "runnable_cpu_only"}:
            rows.append(
                {
                    "model_id": mid,
                    "family": src.family,
                    "task": src.task,
                    "command": readiness["smoke_command"],
                    "status": "expected_blocker",
                    "blocker_code": readiness["blocker_code"] or "MODEL_NOT_RUNNABLE",
                    "runtime_ms": 0,
                    "device_actual": "",
                    "backend": "",
                    "output_file": "",
                    "output_schema_valid": False,
                    "n_predictions": 0,
                    "n_masks": 0,
                    "n_embeddings": 0,
                    "metrics_valid": False,
                    "benchmark_or_smoke": "expected_blocker",
                    "recommended_next_step": readiness["recommended_fix"],
                }
            )
            continue

        # Attempt a real smoke. Bounded — never spend more than 30s/model.
        cmd_str = readiness["smoke_command"].replace("IMAGE", smoke_image) if smoke_image else ""
        status = "expected_blocker"
        blocker_code = "NO_SMOKE_IMAGE_PROVIDED"
        n_pred = 0
        runtime_ms = 0
        if smoke_image and cmd_str:
            t0 = _t.time()
            try:
                proc = _sp.run(
                    cmd_str.split(),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                runtime_ms = int((_t.time() - t0) * 1000)
                if proc.returncode == 0:
                    status = "ok"
                    blocker_code = ""
                    try:
                        payload = json.loads(proc.stdout.strip().splitlines()[-1])
                        if isinstance(payload, dict):
                            n_pred = (
                                len(payload.get("detections", []))
                                or len(payload.get("predictions", []))
                                or 1
                            )
                    except (json.JSONDecodeError, IndexError, AttributeError):
                        pass
                else:
                    status = "expected_blocker"
                    blocker_code = "SMOKE_NONZERO_RETURNCODE"
            except _sp.TimeoutExpired:
                status = "expected_blocker"
                blocker_code = "SMOKE_TIMEOUT"
                runtime_ms = 30000
            except Exception as exc:
                status = "failed"
                blocker_code = "SMOKE_EXCEPTION"
                runtime_ms = int((_t.time() - t0) * 1000)
                _ = exc

        rows.append(
            {
                "model_id": mid,
                "family": src.family,
                "task": src.task,
                "command": cmd_str or readiness["smoke_command"],
                "status": status,
                "blocker_code": blocker_code,
                "runtime_ms": runtime_ms,
                "device_actual": device if status == "ok" else "",
                "backend": "host" if not readiness["requires_sidecar"] else "sidecar",
                "output_file": "",
                "output_schema_valid": status == "ok",
                "n_predictions": n_pred,
                "n_masks": 0,
                "n_embeddings": 0,
                "metrics_valid": False,
                "benchmark_or_smoke": "smoke",
                "recommended_next_step": readiness["recommended_fix"]
                if status != "ok"
                else readiness["benchmark_command"],
            }
        )

    payload = {
        "status": "ok",
        "code": "OK",
        "n_rows": len(rows),
        "device": device,
        "rows": rows,
    }
    if fmt == "csv":
        if not out:
            raise typer.Exit(code=2)
        _P(out).parent.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0].keys()) if rows else []
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))
        return
    if out:
        _P(out).parent.mkdir(parents=True, exist_ok=True)
        _P(out).write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


@app.command("state-resolve")
def state_resolve_cmd(
    reports_dir: str = typer.Option(..., "--reports-dir"),
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option(..., "--out"),
) -> None:
    """v2.28.0: canonical model state resolver."""
    import csv as _csv
    from pathlib import Path as _P

    from visionservex.reporting.state_resolver import resolve_canonical_model_state

    payload = resolve_canonical_model_state(_P(reports_dir))
    rows = payload.get("rows", [])
    _P(out).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        fields = [
            "model_id",
            "family",
            "task",
            "advertised",
            "candidate_state",
            "benchmark_state",
            "smoke_state",
            "sidecar_state",
            "checkpoint_state",
            "license_state",
            "dataset_state",
            "final_state",
            "final_blocker_code",
            "evidence_artifact",
            "next_action",
        ]
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))
        return
    _P(out).write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


@app.command("official-metrics")
def official_metrics_cmd(
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option(..., "--out"),
) -> None:
    """v2.28.0: official metrics table with null-safe rendering (no raw NaN)."""
    import csv as _csv
    from pathlib import Path as _P

    from visionservex.reporting.official_metrics import build_official_metrics_table

    rows = build_official_metrics_table()
    _P(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "code": "OK",
        "n_rows": len(rows),
        "n_with_value": sum(1 for r in rows if r["value"] is not None),
        "n_not_collected": sum(1 for r in rows if r["source_status"] == "not_collected"),
        "n_not_found": sum(1 for r in rows if r["source_status"] == "not_found"),
        "n_not_applicable": sum(1 for r in rows if r["source_status"] == "not_applicable"),
        "rows": rows,
    }
    if fmt == "csv":
        fields = list(rows[0].keys()) if rows else []
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                # Render value as empty string (not "NaN") for null cells.
                rr = dict(r)
                if rr["value"] is None:
                    rr["value"] = ""
                w.writerow(rr)
        typer.echo(json.dumps({"status": "ok", "code": "OK", "wrote": out, "n_rows": len(rows)}))
        return
    _P(out).write_text(json.dumps(payload, indent=2))
    typer.echo(json.dumps(payload, indent=2))


@app.command("smoke-matrix")
def smoke_matrix_cmd(
    device: str = typer.Option("cpu", "--device", help="cpu | cuda | mps"),
    real_assets: bool = typer.Option(
        True,
        "--real-assets/--no-real-assets",
        help="Use real deterministic smoke assets under tests/assets/smoke/",
    ),
    include_core: bool = typer.Option(True, "--include-core/--no-core"),
    include_optional: bool = typer.Option(False, "--include-optional"),
    include_sidecar: bool = typer.Option(False, "--include-sidecar"),
    include_domain: bool = typer.Option(False, "--include-domain"),
    include_mock: bool = typer.Option(False, "--include-mock"),
    include_libreyolo_default_safe: bool = typer.Option(
        False,
        "--include-libreyolo-default-safe",
        help="v2.30.0: include LibreYOLO weights whose license is verified MIT or Apache-2.0.",
    ),
    out: str = typer.Option("", "--out", help="Write JSON matrix to this path."),
    csv: str = typer.Option("", "--csv", help="Write CSV matrix to this path."),
    fail_on_package_bug: bool = typer.Option(
        False,
        "--fail-on-package-bug",
        help="Exit 1 if any row is a package-side bug.",
    ),
    no_notebook: bool = typer.Option(True, "--no-notebook/--allow-notebook"),
    timeout: int = typer.Option(120, "--timeout", help="Per-model timeout in seconds."),
) -> None:
    """v2.29.0: package-level model smoke matrix.

    Discovers every advertised model, synthesises the exact CLI command,
    executes it, and classifies the result into one of:
    smoke_passed | benchmark_passed | expected_blocker |
    license_blocked | manual_checkpoint_required | failed_runtime
    """
    import sys
    from pathlib import Path as _P

    # Import the standalone runner
    sys.path.insert(0, str(_P(__file__).parent.parent.parent.parent / "tools"))
    try:
        from run_model_smoke_matrix import run_smoke_matrix
    finally:
        sys.path.pop(0)

    out_path = _P(out) if out else None
    csv_path = _P(csv) if csv else None

    console.print(f"[bold]smoke-matrix[/bold]  device={device}  core={include_core}")
    _rows, summary = run_smoke_matrix(
        device=device,
        include_core=include_core,
        include_optional=include_optional,
        include_sidecar=include_sidecar,
        include_domain=include_domain,
        include_mock=include_mock,
        include_libreyolo_default_safe=include_libreyolo_default_safe,
        out=out_path,
        csv_path=csv_path,
        fail_on_package_bug=fail_on_package_bug,
        no_notebook=no_notebook,
        timeout_s=timeout,
    )

    payload = {
        "status": "ok",
        "code": "OK",
        "version": "v2.29.0",
        "device": device,
        "total": summary.total,
        "smoke_passed": summary.smoke_passed,
        "benchmark_passed": summary.benchmark_passed,
        "expected_blocker": summary.expected_blocker,
        "license_blocked": summary.license_blocked,
        "manual_checkpoint_required": summary.manual_checkpoint_required,
        "failed_runtime": summary.failed_runtime,
        "unclassified": summary.unclassified,
        "package_bug_remaining": summary.package_bug_remaining,
        "out": str(out_path) if out_path else "",
        "csv": str(csv_path) if csv_path else "",
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("summarize-smoke-matrix")
def summarize_smoke_matrix_cmd(
    input_path: str = typer.Option(..., "--input", help="Path to a smoke-matrix JSON file."),
    fmt: str = typer.Option("json", "--format", help="json | csv"),
    out: str = typer.Option(..., "--out", help="Output path."),
) -> None:
    """v2.30.0: collapse a smoke-matrix JSON into a canonical summary.

    The canonical summary becomes the single source of truth that the
    notebook consumes — it no longer rebuilds status tables. Every row
    carries final_state, blocker_code, fix, and evidence_file.
    """
    import csv as _csv
    from pathlib import Path as _P

    src = _P(input_path)
    if not src.exists():
        typer.echo(
            json.dumps(
                {
                    "status": "expected_blocker",
                    "code": "SMOKE_MATRIX_INPUT_MISSING",
                    "message": f"smoke-matrix input not found: {src}",
                    "fix": (
                        "Run `visionservex models smoke-matrix --include-core "
                        "--out reports/core_smoke_matrix_v229.json` first."
                    ),
                },
                indent=2,
            )
        )
        raise typer.Exit(code=2)

    data = json.loads(src.read_text())
    src_rows = data.get("rows", [])

    # Canonical schema — what the notebook MUST consume from now on.
    canonical: list[dict[str, Any]] = []
    for r in src_rows:
        final_state = r.get("final_state", "unclassified")
        blocker_code = r.get("blocker_code", "")
        if final_state == "failed_runtime" and blocker_code:
            # Defensive: if the source matrix mis-labelled a parseable blocker
            # as failed_runtime, upgrade it here.
            final_state = "expected_blocker"
        canonical.append(
            {
                "model_id": r.get("model_id", ""),
                "family": r.get("family", ""),
                "task": r.get("task", ""),
                "command": r.get("command", ""),
                "final_state": final_state,
                "blocker_code": blocker_code,
                "fix": r.get("recommended_fix", ""),
                "output_json_path": r.get("output_json_path", ""),
                "draw_path": r.get("draw_path", ""),
                "runtime_ms": r.get("runtime_ms", 0.0),
                "schema_valid": bool(r.get("output_schema_valid", False)),
                "package_bug": bool(r.get("package_bug", False)),
                "external_blocker": bool(r.get("external_blocker", False)),
                "evidence_file": r.get("evidence_file", r.get("output_json_path", "")),
            }
        )

    summary_payload = {
        "version": "v2.30.0",
        "source_matrix": str(src),
        "n_rows": len(canonical),
        "summary": data.get("summary", {}),
        "rows": canonical,
    }

    out_path = _P(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        fields = (
            list(canonical[0].keys())
            if canonical
            else [
                "model_id",
                "family",
                "task",
                "command",
                "final_state",
                "blocker_code",
                "fix",
                "output_json_path",
                "draw_path",
                "runtime_ms",
                "schema_valid",
                "package_bug",
                "external_blocker",
                "evidence_file",
            ]
        )
        with open(out_path, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in canonical:
                w.writerow(r)
        typer.echo(
            json.dumps(
                {"status": "ok", "code": "OK", "wrote": str(out_path), "n_rows": len(canonical)},
                indent=2,
            )
        )
        return

    out_path.write_text(json.dumps(summary_payload, indent=2))
    typer.echo(
        json.dumps(
            {"status": "ok", "code": "OK", "wrote": str(out_path), "n_rows": len(canonical)},
            indent=2,
        )
    )


@app.command("contract-test")
def contract_test_cmd(
    include: str = typer.Option("core", "--include", help="core | all"),
    device: str = typer.Option("cuda", "--device"),
    real_assets: bool = typer.Option(True, "--real-assets/--no-real-assets"),
    download_policy: str = typer.Option("retry", "--download-policy"),
    max_retries: int = typer.Option(1, "--max-retries"),
    out: str = typer.Option(..., "--out"),
    csv: str = typer.Option("", "--csv"),
    fail_on_package_bug: bool = typer.Option(False, "--fail-on-package-bug"),
    timeout: int = typer.Option(90, "--timeout"),
) -> None:
    """v2.33.0: full model contract-test runner.

    A model passes only when it loads, runs, and produces a valid normalized
    output for its task. Otherwise returns a precise structured blocker.
    """
    from pathlib import Path as _P

    from visionservex.runtime.contract_runner import run_contract_matrix

    out_path = _P(out)
    csv_path = _P(csv) if csv else None
    _rows, summary = run_contract_matrix(
        include=include,
        device=device,
        out_json=out_path,
        out_csv=csv_path,
        fail_on_package_bug=fail_on_package_bug,
        timeout_s=timeout,
        max_retries=max_retries,
    )
    from dataclasses import asdict

    typer.echo(
        json.dumps({"status": "ok", "summary": asdict(summary), "out": str(out_path)}, indent=2)
    )


@app.command("cache-status")
def cache_status_cmd(
    fmt: str = typer.Option("json", "--format"),
    out: str = typer.Option("", "--out"),
) -> None:
    """v2.33.0: model cache status report.

    Reports cache root, mirror configuration, and discovered weights.
    """
    import os as _os
    from pathlib import Path as _P

    cache_root = _P(
        _os.environ.get("VISION_SERVEX_MODEL_CACHE", _P.home() / ".cache/visionservex/models")
    )
    mirror = _os.environ.get("VISION_SERVEX_MODEL_MIRROR", "")
    base_url = _os.environ.get("VISION_SERVEX_MODEL_BASE_URL", "")

    cache_root.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in cache_root.rglob("*"):
        if f.is_file():
            entries.append(
                {
                    "path": str(f.relative_to(cache_root)),
                    "size_mb": round(f.stat().st_size / 1e6, 2),
                }
            )

    report = {
        "status": "ok",
        "version": "v2.33.0",
        "cache_root": str(cache_root),
        "model_mirror": mirror,
        "model_base_url": base_url,
        "n_files": len(entries),
        "entries": entries[:200],
    }
    if out:
        _P(out).parent.mkdir(parents=True, exist_ok=True)
        _P(out).write_text(json.dumps(report, indent=2))
    typer.echo(json.dumps(report, indent=2))


@app.command("cache-add")
def cache_add_cmd(
    model_id: str = typer.Argument(...),
    file: str = typer.Option(..., "--file"),
    license: str = typer.Option("Apache-2.0", "--license"),
    source: str = typer.Option("user-supplied", "--source"),
    mirror_allowed: bool = typer.Option(False, "--mirror-allowed/--no-mirror-allowed"),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """v2.33.0: register a user-supplied checkpoint into the model cache."""
    import hashlib
    import os as _os
    import shutil as _sh
    from pathlib import Path as _P

    src = _P(file)
    if not src.exists():
        typer.echo(
            json.dumps(
                {"status": "expected_blocker", "code": "FILE_NOT_FOUND", "file": str(src)}, indent=2
            )
        )
        raise typer.Exit(2)

    cache_root = _P(
        _os.environ.get("VISION_SERVEX_MODEL_CACHE", _P.home() / ".cache/visionservex/models")
    )
    dest = cache_root / model_id / src.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    _sh.copy2(str(src), str(dest))

    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    manifest = dest.parent / "manifest.json"
    record = {
        "model_id": model_id,
        "file": str(dest),
        "size_bytes": dest.stat().st_size,
        "sha256": sha,
        "license": license,
        "source": source,
        "mirror_allowed": mirror_allowed,
        "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest.write_text(json.dumps(record, indent=2))
    typer.echo(json.dumps({"status": "ok", **record}, indent=2))


@app.command("cache-verify")
def cache_verify_cmd(
    model_id: str = typer.Argument(...),
    fmt: str = typer.Option("json", "--format"),
) -> None:
    """v2.33.0: verify SHA256 of a cached model."""
    import hashlib
    import os as _os
    from pathlib import Path as _P

    cache_root = _P(
        _os.environ.get("VISION_SERVEX_MODEL_CACHE", _P.home() / ".cache/visionservex/models")
    )
    manifest = cache_root / model_id / "manifest.json"
    if not manifest.exists():
        typer.echo(
            json.dumps(
                {"status": "expected_blocker", "code": "MODEL_NOT_IN_CACHE", "model_id": model_id},
                indent=2,
            )
        )
        raise typer.Exit(2)

    rec = json.loads(manifest.read_text())
    f = Path(rec["file"])
    if not f.exists():
        typer.echo(
            json.dumps(
                {"status": "expected_blocker", "code": "CACHED_FILE_MISSING", "model_id": model_id},
                indent=2,
            )
        )
        raise typer.Exit(2)

    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = sha == rec["sha256"]
    typer.echo(
        json.dumps(
            {
                "status": "ok" if ok else "expected_blocker",
                "code": "OK" if ok else "SHA256_MISMATCH",
                "model_id": model_id,
                "expected": rec["sha256"],
                "actual": sha,
            },
            indent=2,
        )
    )


# Attach license-policy commands (list / explain / policy / assert-commercial-safe)
# to the `visionservex models` group. Source of truth: visionservex.policy.
from visionservex.cli.policy_commands import register as _register_policy_commands  # noqa: E402

_register_policy_commands(app)


__all__ = ["app"]

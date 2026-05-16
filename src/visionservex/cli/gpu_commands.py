# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""GPU / MPS / device smoke-test and benchmark commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="GPU/MPS device validation and smoke tests.")
console = Console()

# Processes that must never be killed by cleanup — system/GUI processes
_PROTECTED_PROCESS_KEYWORDS = frozenset(
    [
        "gnome-shell",
        "kwin",
        "plasmashell",
        "Xwayland",
        "Xorg",
        "xorg",
        "desktop",
        "wayfire",
        "sway",
        "weston",
        "mutter",
        "compiz",
        "firefox",
        "chrome",
        "chromium",
        "code",
        "vscode",
        "pycharm",
        "clion",
        "idea",
        "jupyter",
        "obs",
        "vlc",
        "mpv",
        "discord",
        "slack",
        "brave",
        "electron",
    ]
)

# Processes considered safe to terminate — VisionServeX, test, and benchmark processes
_SAFE_PROCESS_KEYWORDS = frozenset(
    [
        "visionservex",
        "pytest",
        "benchmark",
        "parallel-test",
        "gpu smoke-test",
    ]
)

# Default VRAM safety policy
_DEFAULT_VRAM_POLICY = {
    "max_vram_fraction": float(os.environ.get("VISIONSERVEX_RUNTIME__MAX_VRAM_FRACTION", "0.80")),
    "min_free_vram_gb": float(os.environ.get("VISIONSERVEX_RUNTIME__MIN_FREE_VRAM_GB", "3.0")),
    "reserve_gui_vram": os.environ.get("VISIONSERVEX_RUNTIME__RESERVE_GUI_VRAM", "true").lower()
    == "true",
    "desktop_gpu": os.environ.get("VISIONSERVEX_RUNTIME__DESKTOP_GPU", "true").lower() == "true",
    "allow_high_vram": os.environ.get("VISIONSERVEX_RUNTIME__ALLOW_HIGH_VRAM", "false").lower()
    == "true",
    "oom_guard_strict": os.environ.get("VISIONSERVEX_RUNTIME__OOM_GUARD_STRICT", "true").lower()
    == "true",
    "vram_safety_enabled": os.environ.get(
        "VISIONSERVEX_RUNTIME__VRAM_SAFETY_ENABLED", "true"
    ).lower()
    == "true",
}


def _emit(payload: dict | list, *, json_mode: bool) -> None:
    if json_mode:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


def _get_gpu_processes() -> list[dict]:
    """Return a list of GPU compute processes via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory,process_name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        procs = []
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            pid_str, mem_str, name = parts[0], parts[1], ",".join(parts[2:])
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            mem_mb = None
            if "MiB" in mem_str:
                try:
                    mem_mb = int(mem_str.replace("MiB", "").strip())
                except ValueError:
                    pass

            lower_name = name.lower()
            is_gui = any(kw in lower_name for kw in _PROTECTED_PROCESS_KEYWORDS)
            is_safe = any(kw in lower_name for kw in _SAFE_PROCESS_KEYWORDS)
            procs.append(
                {
                    "pid": pid,
                    "process_name": name,
                    "used_mb": mem_mb,
                    "protected": is_gui,
                    "safe_to_terminate": is_safe and not is_gui,
                }
            )
        return procs
    except Exception:
        return []


def _get_vram_state() -> dict:
    """Return current VRAM state using torch if available, nvidia-smi otherwise."""
    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            used_gb = total_gb - free_gb
            return {
                "source": "torch",
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "device_name": torch.cuda.get_device_name(0),
            }
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            line = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            total_mb, used_mb, free_mb = int(parts[1]), int(parts[2]), int(parts[3])
            return {
                "source": "nvidia-smi",
                "total_gb": round(total_mb / 1024, 2),
                "used_gb": round(used_mb / 1024, 2),
                "free_gb": round(free_mb / 1024, 2),
                "device_name": name,
            }
        except Exception:
            pass

    return {"source": "unavailable", "total_gb": None, "used_gb": None, "free_gb": None}


def _compute_safety_budget(vram: dict, policy: dict) -> dict:
    """Compute the effective VRAM safety budget given current state and policy."""
    if vram["total_gb"] is None:
        return {"status": "unavailable", "allowed_gb": None}

    total = vram["total_gb"]
    free = vram["free_gb"]
    used = vram["used_gb"]

    max_fraction = policy["max_vram_fraction"]
    min_free_gb = policy["min_free_vram_gb"]
    reserve_gui = policy["reserve_gui_vram"] and policy["desktop_gpu"]

    # GUI reserve: 3 GB on desktop, 0 if headless
    gui_reserve_gb = 3.0 if reserve_gui else 0.0

    # Effective minimum free = max(min_free_gb, gui_reserve_gb)
    effective_min_free = max(min_free_gb, gui_reserve_gb)

    # Max fraction limit
    fraction_cap = total * max_fraction

    # VisionServeX can use at most: free - effective_min_free, capped by fraction_cap - used
    budget_from_free = max(0.0, free - effective_min_free)
    budget_from_fraction = max(0.0, fraction_cap - used)
    budget_gb = min(budget_from_free, budget_from_fraction)

    safe = free >= effective_min_free and used <= fraction_cap

    return {
        "total_gb": total,
        "used_gb": used,
        "free_gb": free,
        "gui_reserve_gb": gui_reserve_gb,
        "effective_min_free_gb": effective_min_free,
        "fraction_cap_gb": round(fraction_cap, 2),
        "available_budget_gb": round(budget_gb, 2),
        "safe": safe,
        "status": "safe" if safe else "at_risk",
    }


@app.command("guard-status", help="Show VRAM safety guard status and current GPU memory state.")
def guard_status(
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Display the current VRAM safety guard configuration and live GPU memory state."""
    policy = _DEFAULT_VRAM_POLICY.copy()
    vram = _get_vram_state()
    budget = _compute_safety_budget(vram, policy)
    procs = _get_gpu_processes()

    payload = {
        "policy": policy,
        "vram": vram,
        "budget": budget,
        "gpu_processes": len(procs),
        "gpu_process_list": procs,
    }

    if json_:
        _emit(payload, json_mode=True)
        return

    console.print("[bold]VRAM Safety Guard Status[/bold]")
    console.print(
        f"  Safety enabled:    {'[green]yes[/green]' if policy['vram_safety_enabled'] else '[red]no[/red]'}"
    )
    console.print(f"  Max VRAM fraction: [cyan]{policy['max_vram_fraction'] * 100:.0f}%[/cyan]")
    console.print(f"  Min free VRAM:     [cyan]{policy['min_free_vram_gb']:.1f} GB[/cyan]")
    console.print(
        f"  GUI reservation:   {'[cyan]3.0 GB (desktop GPU)[/cyan]' if policy['reserve_gui_vram'] and policy['desktop_gpu'] else '[grey50]disabled[/grey50]'}"
    )
    console.print(
        f"  High-VRAM allowed: {'[yellow]yes (danger flag set)[/yellow]' if policy['allow_high_vram'] else '[green]no[/green]'}"
    )

    if vram["source"] == "unavailable":
        console.print("\n[grey50]No CUDA GPU detected — VRAM guard not applicable.[/grey50]")
        return

    console.print(f"\n[bold]Live GPU: {vram.get('device_name', 'unknown')}[/bold]")
    console.print(f"  Total VRAM:  {vram['total_gb']:.2f} GB")
    console.print(f"  Used VRAM:   {vram['used_gb']:.2f} GB")
    console.print(f"  Free VRAM:   {vram['free_gb']:.2f} GB")
    console.print(f"  Budget avail:{budget.get('available_budget_gb', '?'):.2f} GB for new loads")

    safe = budget.get("safe", True)
    status_str = "[green]safe[/green]" if safe else "[yellow]at_risk[/yellow]"
    console.print(f"\n  Guard status: {status_str}")
    console.print(f"  Active GPU processes: {len(procs)}")

    if not safe:
        console.print(
            "\n[yellow]VRAM safety buffer is at risk.[/yellow] "
            "Free VRAM is below the configured minimum. "
            "Run [cyan]visionservex gpu processes[/cyan] to inspect and "
            "[cyan]visionservex gpu cleanup[/cyan] to free memory."
        )


@app.command("processes", help="List current GPU compute processes.")
def processes(
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show all GPU compute processes with VisionServeX/pytest/benchmark processes marked."""
    if not shutil.which("nvidia-smi"):
        if json_:
            _emit({"error": "nvidia-smi not found", "processes": []}, json_mode=True)
        else:
            console.print("[grey50]nvidia-smi not found. Cannot list GPU processes.[/grey50]")
        return

    procs = _get_gpu_processes()

    if json_:
        _emit(procs, json_mode=True)
        return

    if not procs:
        console.print("[grey50]No GPU compute processes found.[/grey50]")
        return

    table = Table(title=f"GPU Compute Processes ({len(procs)})")
    table.add_column("PID")
    table.add_column("Memory (MB)")
    table.add_column("Process Name")
    table.add_column("Classification")

    for p in procs:
        if p["protected"]:
            cls = "[red]PROTECTED — do not kill[/red]"
        elif p["safe_to_terminate"]:
            cls = "[yellow]safe to terminate[/yellow]"
        else:
            cls = "[grey50]unknown[/grey50]"

        table.add_row(
            str(p["pid"]),
            str(p.get("used_mb") or "?"),
            p["process_name"][:60],
            cls,
        )

    console.print(table)
    console.print(
        "\nTo safely terminate VisionServeX/test processes: [cyan]visionservex gpu cleanup[/cyan]"
    )
    console.print("Protected processes (GUI/system) are [red]never[/red] touched by cleanup.")


@app.command("cleanup", help="Safely terminate VisionServeX/pytest/benchmark GPU processes.")
def cleanup(
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    include_python: bool = typer.Option(
        False,
        "--include-python",
        help="Also terminate Python GPU processes not matching VisionServeX/pytest names.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be terminated."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Terminate VisionServeX, pytest, and benchmark GPU processes.

    GUI/system processes (gnome-shell, Xwayland, browsers, editors) are
    always protected and never touched.
    """
    if not shutil.which("nvidia-smi"):
        if json_:
            _emit({"error": "nvidia-smi not found"}, json_mode=True)
        else:
            console.print("[grey50]nvidia-smi not found. Cannot inspect GPU processes.[/grey50]")
        return

    procs = _get_gpu_processes()
    targets = [p for p in procs if p["safe_to_terminate"]]

    if include_python:
        # Also include Python processes not already marked, not protected
        python_extra = [
            p
            for p in procs
            if not p["protected"]
            and not p["safe_to_terminate"]
            and "python" in p["process_name"].lower()
        ]
        targets = targets + python_extra

    if not targets:
        if json_:
            _emit(
                {"terminated": [], "message": "No safe-to-terminate GPU processes found."},
                json_mode=True,
            )
        else:
            console.print("[green]No VisionServeX/test GPU processes to clean up.[/green]")
        return

    if not json_:
        console.print(f"[bold]Safe-to-terminate GPU processes ({len(targets)}):[/bold]")
        for p in targets:
            mem = f"{p['used_mb']} MiB" if p.get("used_mb") else "? MiB"
            console.print(f"  PID {p['pid']:6d}  {mem:10s}  {p['process_name'][:60]}")

        if [p for p in procs if p["protected"]]:
            console.print(
                f"\n[dim]{sum(1 for p in procs if p['protected'])} protected GUI/system process(es) will NOT be touched.[/dim]"
            )

    if dry_run:
        if json_:
            _emit({"dry_run": True, "would_terminate": [p["pid"] for p in targets]}, json_mode=True)
        else:
            console.print("\n[dim]--dry-run: no processes terminated.[/dim]")
        return

    if not yes and not json_ and not typer.confirm(f"\nTerminate {len(targets)} process(es)?"):
        raise typer.Exit(0)

    terminated = []
    errors = []
    for p in targets:
        try:
            import signal

            os.kill(p["pid"], signal.SIGTERM)
            terminated.append(p["pid"])
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            errors.append({"pid": p["pid"], "error": str(exc)})

    if json_:
        _emit({"terminated": terminated, "errors": errors}, json_mode=True)
        return

    if terminated:
        console.print(f"\n[green]Sent SIGTERM to {len(terminated)} process(es).[/green]")
    if errors:
        console.print(
            f"\n[yellow]{len(errors)} process(es) could not be terminated (permission).[/yellow]"
        )
        console.print("  Try: [cyan]sudo kill -TERM <pid>[/cyan]")


@app.command("reset-advice", help="Show GPU recovery commands (does not auto-reset).")
def reset_advice() -> None:
    """Print emergency GPU recovery commands.

    VisionServeX will not reset the GPU automatically.
    Use these commands when VRAM is unexpectedly occupied.
    """
    console.print("[bold]GPU Recovery Advice[/bold]\n")

    console.print("[bold]1. Inspect GPU processes:[/bold]")
    console.print(
        "  nvidia-smi\n"
        "  nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader\n"
        "  visionservex gpu processes\n"
    )

    console.print("[bold]2. Terminate known test/benchmark processes:[/bold]")
    console.print(
        "  visionservex gpu cleanup\n"
        "  pkill -TERM -f 'pytest|visionservex|benchmark|parallel-test'\n"
    )

    console.print("[bold]3. If VRAM is still occupied after processes exit:[/bold]")
    console.print(
        "  # Wait a few seconds for driver to release memory\n"
        "  nvidia-smi  # re-check\n"
        "  # If still occupied, a process may hold it:\n"
        "  fuser /dev/nvidia0 2>/dev/null\n"
    )

    console.print("[bold]4. NEVER run these unless you understand the risk:[/bold]")
    console.print(
        "  # nvidia-smi --gpu-reset   # requires exclusive access, stops ALL GPU work\n"
        "  # modprobe -r nvidia       # unloads driver, stops ALL CUDA programs\n"
    )

    console.print(
        "[dim]VisionServeX never auto-resets the GPU. Emergency resets are a manual last resort.[/dim]"
    )


@app.command("smoke-test", help="Run a quick inference smoke test on available GPU devices.")
def smoke_test(
    models: str = typer.Option(
        "mock-detect",
        "--models",
        help="Comma-separated model IDs to test.",
    ),
    device: str = typer.Option("auto", "--device", help="Device to test (auto|cpu|cuda|mps)."),
    serial: bool = typer.Option(
        True, "--serial/--no-serial", help="Run models serially (default: true)."
    ),
    max_vram_fraction: float = typer.Option(
        0.80, "--max-vram-fraction", help="Max VRAM fraction before refusing to load."
    ),
    min_free_vram_gb: float = typer.Option(
        3.0, "--min-free-vram-gb", help="Minimum free VRAM to maintain."
    ),
    stop_on_vram_risk: bool = typer.Option(
        True, "--stop-on-vram-risk/--no-stop-on-vram-risk", help="Stop if VRAM budget is at risk."
    ),
    allow_high_vram: bool = typer.Option(
        False, "--allow-high-vram", help="Override VRAM safety guard (dangerous)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Load each model on the selected device and run one prediction.

    Runs serially by default to avoid VRAM contention.
    """
    from PIL import Image

    from visionservex import VisionModel
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.device import best_device

    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    results = []

    best = best_device()
    if not json_:
        console.print(f"Best device: [bold]{best.name}[/bold] — {best.detail}")
        if serial:
            console.print(
                "[dim]Running serially (--serial default). Use --no-serial to change.[/dim]"
            )

    # VRAM check before GPU tests
    policy = _DEFAULT_VRAM_POLICY.copy()
    if allow_high_vram:
        policy["allow_high_vram"] = True
    policy["max_vram_fraction"] = max_vram_fraction
    policy["min_free_vram_gb"] = min_free_vram_gb

    vram = _get_vram_state()
    if vram["source"] != "unavailable" and device not in ("cpu",):
        budget = _compute_safety_budget(vram, policy)
        if not budget.get("safe", True) and not allow_high_vram and stop_on_vram_risk:
            msg = (
                f"VRAM safety guard: free={vram['free_gb']:.2f}GB < "
                f"minimum {budget.get('effective_min_free_gb', min_free_vram_gb):.2f}GB. "
                "Use --allow-high-vram to override (dangerous) or free VRAM first."
            )
            if json_:
                _emit(
                    {
                        "error": "GPU_MEMORY_GUARD",
                        "message": msg,
                        "vram": vram,
                        "budget": budget,
                    },
                    json_mode=True,
                )
            else:
                console.print(f"[red]GPU_MEMORY_GUARD[/red]: {msg}")
                console.print("[dim]Run: visionservex gpu processes[/dim]")
            raise typer.Exit(1)

        if not json_:
            console.print(
                f"[dim]Desktop GPU: reserving {budget.get('gui_reserve_gb', 3.0):.1f}GB for GUI/system.[/dim]"
            )

    img = Image.new("RGB", (256, 256), color="blue")

    for mid in model_ids:
        entry = {"model_id": mid, "device": device}

        # Per-model VRAM check
        if vram["source"] != "unavailable" and device not in ("cpu",) and not allow_high_vram:
            vram_now = _get_vram_state()
            budget_now = _compute_safety_budget(vram_now, policy)
            if not budget_now.get("safe", True) and stop_on_vram_risk:
                entry["status"] = "skipped_vram_risk"
                entry["reason"] = f"VRAM safety guard: free={vram_now['free_gb']:.2f}GB. Stopping."
                results.append(entry)
                if not json_:
                    console.print(
                        f"  [yellow]skip[/yellow] {mid:30s} VRAM risk — stopping smoke test"
                    )
                break

        try:
            reg = default_registry()
            try:
                reg.get(mid)
            except RegistryError as exc:
                entry["status"] = "model_not_found"
                entry["error"] = str(exc)
                results.append(entry)
                continue

            t0 = time.perf_counter()
            m = VisionModel(mid, device=device)
            load_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            r = m.predict(img)
            infer_ms = (time.perf_counter() - t1) * 1000

            entry.update(
                {
                    "status": "ok",
                    "selected_device": r.device,
                    "precision": r.precision,
                    "backend": r.backend,
                    "load_ms": round(load_ms, 1),
                    "infer_ms": round(infer_ms, 1),
                    "fallback_reason": r.fallback_reason,
                    "warnings": r.warnings,
                }
            )
            if not json_:
                console.print(
                    f"  [green]ok[/green] {mid:30s} device={r.device} "
                    f"precision={r.precision} infer={infer_ms:.1f}ms"
                )

            # Clear CUDA cache between models when running serially
            if serial:
                try:
                    import gc

                    import torch

                    del m
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)[:200]
            if not json_:
                console.print(f"  [red]err[/red] {mid:30s} {str(exc)[:80]}")
        results.append(entry)

    if json_:
        _emit(results, json_mode=True)
    else:
        ok = sum(1 for r in results if r.get("status") == "ok")
        console.print(f"\n{ok}/{len(results)} models passed.")
        if ok < len(results):
            raise typer.Exit(1)


@app.command("doctor", help="Diagnose GPU health and show fix suggestions.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Detailed GPU diagnostics with actionable fix suggestions."""
    import platform

    from visionservex.runtime.device import available_devices
    from visionservex.utils.system import probe_dependencies

    devs = available_devices()
    deps = probe_dependencies()

    nvidia_smi_ok = shutil.which("nvidia-smi") is not None
    cuda_dev = next((d for d in devs if d.name == "cuda"), None)
    mps_dev = next((d for d in devs if d.name == "mps"), None)

    suggestions: list[str] = []

    if cuda_dev and not cuda_dev.available:
        if "not installed" in cuda_dev.detail.lower():
            suggestions.append("Install PyTorch with CUDA: pip install 'visionservex[torch]'")
        elif "broken" in cuda_dev.detail.lower() or "runtime" in cuda_dev.detail.lower():
            suggestions.append(
                "Reinstall matching PyTorch CUDA wheel: https://pytorch.org/get-started/locally/"
            )
            suggestions.append(
                "Check LD_LIBRARY_PATH includes CUDA lib dir (e.g. /usr/local/cuda/lib64)"
            )
            suggestions.append("Install CUDA toolkit: sudo apt install cuda-toolkit-12-x")
            suggestions.append(
                "Or use official CUDA Docker image: nvidia/cuda:12.x-runtime-ubuntu22.04"
            )
        suggestions.append("Temporary workaround: visionservex serve (auto-selects CPU)")

    payload = {
        "system": {"platform": platform.platform(), "python": __import__("sys").version.split()[0]},
        "nvidia_smi_available": nvidia_smi_ok,
        "cuda": cuda_dev.to_dict() if cuda_dev else None,
        "mps": mps_dev.to_dict() if mps_dev else None,
        "torch_installed": deps.get("torch", {}).get("installed", False),
        "torch_version": deps.get("torch", {}).get("version"),
        "suggestions": suggestions,
    }

    if json_:
        _emit(payload, json_mode=True)
        return

    table = Table(title="GPU Doctor Report")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row(
        "nvidia-smi", "[green]found[/green]" if nvidia_smi_ok else "[grey50]not found[/grey50]"
    )
    if cuda_dev:
        avail = "[green]healthy[/green]" if cuda_dev.available else "[red]unavailable[/red]"
        table.add_row("CUDA device", f"{avail} — {cuda_dev.detail[:70]}")
        if cuda_dev.total_vram_gb:
            table.add_row(
                "VRAM",
                f"{cuda_dev.total_vram_gb:.1f} GB total / {cuda_dev.free_vram_gb:.1f} GB free",
            )
    if mps_dev:
        avail = "[green]healthy[/green]" if mps_dev.available else "[grey50]not available[/grey50]"
        table.add_row("Apple MPS", avail)
    table.add_row(
        "torch",
        f"[green]{deps['torch']['version']}[/green]"
        if deps.get("torch", {}).get("installed")
        else "[grey50]not installed[/grey50]",
    )
    console.print(table)

    if suggestions:
        console.print("\n[bold]Fix suggestions:[/bold]")
        for s in suggestions:
            console.print(f"  [cyan]→[/cyan] {s}")


# ---------------------------------------------------------------------------
# VRAM lifecycle commands (v1.5.0)
# ---------------------------------------------------------------------------


@app.command("cleanup-cache", help="Flush GPU VRAM caches without killing any process.")
def cleanup_cache(json_: bool = typer.Option(False, "--json")) -> None:
    """Flush the CUDA allocator cache. Safe — does NOT kill any process or reset GPU."""
    from visionservex.runtime.gpu_lifecycle import (
        clear_torch_cuda_cache,
        get_gpu_memory_state,
    )

    before = get_gpu_memory_state("before")
    clear_torch_cuda_cache()
    after = get_gpu_memory_state("after")

    payload = {
        "before_allocated_mb": before.allocated_mb,
        "before_reserved_mb": before.reserved_mb,
        "after_allocated_mb": after.allocated_mb,
        "after_reserved_mb": after.reserved_mb,
        "freed_allocated_mb": round(before.allocated_mb - after.allocated_mb, 1),
        "freed_reserved_mb": round(before.reserved_mb - after.reserved_mb, 1),
        "cuda_available": after.cuda_available,
    }
    if json_:
        _emit(payload, json_mode=True)
        return
    if not after.cuda_available:
        console.print("[grey50]CUDA not available — no GPU cache to flush.[/grey50]")
        return
    freed = payload["freed_reserved_mb"]
    console.print(
        f"[green]GPU cache flushed.[/green] "
        f"Reserved: {before.reserved_mb:.1f} → {after.reserved_mb:.1f} MB "
        f"(freed {freed:.1f} MB). "
        f"Allocated: {before.allocated_mb:.1f} → {after.allocated_mb:.1f} MB."
    )


@app.command(
    "explain-memory", help="Show GPU memory state with explanation of allocated vs reserved."
)
def explain_memory(json_: bool = typer.Option(False, "--json")) -> None:
    """Print GPU memory state with a clear explanation of allocated vs cached/reserved."""
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state, get_process_gpu_memory

    state = get_gpu_memory_state("current")
    proc = get_process_gpu_memory()

    payload = {
        "cuda_available": state.cuda_available,
        "device_name": state.device_name,
        "allocated_mb": state.allocated_mb,
        "reserved_mb": state.reserved_mb,
        "max_allocated_mb": state.max_allocated_mb,
        "processes": proc.get("processes", []) if proc["available"] else [],
        "explanation": {
            "allocated": "Memory actively held by live PyTorch tensors.",
            "reserved": "Memory held by the CUDA caching allocator (can be reused without OS call).",
            "max_allocated": "Peak allocated during this process session.",
            "cleanup_command": "visionservex gpu cleanup-cache  (flushes reserved back to OS)",
        },
    }

    if json_:
        _emit(payload, json_mode=True)
        return

    if not state.cuda_available:
        console.print("[grey50]CUDA not available.[/grey50]")
        return

    console.print(f"[bold]GPU memory:[/bold] {state.device_name}")
    console.print(f"  Allocated (live tensors):   {state.allocated_mb:.1f} MB")
    console.print(f"  Reserved  (CUDA cache):     {state.reserved_mb:.1f} MB")
    console.print(f"  Peak allocated this session: {state.max_allocated_mb:.1f} MB")
    console.print("\n  [dim]'allocated' = live tensors held by Python objects[/dim]")
    console.print(
        "  [dim]'reserved' = CUDA allocator cache (safe to release with cleanup-cache)[/dim]"
    )

    if proc["available"] and proc.get("processes"):
        console.print("\n[bold]Per-process GPU usage:[/bold]")
        for p in proc["processes"]:
            console.print(
                f"  PID {p['pid']} {p['process_name'][:30]}: {p.get('used_memory_mb', '?')} MB"
            )

    console.print("\n  [cyan]$[/cyan] visionservex gpu cleanup-cache   # flush reserved memory")


@app.command(
    "memory-test",
    help="Load a model N times and check VRAM growth. Safe to stop with Ctrl+C.",
)
def memory_test(
    model_id: str,
    runs: int = typer.Option(5, "--runs", min=1, max=20),
    max_growth_mb: float = typer.Option(512.0, "--max-growth-mb"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Load, predict, unload a model N times. Reports VRAM growth per run."""
    from visionservex.core.model import VisionModel
    from visionservex.runtime.gpu_lifecycle import (
        assert_memory_returned_to_baseline,
        get_gpu_memory_state,
    )

    if not json_:
        console.print(
            f"[bold]VRAM memory test:[/bold] {model_id}  runs={runs}  max_growth={max_growth_mb} MB"
        )

    from PIL import Image as _PIL

    dummy_img = _PIL.new("RGB", (320, 240), "blue")
    baseline = get_gpu_memory_state("baseline")
    snapshots = [baseline.to_dict()]
    warnings_found = []

    for i in range(runs):
        try:
            with VisionModel(model_id) as model:
                model.predict(dummy_img)
        except Exception as exc:
            if not json_:
                console.print(f"  run {i + 1}: [red]error[/red] {str(exc)[:80]}")
            continue

        snap = get_gpu_memory_state(f"after_run_{i + 1}")
        check = assert_memory_returned_to_baseline(baseline, snap, max_growth_mb=max_growth_mb)
        snapshots.append(snap.to_dict())
        if not json_:
            color = "green" if check["status"] == "ok" else "yellow"
            console.print(
                f"  run {i + 1}: [{color}]{check['status']}[/{color}] "
                f"alloc={snap.allocated_mb:.1f} MB  reserved={snap.reserved_mb:.1f} MB  "
                f"growth={check['allocated_growth_mb']:+.1f} MB"
            )
        if check["status"] != "ok":
            warnings_found.append(check["message"])

    final = get_gpu_memory_state("final")
    final_check = assert_memory_returned_to_baseline(baseline, final, max_growth_mb=max_growth_mb)

    payload = {
        "model_id": model_id,
        "runs": runs,
        "baseline_mb": baseline.allocated_mb,
        "final_allocated_mb": final.allocated_mb,
        "total_growth_mb": final_check["allocated_growth_mb"],
        "status": final_check["status"],
        "message": final_check["message"],
        "warnings": warnings_found,
        "snapshots": snapshots,
    }
    if json_:
        _emit(payload, json_mode=True)
    else:
        color = "green" if final_check["status"] == "ok" else "yellow"
        console.print(f"\n[bold]Result:[/bold] [{color}]{final_check['status']}[/{color}]")
        console.print(f"  Total growth: {final_check['allocated_growth_mb']:+.1f} MB allocated")
        if warnings_found:
            for w in warnings_found:
                console.print(f"  [yellow]⚠[/yellow] {w}")


@app.command("memory-test-suite", help="VRAM memory test across multiple models.")
def memory_test_suite(
    models: str = typer.Option("mock-detect", "--models", help="Comma-separated model IDs."),
    max_growth_mb: float = typer.Option(512.0, "--max-growth-mb"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run VRAM memory test for each model sequentially. Reports total VRAM growth."""
    from visionservex.runtime.gpu_lifecycle import get_gpu_memory_state

    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    all_results = []
    baseline = get_gpu_memory_state("suite_baseline")

    for mid in model_ids:
        if not json_:
            console.print(f"  testing [cyan]{mid}[/cyan] ...", end=" ")
        try:
            from PIL import Image as _PIL

            from visionservex.core.model import VisionModel
            from visionservex.runtime.gpu_lifecycle import assert_memory_returned_to_baseline

            before = get_gpu_memory_state(f"before_{mid}")
            with VisionModel(mid) as model:
                model.predict(_PIL.new("RGB", (320, 240)))
            after = get_gpu_memory_state(f"after_{mid}")
            check = assert_memory_returned_to_baseline(before, after, max_growth_mb=max_growth_mb)
            result = {
                "model_id": mid,
                "growth_mb": check["allocated_growth_mb"],
                "status": check["status"],
            }
            if not json_:
                color = "green" if check["status"] == "ok" else "yellow"
                console.print(
                    f"[{color}]{check['status']}[/{color}] growth={check['allocated_growth_mb']:+.1f} MB"
                )
        except Exception as exc:
            result = {"model_id": mid, "status": "error", "error": str(exc)[:100], "growth_mb": 0.0}
            if not json_:
                console.print(f"[red]error[/red] {str(exc)[:60]}")
        all_results.append(result)

    final = get_gpu_memory_state("suite_final")
    suite_result = {
        "models_tested": len(model_ids),
        "baseline_mb": baseline.allocated_mb,
        "final_mb": final.allocated_mb,
        "total_growth_mb": round(final.allocated_mb - baseline.allocated_mb, 1),
        "results": all_results,
    }
    if json_:
        _emit(suite_result, json_mode=True)
    else:
        ok_count = sum(1 for r in all_results if r["status"] == "ok")
        console.print(
            f"\n[bold]Suite result:[/bold] {ok_count}/{len(model_ids)} models within VRAM limit"
        )
        console.print(f"  Total VRAM growth: {suite_result['total_growth_mb']:+.1f} MB")


@app.command("unload-all", help="Unload all models in the current process and flush VRAM caches.")
def unload_all(json_: bool = typer.Option(False, "--json")) -> None:
    """Flush all CUDA caches in the current process. Does NOT kill other processes."""
    from visionservex.runtime.gpu_lifecycle import (
        clear_torch_cuda_cache,
        force_gc,
        get_gpu_memory_state,
    )

    before = get_gpu_memory_state("before")
    force_gc()
    clear_torch_cuda_cache()
    after = get_gpu_memory_state("after")

    payload = {
        "freed_allocated_mb": round(before.allocated_mb - after.allocated_mb, 1),
        "freed_reserved_mb": round(before.reserved_mb - after.reserved_mb, 1),
        "after_allocated_mb": after.allocated_mb,
        "after_reserved_mb": after.reserved_mb,
    }
    if json_:
        _emit(payload, json_mode=True)
    else:
        console.print(
            f"[green]VRAM caches flushed.[/green] "
            f"Allocated: {before.allocated_mb:.1f}→{after.allocated_mb:.1f} MB  "
            f"Reserved: {before.reserved_mb:.1f}→{after.reserved_mb:.1f} MB"
        )


__all__ = ["app"]

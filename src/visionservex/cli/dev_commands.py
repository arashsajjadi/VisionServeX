# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Developer safety commands: test, resources, kill-tests, clean-*."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Developer safety commands — safe test runs, resource diagnostics, cleanup.",
    no_args_is_help=True,
)
console = Console()

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PYTEST_BASE = [
    sys.executable,
    "-m",
    "pytest",
    "-q",
    "--tb=short",
    "--maxfail=1",
    "--durations=20",
    "--no-header",
]
_QUICK_MARKERS = (
    "not slow and not real_model and not gpu and not network "
    "and not sidecar and not release and not benchmark "
    "and not memory and not disk_heavy and not download"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resource_guard():  # lazy import keeps CLI startup fast
    from visionservex.runtime import resource_guard

    return resource_guard


def _run_pytest(args: list[str], *, label: str) -> int:
    """Run pytest synchronously (never in background). Returns exit code."""
    rg = _resource_guard()
    console.rule(f"[bold cyan]{label}")
    rg.print_resource_report("before")
    rg.refuse_if_other_pytest_running()
    rg.acquire_pytest_lock()
    try:
        result = subprocess.run(args, cwd=str(_REPO_ROOT))
        return result.returncode
    finally:
        rg.release_pytest_lock()
        rg.cleanup_after_test()
        rg.print_resource_report("after")


# ---------------------------------------------------------------------------
# test sub-group
# ---------------------------------------------------------------------------

test_app = typer.Typer(help="Safe test runners.", no_args_is_help=True)
app.add_typer(test_app, name="test")


@test_app.command("quick")
def test_quick(
    target: str = typer.Option("", help="Limit to a specific test path or keyword."),
) -> None:
    """Run only fast, safe tests (no model/GPU/network/download).

    Target: finishes in under 60 seconds on a typical dev machine.
    """
    cmd = [*_PYTEST_BASE, "-m", _QUICK_MARKERS]
    if target:
        cmd.extend(["-k", target])
    else:
        cmd.append("tests/")
    rc = _run_pytest(cmd, label="Quick Safe Tests")
    raise typer.Exit(rc)


@test_app.command("targeted")
def test_targeted(
    target: str = typer.Argument(help="Test path, file, or -k keyword to run."),
) -> None:
    """Run a specific test file or keyword expression (still guards resources)."""
    rg = _resource_guard()
    try:
        rg.assert_safe_to_start_test()
    except rg.ResourceGuardError as exc:
        console.print(f"[red]Resource guard blocked test: {exc}[/red]")
        raise typer.Exit(1)
    cmd = [*_PYTEST_BASE, "-m", _QUICK_MARKERS, target]
    rc = _run_pytest(cmd, label=f"Targeted Tests: {target}")
    raise typer.Exit(rc)


@test_app.command("full-release")
def test_full_release() -> None:
    """Run the full test suite — includes resource pre-check and post-run cleanup.

    Only for release validation. Skips heavy markers unless env vars set.
    """
    rg = _resource_guard()
    try:
        rg.assert_safe_to_start_test()
    except rg.ResourceGuardError as exc:
        console.print(f"[red]Resource guard blocked full-release run: {exc}[/red]")
        raise typer.Exit(1)
    cmd = [*_PYTEST_BASE, "--tb=short", "--maxfail=5", "tests/"]
    rc = _run_pytest(cmd, label="Full Release Test Suite")
    raise typer.Exit(rc)


@test_app.command("real-smoke")
def test_real_smoke(
    allow_download: bool = typer.Option(
        False, "--allow-download", help="Allow downloading missing checkpoints."
    ),
    model: str = typer.Option("", "--model", help="Restrict to a specific model_id keyword."),
) -> None:
    """Run opt-in real model smoke tests (smallest models only, no GPU required).

    Requires VISIONSERVEX_RUN_REAL_MODEL_TESTS=1.
    Checks RAM and disk before starting. Loads at most one model at a time.
    """
    import os

    rg = _resource_guard()
    try:
        rg.refuse_if_other_pytest_running()
        rg.refuse_if_ram_above_threshold()
        rg.refuse_if_disk_free_below_threshold()
    except rg.ResourceGuardError as exc:
        console.print(f"[red]Resource guard: {exc}[/red]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["VISIONSERVEX_RUN_REAL_MODEL_TESTS"] = "1"
    env["VISIONSERVEX_ALLOW_CONCURRENT_PYTEST"] = "1"
    if allow_download:
        env["VISIONSERVEX_RUN_DOWNLOAD_TESTS"] = "1"

    markers = "real_model and smoke and not gpu and not slow and not sidecar"
    cmd = [
        *_PYTEST_BASE,
        "-m",
        markers,
        "--maxfail=1",
        "tests/",
    ]
    if model:
        cmd.extend(["-k", model])

    rg.print_resource_report("before-real-smoke")
    rg.acquire_pytest_lock()
    try:
        import subprocess

        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        rc = result.returncode
    finally:
        rg.release_pytest_lock()
        rg.cleanup_after_test()
        rg.print_resource_report("after-real-smoke")
    raise typer.Exit(rc)


@test_app.command("gpu-smoke")
def test_gpu_smoke(
    allow_gpu: bool = typer.Option(
        False, "--allow-gpu", help="Same as VISIONSERVEX_RUN_GPU_TESTS=1."
    ),
) -> None:
    """Run opt-in GPU smoke tests (VRAM-safe, tiny models only).

    Requires VISIONSERVEX_RUN_GPU_TESTS=1 or --allow-gpu.
    Checks free VRAM (>= 2 GB reserve) before starting.
    """
    import os

    rg = _resource_guard()
    gpu = rg.get_gpu_memory_state()

    if not gpu.cuda_available:
        console.print(
            "[yellow]No CUDA GPU detected. GPU smoke tests require a healthy GPU.[/yellow]"
        )
        raise typer.Exit(1)

    if not allow_gpu and os.environ.get("VISIONSERVEX_RUN_GPU_TESTS", "0") != "1":
        console.print("[red]GPU tests require --allow-gpu or VISIONSERVEX_RUN_GPU_TESTS=1[/red]")
        raise typer.Exit(1)

    try:
        rg.refuse_if_other_pytest_running()
        rg.refuse_if_vram_above_threshold(required_vram_gb=1.0)
        rg.refuse_if_ram_above_threshold()
    except rg.ResourceGuardError as exc:
        console.print(f"[red]Resource guard: {exc}[/red]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["VISIONSERVEX_RUN_GPU_TESTS"] = "1"
    env["VISIONSERVEX_ALLOW_CONCURRENT_PYTEST"] = "1"

    markers = "gpu and smoke and not slow and not sidecar"
    cmd = [*_PYTEST_BASE, "-m", markers, "--maxfail=1", "tests/"]

    rg.print_resource_report("before-gpu-smoke")
    rg.acquire_pytest_lock()
    try:
        import subprocess

        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        rc = result.returncode
    finally:
        rg.release_pytest_lock()
        rg.cleanup_after_test()
        rg.print_resource_report("after-gpu-smoke")
    raise typer.Exit(rc)


@test_app.command("benchmark-smoke")
def test_benchmark_smoke(
    out: str = typer.Option("", "--out", help="Output directory for results (default: tmp dir)."),
) -> None:
    """Run process-isolated benchmark smoke tests (max 3 images, bounded output).

    Requires VISIONSERVEX_RUN_BENCHMARK_TESTS=1 (set automatically here).
    """
    import os
    import tempfile

    rg = _resource_guard()
    try:
        rg.refuse_if_other_pytest_running()
        rg.refuse_if_ram_above_threshold()
        rg.refuse_if_disk_free_below_threshold()
    except rg.ResourceGuardError as exc:
        console.print(f"[red]Resource guard: {exc}[/red]")
        raise typer.Exit(1)

    env = os.environ.copy()
    env["VISIONSERVEX_RUN_BENCHMARK_TESTS"] = "1"
    env["VISIONSERVEX_ALLOW_CONCURRENT_PYTEST"] = "1"
    if out:
        env["VISIONSERVEX_BENCHMARK_OUT"] = out
    else:
        tmp = tempfile.mkdtemp(prefix="visionservex_bench_smoke_")
        env["VISIONSERVEX_BENCHMARK_OUT"] = tmp
        console.print(f"[dim]Benchmark output: {tmp}[/dim]")

    markers = "benchmark and smoke and not slow and not sidecar and not gpu"
    cmd = [*_PYTEST_BASE, "-m", markers, "--maxfail=1", "tests/"]

    rg.print_resource_report("before-benchmark-smoke")
    rg.acquire_pytest_lock()
    try:
        import subprocess

        result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
        rc = result.returncode
    finally:
        rg.release_pytest_lock()
        rg.cleanup_after_test()
        rg.print_resource_report("after-benchmark-smoke")
    raise typer.Exit(rc)


# ---------------------------------------------------------------------------
# top-level dev commands
# ---------------------------------------------------------------------------


@app.command("kill-tests")
def kill_tests() -> None:
    """Kill VisionServeX pytest processes running in this repo (repo-scoped only)."""
    from visionservex.runtime.resource_guard import kill_visionservex_tests

    killed = kill_visionservex_tests()
    if killed:
        console.print(f"[yellow]Killed PIDs: {killed}[/yellow]")
    else:
        console.print("[green]No active VisionServeX test processes found.[/green]")


@app.command("resources")
def resources() -> None:
    """Print a full resource usage report (RAM, VRAM, CPU, disk, processes)."""
    from visionservex.runtime.resource_guard import enforce_resource_budget, print_resource_report

    print_resource_report("current")
    budget = enforce_resource_budget()
    data = budget.to_dict()

    table = Table(title="Resource Budget Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for k, v in data.items():
        table.add_row(str(k), str(v))
    console.print(table)


@app.command("clean-temp")
def clean_temp() -> None:
    """Remove test temp directories under /tmp matching visionservex/pytest patterns."""
    import glob

    patterns = [
        "/tmp/visionservex_*",
        "/tmp/pytest-*",
        "/tmp/tmp_visionservex*",
    ]
    removed = []
    for pat in patterns:
        for path in glob.glob(pat):
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                removed.append(path)
            elif p.is_file():
                try:
                    p.unlink()
                    removed.append(path)
                except OSError:
                    pass
    if removed:
        console.print(f"[green]Removed {len(removed)} temp path(s).[/green]")
        for r in removed:
            console.print(f"  {r}")
    else:
        console.print("[dim]No temp artifacts found.[/dim]")


@app.command("clean-reports")
def clean_reports() -> None:
    """Remove generated reports/ and outputs/ directories from repo root."""
    dirs = [_REPO_ROOT / "reports", _REPO_ROOT / "outputs", _REPO_ROOT / ".pytest_cache"]
    removed = []
    for d in dirs:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))
    if removed:
        console.print(f"[green]Removed: {removed}[/green]")
    else:
        console.print("[dim]Nothing to remove.[/dim]")


@app.command("clean-cache")
def clean_cache(
    safe: bool = typer.Option(False, "--safe", help="Only remove safe caches."),
) -> None:
    """Remove __pycache__, *.pyc, and .mypy_cache from the repo tree."""
    removed = 0
    for cache_dir in _REPO_ROOT.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        removed += 1
    for pyc in _REPO_ROOT.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)
        removed += 1
    mypy_cache = _REPO_ROOT / ".mypy_cache"
    if mypy_cache.exists():
        shutil.rmtree(mypy_cache, ignore_errors=True)
        removed += 1
    if not safe:
        ruff_cache = _REPO_ROOT / ".ruff_cache"
        if ruff_cache.exists():
            shutil.rmtree(ruff_cache, ignore_errors=True)
            removed += 1
    console.print(f"[green]Cleaned {removed} cache path(s).[/green]")


@app.command("disk-report")
def disk_report() -> None:
    """Show disk usage breakdown for key repo directories."""
    from visionservex.runtime.resource_guard import get_disk_state

    paths = [
        _REPO_ROOT,
        _REPO_ROOT / "reports",
        _REPO_ROOT / "outputs",
        _REPO_ROOT / "dist",
        Path("/tmp"),
    ]
    table = Table(title="Disk Report", show_header=True)
    table.add_column("Path", style="cyan")
    table.add_column("Free GB", style="green")
    table.add_column("Used %", style="yellow")
    for p in paths:
        if not p.exists():
            continue
        d = get_disk_state(p)
        table.add_row(str(p), f"{d.free_gb:.1f}", f"{d.used_pct:.1f}%")
    console.print(table)

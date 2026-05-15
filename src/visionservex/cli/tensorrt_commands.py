# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""TensorRT doctor, build, and benchmark commands."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="TensorRT engine build and benchmark (experimental).")
console = Console()


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except ImportError:
        return False


def _onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except ImportError:
        return False


@app.command("doctor", help="Check TensorRT and ONNX Runtime availability.")
def doctor(json_: bool = typer.Option(False, "--json")) -> None:
    """Check TensorRT, ONNX Runtime, and related dependency status."""
    import shutil

    trt = _trt_available()
    ort = _onnx_available()
    trtexec = shutil.which("trtexec") is not None

    try:
        import tensorrt as trt_mod

        trt_version = getattr(trt_mod, "__version__", "unknown")
    except Exception:
        trt_version = None

    try:
        import onnxruntime as ort_mod

        ort_version = getattr(ort_mod, "__version__", "unknown")
    except Exception:
        ort_version = None

    payload = {
        "tensorrt_installed": trt,
        "tensorrt_version": trt_version,
        "trtexec_available": trtexec,
        "onnxruntime_installed": ort,
        "onnxruntime_version": ort_version,
        "status": ("ready" if trt and trtexec else "onnx_only" if ort else "not_available"),
        "suggestions": [],
    }

    if not trt:
        payload["suggestions"].append(
            "Install TensorRT: follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
        )
        payload["suggestions"].append("For development: pip install tensorrt (NVIDIA wheel)")
    if not trtexec:
        payload["suggestions"].append("Install trtexec: included in TensorRT SDK under bin/")
    if not ort:
        payload["suggestions"].append("Install ONNXRuntime: pip install 'visionservex[onnx]'")

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print("[bold]TensorRT Doctor[/bold]")
    table = Table(show_header=False, box=None)
    table.add_column("k", style="cyan")
    table.add_column("v")
    table.add_row("TensorRT installed", "[green]yes[/green]" if trt else "[grey50]no[/grey50]")
    if trt_version:
        table.add_row("TensorRT version", trt_version)
    table.add_row("trtexec available", "[green]yes[/green]" if trtexec else "[grey50]no[/grey50]")
    table.add_row("ONNXRuntime installed", "[green]yes[/green]" if ort else "[grey50]no[/grey50]")
    if ort_version:
        table.add_row("ONNXRuntime version", ort_version)
    console.print(table)

    if payload["suggestions"]:
        console.print("\n[bold]Suggestions:[/bold]")
        for s in payload["suggestions"]:
            console.print(f"  [cyan]→[/cyan] {s}")
    else:
        console.print("\n[green]TensorRT is available.[/green]")


@app.command("build", help="Build a TensorRT engine from an ONNX file.")
def build(
    model_id: str,
    onnx_path: Path = typer.Option(..., "--onnx", help="Path to input ONNX model."),
    output: Path | None = typer.Option(None, "--output", help="Output engine path."),
    fp16: bool = typer.Option(True, "--fp16/--no-fp16"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Build a TensorRT engine from an ONNX model using trtexec or TRT Python API.

    If TensorRT is not installed, prints the equivalent trtexec command.
    Generated .engine files are excluded from Git via .gitignore.
    """
    import shutil

    if output is None:
        output = onnx_path.with_suffix(".engine")

    if not onnx_path.exists():
        typer.echo(f"ONNX file not found: {onnx_path}", err=True)
        raise typer.Exit(1)

    trtexec = shutil.which("trtexec")
    trt = _trt_available()

    if dry_run or (not trtexec and not trt):
        cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={output}"]
        if fp16:
            cmd.append("--fp16")
        payload = {
            "model_id": model_id,
            "onnx": str(onnx_path),
            "output": str(output),
            "fp16": fp16,
            "dry_run": True,
            "command": " ".join(cmd),
            "trt_available": trt,
            "trtexec_available": bool(trtexec),
            "note": (
                "TensorRT is not installed. Install it and run the command above."
                if not trt
                else "Dry run mode — engine not built."
            ),
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2, default=str))
        else:
            console.print("[yellow]Dry run (TensorRT not installed or --dry-run):[/yellow]")
            console.print(f"  [cyan]{' '.join(cmd)}[/cyan]")
            console.print("\n  docs/tensorrt.md for full instructions.")
        return

    # Actual build via trtexec
    import subprocess

    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={output}"]
    if fp16:
        cmd.append("--fp16")

    console.print(f"Building TensorRT engine: {output}")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        console.print(f"[green]Engine saved to {output}[/green]")
        console.print("[yellow]Note: .engine files are excluded from Git via .gitignore[/yellow]")
    else:
        console.print("[red]trtexec failed:[/red]")
        typer.echo(result.stderr[-2000:])
        raise typer.Exit(result.returncode)


@app.command("benchmark", help="Benchmark a TensorRT engine.")
def benchmark_engine(
    engine_path: Path = typer.Argument(..., help="Path to .engine file."),
    runs: int = typer.Option(10, "--runs"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run a quick latency benchmark on a TensorRT engine using trtexec."""
    import shutil
    import subprocess

    trtexec = shutil.which("trtexec")
    if not trtexec:
        console.print("[red]trtexec not found.[/red] See docs/tensorrt.md")
        raise typer.Exit(1)

    if not engine_path.exists():
        console.print(f"[red]Engine not found: {engine_path}[/red]")
        raise typer.Exit(1)

    cmd = [trtexec, f"--loadEngine={engine_path}", f"--iterations={runs}"]
    console.print(f"Benchmarking {engine_path} ...")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        # Parse throughput from trtexec output
        for line in result.stdout.splitlines():
            if "Throughput" in line or "mean" in line.lower() or "Latency" in line:
                console.print(f"  {line.strip()}")
    else:
        typer.echo(result.stderr[-1000:])


__all__ = ["app"]

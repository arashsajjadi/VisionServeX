# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""GPU / MPS / device smoke-test and benchmark commands."""

from __future__ import annotations

import json
import time

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="GPU/MPS device validation and smoke tests.")
console = Console()


def _emit(payload: dict | list, *, json_mode: bool) -> None:
    if json_mode:
        typer.echo(json.dumps(payload, indent=2, default=str))
    else:
        typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("smoke-test", help="Run a quick inference smoke test on available GPU devices.")
def smoke_test(
    models: str = typer.Option(
        "mock-detect",
        "--models",
        help="Comma-separated model IDs to test.",
    ),
    device: str = typer.Option("auto", "--device", help="Device to test (auto|cpu|cuda|mps)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Load each model on the selected device and run one prediction."""
    from PIL import Image

    from visionservex import VisionModel
    from visionservex.registry import RegistryError, default_registry
    from visionservex.runtime.device import best_device

    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    results = []

    best = best_device()
    if not json_:
        console.print(f"Best device: [bold]{best.name}[/bold] — {best.detail}")

    img = Image.new("RGB", (256, 256), color="blue")

    for mid in model_ids:
        entry = {"model_id": mid, "device": device}
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
    import shutil

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


__all__ = ["app"]

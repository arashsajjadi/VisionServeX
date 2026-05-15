# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Benchmark matrix, parallel test, and server load benchmarks."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Benchmarks: latency matrix, concurrency, server load.")
console = Console()


def _load_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB") if path.exists() else Image.new("RGB", (320, 240))


def _run_benchmark_model(
    model_id: str,
    device: str,
    image_path: Path,
    runs: int,
    warmup: int,
) -> dict:
    from visionservex import VisionModel
    from visionservex.engines.base import MissingDependencyError
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired

    result = {
        "model_id": model_id,
        "requested_device": device,
        "runs": runs,
        "warmup": warmup,
    }
    try:
        img = _load_image(image_path)
        t0 = time.perf_counter()
        m = VisionModel(model_id, device=device)
        m._ensure_loaded()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warmup
        for _ in range(warmup):
            m.predict(img)

        # Measured runs
        latencies = []
        for _ in range(runs):
            t = time.perf_counter()
            r = m.predict(img)
            latencies.append((time.perf_counter() - t) * 1000)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        p99 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.99))]

        result.update(
            {
                "status": "ok",
                "selected_device": r.device,
                "precision": r.precision,
                "backend": r.backend,
                "cold_load_ms": round(cold_ms, 1),
                "warm_p50_ms": round(p50, 2),
                "warm_p95_ms": round(p95, 2),
                "warm_p99_ms": round(p99, 2),
                "throughput_req_s": round(1000.0 / p50, 2),
                "fallback_reason": r.fallback_reason,
            }
        )
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        result["status"] = "skip"
        result["skip_reason"] = str(exc)[:120]
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:200]
    return result


@app.command("benchmark-matrix", help="Benchmark multiple models across multiple devices.")
def benchmark_matrix(
    models: str = typer.Option("mock-detect", "--models", help="Comma-separated model IDs."),
    devices: str = typer.Option(
        "cpu", "--devices", help="Comma-separated devices (cpu,cuda,mps,auto)."
    ),
    runs: int = typer.Option(5, "--runs", min=1, max=200),
    warmup: int = typer.Option(2, "--warmup", min=0, max=20),
    input_path: Path = typer.Option(
        Path("examples/images/street.jpg"),
        "--input",
        exists=False,
    ),
    out: Path | None = typer.Option(None, "--out", help="Write results JSON to this path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run inference across a matrix of (model, device) combinations."""
    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    device_list = [d.strip() for d in devices.split(",") if d.strip()]

    all_results = []
    for mid in model_ids:
        for dev in device_list:
            if not json_:
                console.print(f"  benchmarking {mid} @ {dev} ...", end=" ")
            r = _run_benchmark_model(mid, dev, input_path, runs, warmup)
            all_results.append(r)
            if not json_:
                status = r.get("status", "?")
                if status == "ok":
                    console.print(
                        f"[green]{status}[/green] p50={r.get('warm_p50_ms')}ms device={r.get('selected_device')}"
                    )
                elif status == "skip":
                    console.print(f"[yellow]skip[/yellow] {r.get('skip_reason', '')[:60]}")
                else:
                    console.print(f"[red]{status}[/red] {r.get('error', '')[:60]}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
        if not json_:
            console.print(f"\nResults written to {out}")

    if json_:
        typer.echo(json.dumps(all_results, indent=2, default=str))
        return

    # Print summary table
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    if ok_results:
        table = Table(title="Benchmark Matrix (warm latency)")
        for col in ("Model", "Device", "Prec", "P50 ms", "P95 ms", "Req/s", "Fallback"):
            table.add_column(col)
        for r in all_results:
            st = r.get("status", "?")
            if st == "ok":
                table.add_row(
                    r["model_id"],
                    r.get("selected_device", "?"),
                    r.get("precision", "?"),
                    str(r.get("warm_p50_ms", "?")),
                    str(r.get("warm_p95_ms", "?")),
                    str(r.get("throughput_req_s", "?")),
                    r.get("fallback_reason") or "-",
                )
            else:
                table.add_row(r["model_id"], r.get("requested_device", "?"), "-", "-", "-", "-", st)
        console.print(table)


@app.command("parallel-test", help="Test concurrent inference throughput and scheduler behavior.")
def parallel_test(
    model_id: str,
    input_path: Path,
    concurrency: int = typer.Option(2, "--concurrency", min=1, max=16),
    runs: int = typer.Option(5, "--runs", min=1, max=100),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Run `concurrency` requests in parallel and compare to sequential baseline."""
    from visionservex import VisionModel

    if not input_path.exists():
        # Use synthetic image
        from PIL import Image

        img = Image.new("RGB", (320, 240), "blue")
    else:
        from PIL import Image

        img = Image.open(input_path).convert("RGB")

    m = VisionModel(model_id, device=device)
    m._ensure_loaded()

    # Sequential baseline
    seq_times = []
    for _ in range(runs):
        t = time.perf_counter()
        m.predict(img)
        seq_times.append((time.perf_counter() - t) * 1000)
    seq_times.sort()
    seq_p50 = seq_times[len(seq_times) // 2]

    # Concurrent run
    import threading

    barrier = threading.Barrier(concurrency)

    def _worker(results: list, idx: int) -> None:
        barrier.wait()  # release all at once
        t = time.perf_counter()
        try:
            m.predict(img)
            results.append((time.perf_counter() - t) * 1000)
        except Exception as exc:
            results.append(str(exc))

    wall_times = []
    for _ in range(runs):
        conc_per_run: list = []
        threads = [
            threading.Thread(target=_worker, args=(conc_per_run, i)) for i in range(concurrency)
        ]
        t_wall = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall_times.append((time.perf_counter() - t_wall) * 1000)

    valid_wall = [w for w in wall_times if isinstance(w, (int, float))]
    wall_p50 = sorted(valid_wall)[len(valid_wall) // 2] if valid_wall else 0.0
    slowdown = (wall_p50 / seq_p50 - 1.0) * 100 if seq_p50 > 0 else 0.0

    if slowdown <= 10:
        status = "excellent_parallelism"
    elif slowdown <= 25:
        status = "acceptable_parallelism"
    else:
        status = "scheduler_needs_queueing"

    payload = {
        "model_id": model_id,
        "device": device,
        "concurrency": concurrency,
        "runs": runs,
        "sequential_p50_ms": round(seq_p50, 2),
        "concurrent_wall_p50_ms": round(wall_p50, 2),
        "slowdown_pct": round(slowdown, 1),
        "status": status,
    }

    if json_:
        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    console.print(f"\n[bold]Parallel test:[/bold] {model_id} x {concurrency} concurrent")
    console.print(f"  Sequential p50:    {seq_p50:.1f} ms")
    console.print(f"  Concurrent wall p50: {wall_p50:.1f} ms")
    console.print(f"  Slowdown:          {slowdown:.1f}%")
    color = (
        "green"
        if status == "excellent_parallelism"
        else "yellow"
        if "acceptable" in status
        else "red"
    )
    console.print(f"  Status:            [{color}]{status}[/{color}]")


@app.command("benchmark-server", help="Load-test a running VisionServeX server.")
def benchmark_server(
    url: str = typer.Option("http://127.0.0.1:8080", "--url"),
    model: str = typer.Option("mock-detect", "--model"),
    concurrency: str = typer.Option("1,2,4", "--concurrency", help="Comma-separated values."),
    runs: int = typer.Option(10, "--runs"),
    image: Path = typer.Option(Path("examples/images/street.jpg"), "--image"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark a running server at different concurrency levels."""
    try:
        import httpx
    except ImportError:
        typer.echo("httpx is required: pip install httpx")
        raise typer.Exit(1)

    if not image.exists():
        import io

        from PIL import Image

        img = Image.new("RGB", (320, 240), "blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()
    else:
        image_bytes = image.read_bytes()

    levels = [int(c.strip()) for c in concurrency.split(",") if c.strip()]
    all_results = []

    async def _run_level(lvl: int, img_b: bytes, runs_: int, lat_out: list, busy_out: list) -> None:
        async def _req(cli: httpx.AsyncClient) -> None:
            t = time.perf_counter()
            r = await cli.post(
                f"{url}/detect",
                files={"image": ("img.jpg", img_b, "image/jpeg")},
                data={"model_id": model},
                timeout=30.0,
            )
            elapsed = (time.perf_counter() - t) * 1000
            if r.status_code == 200:
                lat_out.append(elapsed)
            elif r.status_code == 503:
                busy_out.append(1)

        async with httpx.AsyncClient() as cli:
            for _ in range(runs_):
                await asyncio.gather(*[_req(cli) for _ in range(lvl)])

    for level in levels:
        lat_list: list = []
        busy_list: list = []
        asyncio.run(_run_level(level, image_bytes, runs, lat_list, busy_list))
        latencies = lat_list
        busy_count = len(busy_list)

        lat = sorted(latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else 0.0
        entry = {
            "concurrency": level,
            "successful": len(latencies),
            "busy_rejected": busy_count,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "throughput_req_s": round(len(latencies) / (p50 / 1000 * runs) if p50 > 0 else 0, 2),
        }
        all_results.append(entry)
        if not json_:
            console.print(
                f"  concurrency={level}: p50={p50:.1f}ms p95={p95:.1f}ms "
                f"ok={len(latencies)} busy={busy_count}"
            )

    if json_:
        typer.echo(json.dumps(all_results, indent=2, default=str))


__all__ = ["app"]

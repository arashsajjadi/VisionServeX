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


@app.command(
    "benchmark-competitiveness",
    help=(
        "Compare detection models head-to-head. "
        "Reports latency + detection health (synthetic mode) OR real AP50/mAP50:95 (with --dataset)."
    ),
)
def benchmark_competitiveness(
    models: str = typer.Option(
        "dfine-s-o365-coco,rfdetr-small",
        "--models",
        help=(
            "Comma-separated model IDs (detect task). "
            "Prefix with 'ultralytics:yolo11n' for YOLO baseline."
        ),
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help=(
            "Dataset for real AP evaluation. Options: "
            "'synthetic' (default, latency-only), "
            "'yolo:<path>' (YOLO-format directory), "
            "'coco-json:<images_dir>:<ann_file>' (COCO JSON). "
            "When omitted: synthetic images, latency + detection health only."
        ),
    ),
    max_images: int = typer.Option(
        20, "--max-images", min=1, max=5000, help="Number of images to evaluate."
    ),
    threshold: float = typer.Option(
        0.3,
        "--threshold",
        min=0.01,
        max=1.0,
        help="Confidence threshold for synthetic mode. AP mode uses threshold=0.01 for full PR curve.",
    ),
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(None, "--out", help="Write JSON results here."),
    unload_between_models: bool = typer.Option(
        True,
        "--unload-between-models/--no-unload",
        help="Flush GPU caches between models to prevent VRAM accumulation.",
    ),
    isolate_process: bool = typer.Option(
        False,
        "--isolate-process",
        help="Run each model in a child process for full CUDA context isolation.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Competitiveness benchmark: honest head-to-head comparison.

    Synthetic mode (default, no --dataset): latency + detection health on synthetic images.
    Dataset mode (--dataset yolo:<path>): real AP50/mAP50:95 from annotated ground truth.

    AP is computed with COCO-style 101-point interpolated PR curve.
    --unload-between-models flushes GPU cache between models (default: on).
    --isolate-process runs each model in a separate subprocess for full VRAM isolation.
    Honest by design: if YOLO beats VisionServeX models, this command will say so.
    """

    from visionservex.registry import default_registry

    model_ids = [m.strip() for m in models.split(",") if m.strip()]
    reg = default_registry()

    # Validate all model IDs before running
    validated: list[str] = []
    for mid in model_ids:
        if mid.startswith("ultralytics:"):
            validated.append(mid)
            continue
        try:
            entry = reg.get(mid)
            if entry.task != "detect":
                if not json_:
                    console.print(f"[yellow]skip[/yellow] {mid}: task={entry.task} (not detect)")
                continue
        except Exception as exc:
            if not json_:
                console.print(f"[yellow]skip[/yellow] {mid}: {exc}")
            continue
        validated.append(mid)

    if not validated:
        if not json_:
            console.print("[red]No valid detection models to benchmark.[/red]")
        raise typer.Exit(1)

    # ---- Dataset mode: real AP/mAP ----
    dataset_str = (dataset or "synthetic").strip()
    if dataset_str not in ("synthetic", "") and not dataset_str.startswith("synthetic"):
        if isolate_process:
            _run_ap_benchmark_isolated(validated, dataset_str, max_images, device, out, json_)
        else:
            _run_ap_benchmark(validated, dataset_str, max_images, device, out, json_)
        return

    # Generate synthetic test images
    test_images = _make_test_images(max_images)

    all_results = []
    for mid in validated:
        if not json_:
            console.print(f"  benchmarking [cyan]{mid}[/cyan] ...", end=" ")

        result = _run_competitiveness_model(mid, test_images, threshold=threshold, device=device)
        all_results.append(result)

        if unload_between_models:
            from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache

            clear_torch_cuda_cache()

        if not json_:
            st = result.get("status", "?")
            if st == "ok":
                console.print(
                    f"[green]ok[/green] "
                    f"p50={result.get('latency_p50_ms')}ms "
                    f"avg_detections={result.get('avg_detections'):.1f} "
                    f"zero_det_rate={result.get('zero_detection_rate'):.0%}"
                )
            else:
                console.print(
                    f"[red]{st}[/red] {result.get('error', result.get('skip_reason', ''))[:80]}"
                )

    # Add honest conclusion
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    conclusion = _generate_conclusion(ok_results)
    payload = {
        "benchmark_type": "competitiveness_latency_and_detection_health",
        "note": "AP50/mAP require ground-truth annotations (not computed here). Results show latency and detection statistics only.",
        "images_tested": max_images,
        "threshold": threshold,
        "device": device,
        "models": all_results,
        "conclusion": conclusion,
    }

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        import json

        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if not json_:
            console.print(f"\nResults written to {out}")

    if json_:
        import json

        typer.echo(json.dumps(payload, indent=2, default=str))
        return

    if ok_results:
        table = Table(title="Competitiveness Benchmark (latency + detection health)")
        for col in (
            "Model",
            "Category",
            "P50 ms",
            "P95 ms",
            "Avg Dets",
            "Zero-Det%",
            "Invalid%",
            "Status",
        ):
            table.add_column(col)
        for r in all_results:
            st = r.get("status", "?")
            if st == "ok":
                category = r.get("model_category", "-") or "-"
                zero_pct = f"{r.get('zero_detection_rate', 0):.0%}"
                inv_pct = f"{r.get('invalid_box_rate', 0):.0%}"
                table.add_row(
                    r["model_id"],
                    category,
                    str(r.get("latency_p50_ms", "?")),
                    str(r.get("latency_p95_ms", "?")),
                    f"{r.get('avg_detections', 0):.1f}",
                    zero_pct,
                    inv_pct,
                    "[green]ok[/green]",
                )
            else:
                table.add_row(r["model_id"], "-", "-", "-", "-", "-", "-", f"[red]{st}[/red]")
        console.print(table)

    console.print(f"\n[bold]Conclusion:[/bold] {conclusion}")
    console.print(
        "\n[dim]Note: This benchmark measures latency and output health, not AP50/mAP. "
        "Do not use these results alone to judge model accuracy. "
        "If YOLO wins on latency or detection count, that is an honest result.[/dim]"
    )


def _make_test_images(n: int) -> list:
    """Generate diverse synthetic test images."""
    from PIL import Image, ImageDraw

    images = []
    colors = [
        (200, 100, 80),
        (80, 150, 200),
        (120, 200, 80),
        (200, 200, 60),
        (180, 80, 180),
        (60, 180, 180),
        (220, 160, 80),
        (80, 80, 200),
    ]
    sizes = [(640, 480), (800, 600), (1280, 720), (480, 360), (640, 640)]
    for i in range(n):
        size = sizes[i % len(sizes)]
        img = Image.new("RGB", size, color=(200, 210, 220))
        draw = ImageDraw.Draw(img)
        c = colors[i % len(colors)]
        x1, y1 = size[0] // 8, size[1] // 8
        x2, y2 = size[0] * 3 // 4, size[1] * 3 // 4
        draw.rectangle([x1, y1, x2, y2], outline=c, width=3, fill=c)
        if i % 3 == 0:
            draw.ellipse(
                [x2 // 4, y2 // 4, x2 * 3 // 4, y2 * 3 // 4],
                outline=(200, 50, 50),
                width=2,
                fill=(220, 100, 100),
            )
        images.append(img)
    return images


def _run_competitiveness_model(
    model_id: str,
    images: list,
    *,
    threshold: float,
    device: str,
) -> dict:
    """Run a single model on all test images and collect statistics."""
    import statistics
    import time

    from visionservex import VisionModel
    from visionservex.core.results import DetectionResult
    from visionservex.engines.base import MissingDependencyError
    from visionservex.registry import default_registry
    from visionservex.runtime.downloads import DownloadError, ManualDownloadRequired

    result: dict = {"model_id": model_id}

    # Handle ultralytics: prefix
    if model_id.startswith("ultralytics:"):
        return _run_ultralytics_model(model_id, images, threshold=threshold, device=device)

    try:
        entry = default_registry().get(model_id)
        result["model_category"] = entry.model_category
        result["implementation_status"] = entry.implementation_status
    except Exception:
        pass

    try:
        model = VisionModel(model_id, device=device)
        model._ensure_loaded()

        latencies = []
        all_det_counts = []
        invalid_boxes = 0
        total_boxes = 0

        for img in images:
            t0 = time.perf_counter()
            pred = model.predict(img, threshold=threshold)
            latencies.append((time.perf_counter() - t0) * 1000)

            if isinstance(pred, DetectionResult):
                n = len(pred.detections)
                all_det_counts.append(n)
                for det in pred.detections:
                    total_boxes += 1
                    b = det.box
                    w, h = img.size
                    if b.x1 < 0 or b.y1 < 0 or b.x2 > w or b.y2 > h or b.x1 >= b.x2 or b.y1 >= b.y2:
                        invalid_boxes += 1
            else:
                all_det_counts.append(0)

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
        zero_rate = sum(1 for c in all_det_counts if c == 0) / max(len(all_det_counts), 1)
        invalid_rate = invalid_boxes / max(total_boxes, 1)

        result.update(
            {
                "status": "ok",
                "images_tested": len(images),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
                "avg_detections": round(statistics.mean(all_det_counts), 2),
                "min_detections": min(all_det_counts),
                "max_detections": max(all_det_counts),
                "zero_detection_rate": round(zero_rate, 4),
                "invalid_box_rate": round(invalid_rate, 4),
                "total_boxes": total_boxes,
                "invalid_boxes": invalid_boxes,
            }
        )
    except (MissingDependencyError, ManualDownloadRequired, DownloadError) as exc:
        result["status"] = "skip"
        result["skip_reason"] = str(exc)[:160]
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:200]
    return result


def _run_ultralytics_model(
    model_id: str,
    images: list,
    *,
    threshold: float,
    device: str,
) -> dict:
    """Run ultralytics model for baseline comparison."""
    import statistics
    import time

    parts = model_id.split(":", 1)
    yolo_name = parts[1] if len(parts) > 1 else "yolo11n"
    result: dict = {"model_id": model_id, "model_category": "baseline_yolo"}

    try:
        from ultralytics import YOLO  # type: ignore

        yolo = YOLO(yolo_name)
        latencies = []
        det_counts = []

        for img in images:
            t0 = time.perf_counter()
            preds = yolo(
                img, conf=threshold, verbose=False, device=device if device != "auto" else None
            )
            latencies.append((time.perf_counter() - t0) * 1000)
            if preds and hasattr(preds[0], "boxes"):
                det_counts.append(len(preds[0].boxes))
            else:
                det_counts.append(0)

        sorted_lat = sorted(latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))]
        zero_rate = sum(1 for c in det_counts if c == 0) / max(len(det_counts), 1)
        result.update(
            {
                "status": "ok",
                "images_tested": len(images),
                "latency_p50_ms": round(p50, 2),
                "latency_p95_ms": round(p95, 2),
                "avg_detections": round(statistics.mean(det_counts), 2),
                "zero_detection_rate": round(zero_rate, 4),
                "invalid_box_rate": 0.0,
                "total_boxes": sum(det_counts),
                "invalid_boxes": 0,
                "implementation_status": "ultralytics_package",
            }
        )
    except ImportError:
        result["status"] = "skip"
        result["skip_reason"] = "ultralytics not installed; pip install ultralytics"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:200]
    return result


def _generate_conclusion(results: list[dict]) -> str:
    if not results:
        return "No models ran successfully. Check dependencies with `visionservex doctor`."
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return "No successful runs. Check model availability and dependencies."
    fastest = min(ok, key=lambda r: r.get("latency_p50_ms", 99999))
    most_dets = max(ok, key=lambda r: r.get("avg_detections", 0))
    parts = [
        f"Fastest model (P50 latency): {fastest['model_id']} at {fastest.get('latency_p50_ms')} ms.",
    ]
    if most_dets["model_id"] != fastest["model_id"]:
        parts.append(
            f"Most detections per image: {most_dets['model_id']} "
            f"(avg {most_dets.get('avg_detections'):.1f} boxes — higher may mean better recall on real images, "
            f"or more false positives on synthetics)."
        )
    zero_det = [r for r in ok if r.get("zero_detection_rate", 0) > 0.5]
    if zero_det:
        ids = ", ".join(r["model_id"] for r in zero_det)
        parts.append(
            f"WARNING: {ids} returned zero detections on >50% of synthetic images. "
            "This may indicate a postprocessing issue, low threshold, or that the model "
            "correctly finds nothing in synthetic images. Run debug-output to diagnose."
        )
    parts.append(
        "NOTE: synthetic images are not real photos. These results show latency and output "
        "health only, not accuracy. Run on real COCO images for AP50/mAP comparison."
    )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Process-isolated benchmark (each model in child process)
# ---------------------------------------------------------------------------


def _child_model_eval(
    model_id: str, dataset_str: str, max_images: int, device: str, result_file: str
) -> None:
    """Run in a child process: load model, evaluate, write JSON to result_file."""
    import json as _json
    import sys

    try:
        from pathlib import Path as _Path

        from visionservex.runtime.evaluation import (
            load_coco_json,
            load_yolo_format,
            run_model_on_dataset,
        )

        if dataset_str.startswith("yolo:"):
            samples, _ = load_yolo_format(_Path(dataset_str[5:]), max_images=max_images)
            dname = f"yolo:{_Path(dataset_str[5:]).name}"
        elif dataset_str.startswith("coco-json:"):
            parts = dataset_str[10:].split(":", 1)
            samples, _ = load_coco_json(_Path(parts[0]), _Path(parts[1]), max_images=max_images)
            dname = f"coco-json:{_Path(parts[1]).stem}"
        else:
            samples, _ = load_yolo_format(_Path(dataset_str), max_images=max_images)
            dname = _Path(dataset_str).name

        result = run_model_on_dataset(model_id, samples, device=device, dataset_name=dname)
        from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache

        clear_torch_cuda_cache()
        with open(result_file, "w", encoding="utf-8") as f:
            _json.dump(result.to_dict(), f)
    except Exception as exc:
        with open(result_file, "w", encoding="utf-8") as f:
            _json.dump({"status": "error", "model_id": model_id, "error": str(exc)[:200]}, f)
        sys.exit(1)


def _run_ap_benchmark_isolated(
    model_ids: list[str],
    dataset_str: str,
    max_images: int,
    device: str,
    out: Path | None,
    json_: bool,
) -> None:
    """Run AP benchmark with each model in a separate child process."""
    import json as _json
    import multiprocessing
    import tempfile

    if not json_:
        console.print(
            f"[bold]AP benchmark (process-isolated):[/bold] {len(model_ids)} models | device={device}"
        )
        console.print(
            "[dim]Each model runs in a child process — CUDA context fully released after each.[/dim]\n"
        )

    all_results = []
    for mid in model_ids:
        if not json_:
            console.print(f"  [cyan]{mid}[/cyan] (isolated) ...", end=" ")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            result_file = tf.name

        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_child_model_eval,
            args=(mid, dataset_str, max_images, device, result_file),
            daemon=False,
        )
        proc.start()
        proc.join(timeout=600)

        if proc.exitcode is None:
            proc.terminate()
            result = {"status": "error", "model_id": mid, "error": "child process timed out (600s)"}
        elif proc.exitcode != 0:
            try:
                with open(result_file, encoding="utf-8") as f:
                    result = _json.load(f)
            except Exception:
                result = {
                    "status": "error",
                    "model_id": mid,
                    "error": f"child process exit code {proc.exitcode}",
                }
        else:
            try:
                with open(result_file, encoding="utf-8") as f:
                    result = _json.load(f)
            except Exception:
                result = {
                    "status": "error",
                    "model_id": mid,
                    "error": "could not read child output",
                }

        all_results.append(result)
        if not json_:
            st = result.get("status", "?")
            if st == "ok":
                console.print(
                    f"[green]ok[/green] AP50={result.get('ap50', '?'):.3f} "
                    f"mAP50:95={result.get('map50_95', '?'):.3f}"
                )
            else:
                console.print(f"[red]{st}[/red] {result.get('error', '')[:60]}")

    from visionservex.runtime.evaluation import EvaluationResult, generate_honest_conclusion

    ok_results = []
    for r in all_results:
        if r.get("status") == "ok":
            try:
                er = EvaluationResult(
                    model_id=r["model_id"],
                    dataset=r.get("dataset", ""),
                    n_images=r.get("n_images", 0),
                    ap50=r.get("ap50", 0.0),
                    map50_95=r.get("map50_95", 0.0),
                    precision=r.get("precision", 0.0),
                    recall=r.get("recall", 0.0),
                    f1=r.get("f1", 0.0),
                    latency_p50_ms=r.get("latency_p50_ms", 0.0),
                    latency_p95_ms=r.get("latency_p95_ms", 0.0),
                    n_classes_with_gt=r.get("n_classes_with_gt", 0),
                )
                ok_results.append(er)
            except Exception:
                pass

    conclusion = generate_honest_conclusion(ok_results)
    payload = {
        "benchmark_type": "competitiveness_ap_isolated_process",
        "dataset": dataset_str,
        "n_models": len(model_ids),
        "models": all_results,
        "conclusion": conclusion,
    }

    if out:
        op = out.with_suffix(".json") if out.suffix != ".json" else out
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if not json_:
            console.print(f"\nResults written to {op}")

    if json_:
        typer.echo(_json.dumps(payload, indent=2, default=str))
    else:
        console.print(f"\n[bold]Conclusion:[/bold] {conclusion}")


# ---------------------------------------------------------------------------
# Real AP/mAP benchmark (dataset mode)
# ---------------------------------------------------------------------------


def _run_ap_benchmark(
    model_ids: list[str],
    dataset_str: str,
    max_images: int,
    device: str,
    out: Path | None,
    json_: bool,
) -> None:
    """Run real AP/mAP evaluation with ground-truth annotations."""
    from visionservex.runtime.evaluation import (
        generate_honest_conclusion,
        load_coco_json,
        load_yolo_format,
        run_model_on_dataset,
    )

    # Parse dataset string
    samples = None
    dataset_name = "unknown"

    if dataset_str.startswith("yolo:"):
        yolo_path = Path(dataset_str[5:])
        if not yolo_path.exists():
            if not json_:
                console.print(f"[red]YOLO dataset path not found: {yolo_path}[/red]")
                console.print(
                    "[dim]Tip: download COCO128 with: "
                    'python -c "from ultralytics.utils import DATASETS_DIR; '
                    'import ultralytics; ultralytics.checks()"[/dim]'
                )
            raise typer.Exit(1)
        if not json_:
            console.print(f"Loading YOLO dataset from {yolo_path} (max {max_images} images)...")
        samples, _ = load_yolo_format(yolo_path, max_images=max_images)
        dataset_name = f"yolo:{yolo_path.name}"

    elif dataset_str.startswith("coco-json:"):
        parts = dataset_str[10:].split(":", 1)
        if len(parts) != 2:
            if not json_:
                console.print("[red]coco-json format: coco-json:<images_dir>:<ann_file>[/red]")
            raise typer.Exit(1)
        images_dir = Path(parts[0])
        ann_file = Path(parts[1])
        if not images_dir.exists() or not ann_file.exists():
            if not json_:
                console.print(
                    f"[red]images_dir={images_dir} or ann_file={ann_file} not found.[/red]"
                )
            raise typer.Exit(1)
        if not json_:
            console.print(f"Loading COCO JSON dataset: {ann_file.name}...")
        samples, _ = load_coco_json(images_dir, ann_file, max_images=max_images)
        dataset_name = f"coco-json:{ann_file.stem}"

    elif dataset_str == "coco128":
        if not json_:
            console.print(
                "[yellow]coco128 shortcut: please provide the path explicitly.[/yellow]\n"
                "Example: --dataset yolo:/path/to/coco128\n"
                "Get COCO128: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
            )
        raise typer.Exit(1)

    else:
        # Treat as YOLO path directly
        yolo_path = Path(dataset_str)
        if not yolo_path.exists():
            if not json_:
                console.print(
                    f"[red]Dataset path not found: {yolo_path}[/red]\n"
                    "Specify dataset with: --dataset yolo:<path> or --dataset coco-json:<img_dir>:<ann_file>"
                )
            raise typer.Exit(1)
        samples, _ = load_yolo_format(yolo_path, max_images=max_images)
        dataset_name = yolo_path.name

    if not samples:
        if not json_:
            console.print("[red]No images found in dataset.[/red]")
        raise typer.Exit(1)

    if not json_:
        console.print(
            f"[bold]AP benchmark:[/bold] {len(samples)} images | "
            f"{len(model_ids)} models | device={device}"
        )
        console.print("[dim]Using threshold=0.01 for full PR curve (AP computation).[/dim]\n")

    all_results = []
    for mid in model_ids:
        if mid.startswith("ultralytics:"):
            r = _run_ultralytics_ap(mid, samples, device=device)
        else:
            if not json_:
                console.print(f"  evaluating [cyan]{mid}[/cyan] ...", end=" ")
            r = run_model_on_dataset(mid, samples, device=device, dataset_name=dataset_name)
            if not json_:
                st = r.status
                if st == "ok":
                    console.print(
                        f"[green]ok[/green] AP50={r.ap50:.3f} mAP50:95={r.map50_95:.3f} "
                        f"P50={r.latency_p50_ms:.1f}ms"
                    )
                else:
                    console.print(f"[red]{st}[/red] {r.error[:60]}")
        all_results.append(r)

    ok_results = [r for r in all_results if r.status == "ok"]

    conclusion = generate_honest_conclusion(ok_results)

    payload = {
        "benchmark_type": "competitiveness_with_ap_evaluation",
        "dataset": dataset_name,
        "n_images": len(samples),
        "device": device,
        "ap_method": "COCO-style 101-point interpolated PR curve",
        "models": [r.to_dict() if hasattr(r, "to_dict") else r for r in all_results],
        "conclusion": conclusion,
    }

    if out:
        out_json = out.with_suffix(".json") if out.suffix != ".json" else out
        out_json.parent.mkdir(parents=True, exist_ok=True)
        import json as _json

        out_json.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if not json_:
            console.print(f"\nResults written to {out_json}")

        # Also write summary CSV
        try:
            import csv

            csv_path = out.with_suffix(".csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "model_id",
                        "ap50",
                        "map50_95",
                        "precision",
                        "recall",
                        "f1",
                        "latency_p50_ms",
                        "latency_p95_ms",
                        "n_images",
                        "n_no_detection",
                        "n_classes",
                        "status",
                    ]
                )
                for r in all_results:
                    if hasattr(r, "ap50"):
                        writer.writerow(
                            [
                                r.model_id,
                                r.ap50,
                                r.map50_95,
                                r.precision,
                                r.recall,
                                r.f1,
                                r.latency_p50_ms,
                                r.latency_p95_ms,
                                r.n_images,
                                r.n_no_detection,
                                r.n_classes_with_gt,
                                r.status,
                            ]
                        )
            if not json_:
                console.print(f"Summary CSV: {csv_path}")
        except Exception:
            pass

    if json_:
        import json as _json

        typer.echo(_json.dumps(payload, indent=2, default=str))
        return

    # Print results table
    if ok_results:
        table = Table(title="AP Benchmark Results")
        for col in ("Model", "AP50", "mAP50:95", "Precision", "Recall", "F1", "P50 ms", "Classes"):
            table.add_column(col)
        for r in sorted(
            all_results, key=lambda x: -(x.ap50 if hasattr(x, "ap50") and x.status == "ok" else -1)
        ):
            if hasattr(r, "ap50") and r.status == "ok":
                table.add_row(
                    r.model_id,
                    f"{r.ap50:.3f}",
                    f"{r.map50_95:.3f}",
                    f"{r.precision:.3f}",
                    f"{r.recall:.3f}",
                    f"{r.f1:.3f}",
                    f"{r.latency_p50_ms:.1f}",
                    str(r.n_classes_with_gt),
                )
            else:
                status = getattr(r, "status", "error")
                model_id = getattr(r, "model_id", str(r))
                table.add_row(model_id, "-", "-", "-", "-", "-", "-", f"[red]{status}[/red]")
        console.print(table)

    console.print(f"\n[bold]Conclusion:[/bold] {conclusion}")
    console.print(
        "\n[dim]AP computed with COCO-style 101-point interpolation. "
        "mAP50:95 sweeps IoU 0.50→0.95 in 0.05 steps. "
        "Results depend on dataset quality and class label matching. "
        "Small datasets (<100 images) have high variance.[/dim]"
    )


def _run_ultralytics_ap(
    model_id: str,
    samples: list,
    *,
    device: str,
) -> object:
    """Run Ultralytics model for AP comparison. Returns an EvaluationResult-like dict."""
    import time

    from visionservex.runtime.evaluation import (
        DetectionEvaluator,
        EvaluationResult,
        PerClassMetric,
    )

    parts = model_id.split(":", 1)
    yolo_name = parts[1] if len(parts) > 1 else "yolo11n"

    try:
        from PIL import Image as _PIL
        from ultralytics import YOLO  # type: ignore

        yolo = YOLO(yolo_name)
        evaluator = DetectionEvaluator()
        latencies: list[float] = []
        n_no_det = 0

        for sample in samples:
            try:
                img = _PIL.open(sample.image_path).convert("RGB")
                t0 = time.perf_counter()
                preds = yolo(
                    img, conf=0.01, verbose=False, device=device if device != "auto" else None
                )
                latencies.append((time.perf_counter() - t0) * 1000)

                pred_boxes: list[list[float]] = []
                pred_scores: list[float] = []
                pred_classes: list[str] = []

                if preds and hasattr(preds[0], "boxes") and preds[0].boxes is not None:
                    boxes = preds[0].boxes
                    xyxy = boxes.xyxy.tolist() if hasattr(boxes.xyxy, "tolist") else []
                    confs = boxes.conf.tolist() if hasattr(boxes.conf, "tolist") else []
                    cls_ids = boxes.cls.tolist() if hasattr(boxes.cls, "tolist") else []
                    for box, conf, cls_id in zip(xyxy, confs, cls_ids, strict=False):
                        from visionservex.runtime.evaluation import COCO80_CLASSES

                        cls_name = (
                            COCO80_CLASSES[int(cls_id)]
                            if int(cls_id) < len(COCO80_CLASSES)
                            else f"class_{int(cls_id)}"
                        )
                        pred_boxes.append(box[:4])
                        pred_scores.append(float(conf))
                        pred_classes.append(cls_name)

                if not pred_boxes:
                    n_no_det += 1

                evaluator.add_image(
                    pred_boxes, pred_scores, pred_classes, sample.gt_boxes, sample.gt_classes
                )
            except Exception:
                evaluator.add_image([], [], [], sample.gt_boxes, sample.gt_classes)
                n_no_det += 1

        s_lat = sorted(latencies) if latencies else [0.0]
        lat_p50 = s_lat[len(s_lat) // 2]
        lat_p95 = s_lat[min(len(s_lat) - 1, int(len(s_lat) * 0.95))]

        metrics = evaluator.compute_map50_95()
        per_class = [
            PerClassMetric(
                class_name=d["class"],
                ap50=d["ap"],
                precision=d["precision"],
                recall=d["recall"],
                f1=d["f1"],
                n_gt=d["n_gt"],
                n_pred=d["n_pred"],
            )
            for d in metrics["per_class"]
        ]

        return EvaluationResult(
            model_id=model_id,
            dataset="user_dataset",
            n_images=len(samples),
            ap50=metrics["map50"],
            map50_95=metrics["map50_95"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            per_class=per_class,
            latency_p50_ms=round(lat_p50, 2),
            latency_p95_ms=round(lat_p95, 2),
            n_no_detection=n_no_det,
            n_classes_with_gt=metrics["n_classes_with_gt"],
            device=str(device),
            status="ok",
        )

    except ImportError:
        from visionservex.runtime.evaluation import EvaluationResult

        return EvaluationResult(
            model_id=model_id,
            dataset="user_dataset",
            n_images=0,
            ap50=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status="skip",
            error="ultralytics not installed; pip install ultralytics",
        )
    except Exception as exc:
        from visionservex.runtime.evaluation import EvaluationResult

        return EvaluationResult(
            model_id=model_id,
            dataset="user_dataset",
            n_images=0,
            ap50=0.0,
            map50_95=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            status="error",
            error=str(exc)[:200],
        )


# ---------------------------------------------------------------------------
# Non-detection benchmark stubs (honest roadmap)
# ---------------------------------------------------------------------------

_NOT_IMPLEMENTED_TPLS: dict[str, dict] = {
    "segmentation": {
        "task": "instance segmentation",
        "annotation_format": "COCO JSON with 'segmentation' field (polygon or RLE masks)",
        "recommended_dataset": "COCO val2017 (5000 images), COCO-128-seg",
        "metrics": ["mask AP50", "mask AP50:95", "box AP50", "boundary IoU"],
        "models": ["rfdetr-seg-small", "rfdetr-seg-medium", "oneformer-swin-large"],
        "roadmap": "v1.4.0 — mask IoU matching and COCO evaluator integration planned.",
    },
    "classification": {
        "task": "image classification",
        "annotation_format": "folder structure (class_name/img.jpg) or CSV (path,label)",
        "recommended_dataset": "ImageNet-1k val set (50000 images), custom classification dataset",
        "metrics": ["top-1 accuracy", "top-5 accuracy", "confusion matrix", "per-class recall"],
        "models": ["swinv2-tiny", "swinv2-base", "swinv2-large"],
        "roadmap": "v1.4.0 — classification evaluator planned.",
    },
    "open-vocab": {
        "task": "open-vocabulary detection",
        "annotation_format": "COCO JSON with category names matching prompt",
        "recommended_dataset": "LVIS v1 val (19809 images), COCO val2017 zero-shot",
        "metrics": [
            "AP50 (class-aware)",
            "AP rare/common/frequent (LVIS)",
            "prompt-conditioned AP",
        ],
        "models": ["grounding-dino-swin-b", "grounding-dino-tiny"],
        "roadmap": "v1.4.0 — zero-shot AP evaluation with prompt templates planned.",
    },
    "pose": {
        "task": "pose estimation (keypoints)",
        "annotation_format": "COCO JSON with 'keypoints' field",
        "recommended_dataset": "COCO val2017 person keypoints",
        "metrics": ["OKS AP50", "OKS AP50:95", "per-keypoint PCK"],
        "models": ["rtmpose-s", "rtmpose-m", "rtmpose-l"],
        "roadmap": "v1.4.0 — OKS AP evaluation planned when RTMPose native path is wired.",
    },
    "obb": {
        "task": "oriented bounding box detection",
        "annotation_format": "DOTA format (8-point polygon per box) or COCO-OBB JSON",
        "recommended_dataset": "DOTA v1.0 val, HRSC2016",
        "metrics": ["rotated IoU AP50", "rotated IoU AP50:95"],
        "models": ["rtmdet-r-s", "rtmdet-r2-s"],
        "roadmap": "v1.4.0 — rotated IoU matching planned when RTMDet-R native path is wired.",
    },
}


def _benchmark_not_implemented(task_key: str, json_: bool) -> None:
    info = _NOT_IMPLEMENTED_TPLS.get(task_key, {})
    payload = {
        "status": "BENCHMARK_NOT_IMPLEMENTED",
        "task": info.get("task", task_key),
        "message": f"The {task_key} benchmark evaluator is not yet implemented in VisionServeX.",
        "annotation_format": info.get("annotation_format", "see task-specific format"),
        "recommended_dataset": info.get("recommended_dataset", "N/A"),
        "metrics": info.get("metrics", []),
        "models": info.get("models", []),
        "roadmap": info.get("roadmap", "roadmap TBD"),
        "detection_note": (
            "Detection AP/mAP is implemented: use 'visionservex benchmark benchmark-competitiveness "
            "--dataset yolo:<path>'"
        ),
    }
    if json_:
        import json as _json

        typer.echo(_json.dumps(payload, indent=2))
    else:
        from rich.panel import Panel

        console.print(
            Panel.fit(
                f"[yellow]BENCHMARK_NOT_IMPLEMENTED: {task_key}[/yellow]",
                border_style="yellow",
            )
        )
        console.print(f"Task: {payload['task']}")
        console.print(f"Annotation format: {payload['annotation_format']}")
        console.print(f"Recommended dataset: {payload['recommended_dataset']}")
        console.print(f"Planned metrics: {', '.join(payload['metrics'])}")
        console.print(f"Expected models: {', '.join(payload['models'])}")
        console.print(f"Roadmap: {payload['roadmap']}")
        console.print(
            "\n[dim]Note: Detection AP/mAP IS implemented. "
            "Run: visionservex benchmark benchmark-competitiveness --dataset yolo:<path>[/dim]"
        )
    raise typer.Exit(2)


@app.command(
    "benchmark-segmentation",
    help="Mask AP benchmark for instance segmentation models.",
)
def benchmark_segmentation(
    models: str = typer.Option(
        "rfdetr-seg-small",
        "--models",
        help="Comma-separated segmentation model IDs.",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        help="Dataset: 'synthetic' (latency only), 'coco-json:<img_dir>:<ann_file>'",
    ),
    max_images: int = typer.Option(20, "--max-images", min=1, max=2000),
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    unload_between_models: bool = typer.Option(True, "--unload-between-models/--no-unload"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Mask AP50/mAP50:95 benchmark for instance segmentation models.

    Requires COCO JSON annotations with polygon/RLE masks for real AP.
    Without --dataset, runs in synthetic latency-only mode.
    """
    from visionservex.registry import default_registry

    model_ids = [m.strip() for m in models.split(",") if m.strip()]

    # Validate all model IDs
    validated: list[str] = []
    for mid in model_ids:
        try:
            entry = default_registry().get(mid)
            if entry.task not in ("segment", "grounded_segment", "foundation_segment"):
                if not json_:
                    console.print(f"[yellow]skip[/yellow] {mid}: task={entry.task} (not segment)")
                continue
        except Exception as exc:
            if not json_:
                console.print(f"[yellow]skip[/yellow] {mid}: {exc}")
            continue
        validated.append(mid)

    if not validated:
        if not json_:
            console.print("[red]No valid segmentation models to benchmark.[/red]")
        raise typer.Exit(1)

    dataset_str = (dataset or "synthetic").strip()

    # Synthetic mode: latency + detection count only
    if dataset_str in ("synthetic", ""):
        test_images = _make_test_images(max_images)
        all_results = []
        for mid in validated:
            if not json_:
                console.print(f"  [cyan]{mid}[/cyan] (synthetic latency) ...", end=" ")
            r = _run_competitiveness_model(mid, test_images, threshold=0.3, device=device)
            all_results.append(r)
            if unload_between_models:
                from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache

                clear_torch_cuda_cache()
            if not json_:
                st = r.get("status", "?")
                if st == "ok":
                    console.print(f"[green]ok[/green] P50={r.get('latency_p50_ms')}ms")
                else:
                    console.print(f"[red]{st}[/red]")

        payload = {
            "benchmark_type": "segmentation_latency_synthetic",
            "note": "Synthetic mode — no mask GT. Use --dataset coco-json:... for mask AP.",
            "models": all_results,
        }
        if json_:
            import json as _json

            typer.echo(_json.dumps(payload, indent=2, default=str))
        else:
            console.print(
                "[dim]Synthetic mode: latency + detection health only (no mask AP). Provide --dataset for real AP.[/dim]"
            )
        return

    # COCO JSON mode: real mask AP
    if dataset_str.startswith("coco-json:"):
        parts = dataset_str[10:].split(":", 1)
        if len(parts) != 2:
            if not json_:
                console.print("[red]coco-json format: coco-json:<images_dir>:<ann_file>[/red]")
            raise typer.Exit(1)
        images_dir = Path(parts[0])
        ann_file = Path(parts[1])
        if not images_dir.exists() or not ann_file.exists():
            if not json_:
                console.print(
                    f"[red]images_dir={images_dir} or ann_file={ann_file} not found.[/red]"
                )
            raise typer.Exit(1)

        from visionservex.runtime.segmentation_eval import (
            load_coco_segmentation_json,
            run_segmentation_evaluation,
        )

        if not json_:
            console.print(
                f"Loading segmentation dataset: {ann_file.name} (max {max_images} images)..."
            )
        samples, _ = load_coco_segmentation_json(images_dir, ann_file, max_images=max_images)
        if not samples:
            if not json_:
                console.print("[red]No images found in dataset.[/red]")
            raise typer.Exit(1)

        all_results = []
        for mid in validated:
            if not json_:
                console.print(f"  evaluating [cyan]{mid}[/cyan] ...", end=" ")
            result = run_segmentation_evaluation(mid, samples, device=device)
            all_results.append(result.to_dict())
            if unload_between_models:
                from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache

                clear_torch_cuda_cache()
            if not json_:
                st = result.status
                if st == "ok":
                    console.print(
                        f"[green]ok[/green] "
                        f"mask_AP50={result.mask_ap50:.3f} "
                        f"mask_mAP50:95={result.mask_map50_95:.3f} "
                        f"P50={result.latency_p50_ms:.1f}ms"
                    )
                else:
                    console.print(f"[red]{st}[/red] {result.error[:60]}")

        payload = {
            "benchmark_type": "segmentation_mask_ap",
            "dataset": dataset_str,
            "n_images": len(samples),
            "models": all_results,
            "ap_method": "COCO-style 101-point interpolated mask IoU PR curve",
        }
        if out:
            import json as _json

            op = out.with_suffix(".json")
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(_json.dumps(payload, indent=2, default=str), encoding="utf-8")
            if not json_:
                console.print(f"\nResults written to {op}")
        if json_:
            import json as _json

            typer.echo(_json.dumps(payload, indent=2, default=str))
        else:
            from rich.table import Table

            table = Table(title="Segmentation Mask AP Results")
            for col in (
                "Model",
                "Mask AP50",
                "Mask mAP50:95",
                "Precision",
                "Recall",
                "Latency P50",
                "Status",
            ):
                table.add_column(col)
            for r in all_results:
                if r.get("status") == "ok":
                    table.add_row(
                        r["model_id"],
                        f"{r['mask_ap50']:.3f}",
                        f"{r['mask_map50_95']:.3f}",
                        f"{r['precision']:.3f}",
                        f"{r['recall']:.3f}",
                        f"{r['latency_p50_ms']:.1f} ms",
                        "[green]ok[/green]",
                    )
                else:
                    table.add_row(
                        r["model_id"], "-", "-", "-", "-", "-", f"[red]{r.get('status', '?')}[/red]"
                    )
            console.print(table)
            console.print(
                "\n[dim]Mask AP uses binary mask IoU (not box IoU). Do NOT compare with detection AP50.[/dim]"
            )
        return

    # Unknown dataset format
    if not json_:
        console.print(
            f"[red]Unknown dataset format: {dataset_str!r}. Use 'synthetic' or 'coco-json:<img>:<ann>'[/red]"
        )
    raise typer.Exit(1)


@app.command(
    "benchmark-classification",
    help="[PLANNED v1.4] Classification benchmark — returns BENCHMARK_NOT_IMPLEMENTED.",
)
def benchmark_classification(json_: bool = typer.Option(False, "--json")) -> None:
    """Honest stub: classification top-k benchmark is roadmap item for v1.4."""
    _benchmark_not_implemented("classification", json_)


@app.command(
    "benchmark-open-vocab",
    help="[PLANNED v1.4] Open-vocabulary detection benchmark — returns BENCHMARK_NOT_IMPLEMENTED.",
)
def benchmark_open_vocab(json_: bool = typer.Option(False, "--json")) -> None:
    """Honest stub: zero-shot AP benchmark is roadmap item for v1.4."""
    _benchmark_not_implemented("open-vocab", json_)


@app.command(
    "benchmark-pose",
    help="[PLANNED v1.4] Pose estimation (OKS AP) benchmark — returns BENCHMARK_NOT_IMPLEMENTED.",
)
def benchmark_pose(json_: bool = typer.Option(False, "--json")) -> None:
    """Honest stub: OKS AP benchmark is roadmap item for v1.4."""
    _benchmark_not_implemented("pose", json_)


@app.command(
    "benchmark-obb",
    help="[PLANNED v1.4] Oriented bounding box benchmark — returns BENCHMARK_NOT_IMPLEMENTED.",
)
def benchmark_obb(json_: bool = typer.Option(False, "--json")) -> None:
    """Honest stub: rotated IoU AP benchmark is roadmap item for v1.4."""
    _benchmark_not_implemented("obb", json_)


__all__ = ["app"]

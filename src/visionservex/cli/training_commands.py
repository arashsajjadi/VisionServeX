# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Training, export, val, and video/tracking capability commands.

These provide Ultralytics-like ergonomics without requiring Ultralytics.
Unsupported operations return structured errors, never raw tracebacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

# Separate typers so they can be mounted as sub-commands
training_app = typer.Typer(help="Training and fine-tuning capabilities.")
export_app = typer.Typer(help="Export capabilities and ONNX/TensorRT commands.")
video_app = typer.Typer(help="Video inference and tracking (roadmap/stubs).")


# ---------------------------------------------------------------------------
# Training capabilities
# ---------------------------------------------------------------------------


@training_app.command("capabilities", help="Show training/fine-tuning support by model.")
def training_capabilities(
    model_id: str | None = typer.Option(None, "--model", help="Specific model ID."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.core.model import _training_capabilities
    from visionservex.registry import default_registry

    if model_id:
        payload = _training_capabilities(model_id)
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            _print_training_info(payload)
        return

    # All model families
    families = {
        e.family
        for e in default_registry().list()
        if e.implementation_status in ("wired", "partial")
    }
    all_info = [_training_capabilities(f) for f in sorted(families)]

    if json_:
        typer.echo(json.dumps(all_info, indent=2))
        return

    table = Table(title="Training/fine-tuning capabilities")
    for col in ("Family", "Train", "Finetune", "Resume", "Export", "Notes"):
        table.add_column(col)
    for info in all_info:
        train = "[green]yes[/green]" if info.get("train_supported") else "[red]no[/red]"
        ft = "[green]yes[/green]" if info.get("finetune_supported") else "[red]no[/red]"
        resume = "[green]yes[/green]" if info.get("resume_supported") else "[red]no[/red]"
        exports = ", ".join(info.get("export_supported", [])) or "-"
        notes = (info.get("notes", "")[:60]) or "-"
        table.add_row(info.get("family", "-"), train, ft, resume, exports, notes)
    console.print(table)
    console.print(
        "\n[dim]For model-specific details: visionservex training capabilities --model MODEL[/dim]"
    )


def _print_training_info(info: dict) -> None:
    console.print(f"[bold]Training capabilities:[/bold] {info.get('model_id', '-')}")
    train = "[green]yes[/green]" if info.get("train_supported") else "[red]no[/red]"
    ft = "[green]yes[/green]" if info.get("finetune_supported") else "[red]no[/red]"
    resume = "[green]yes[/green]" if info.get("resume_supported") else "[red]no[/red]"
    console.print(f"  train_supported:    {train}")
    console.print(f"  finetune_supported: {ft}")
    console.print(f"  resume_supported:   {resume}")
    console.print(f"  export_supported:   {info.get('export_supported', [])}")
    console.print(f"  dataset_formats:    {info.get('supported_dataset_formats', [])}")
    if info.get("notes"):
        console.print(f"\n  [yellow]Note:[/yellow] {info['notes']}")
    if info.get("docs"):
        console.print(f"  [dim]Docs: {info['docs']}[/dim]")


# ---------------------------------------------------------------------------
# Export capabilities
# ---------------------------------------------------------------------------


@export_app.command("capabilities", help="Show ONNX/TensorRT/other export support by model.")
def export_capabilities(
    model_id: str | None = typer.Option(None, "--model"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.core.model import _export_capabilities
    from visionservex.registry import default_registry

    if model_id:
        payload = _export_capabilities(model_id)
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            _print_export_info(payload)
        return

    families = {
        e.family
        for e in default_registry().list()
        if e.implementation_status in ("wired", "partial")
    }
    all_info = [_export_capabilities(f) for f in sorted(families)]
    if json_:
        typer.echo(json.dumps(all_info, indent=2))
        return

    table = Table(title="Export capabilities")
    for col in ("Family", "ONNX", "TensorRT", "TorchScript", "HF save"):
        table.add_column(col)
    for info in all_info:
        row = [info.get("family", "-")]
        for key in ("onnx", "tensorrt", "torchscript", "hf_save_pretrained"):
            s = info.get(key, {}).get("status", "unsupported")
            color = {"supported": "green", "experimental": "yellow", "unsupported": "red"}.get(
                s, "grey50"
            )
            row.append(f"[{color}]{s}[/{color}]")
        table.add_row(*row)
    console.print(table)


def _print_export_info(info: dict) -> None:
    console.print(f"[bold]Export capabilities:[/bold] {info.get('model_id', '-')}")
    for fmt in ("onnx", "tensorrt", "torchscript", "openvino", "hf_save_pretrained"):
        entry = info.get(fmt)
        if entry:
            status = entry.get("status", "unsupported")
            color = {"supported": "green", "experimental": "yellow"}.get(status, "grey50")
            notes = entry.get("notes", "")[:80]
            console.print(f"  {fmt:25s}: [{color}]{status}[/{color}]  {notes}")


@export_app.command("export", help="Export a model to ONNX or another format.")
def export_model(
    model_id: str,
    format: str = typer.Option("onnx", "--format"),
    out: Path = typer.Option(..., "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.core.model import VisionModel, _export_capabilities

    capabilities = _export_capabilities(model_id)
    fmt_info = capabilities.get(format.lower(), {})
    status = fmt_info.get("status", "unsupported")

    if status == "unsupported":
        payload = {
            "status": "EXPORT_UNSUPPORTED",
            "model_id": model_id,
            "format": format,
            "reason": fmt_info.get("notes", f"Format '{format}' is not supported for {model_id}."),
            "supported_formats": [
                k
                for k, v in capabilities.items()
                if isinstance(v, dict) and v.get("status") not in ("unsupported", None)
            ],
            "hint": f"Run 'visionservex export capabilities --model {model_id}' to see supported formats.",
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]EXPORT_UNSUPPORTED:[/red] {format} for {model_id}")
            console.print(f"  Reason: {payload['reason']}")
        raise typer.Exit(2)

    try:
        model = VisionModel(model_id)
        path = model.export(format=format, output_path=out)
        payload = {"model_id": model_id, "format": format, "path": str(path), "status": "ok"}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[green]exported to {path}[/green]")
    except Exception as exc:
        payload = {
            "status": "EXPORT_FAILED",
            "model_id": model_id,
            "format": format,
            "error": str(exc)[:200],
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]export failed:[/red] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Train / finetune stubs
# ---------------------------------------------------------------------------


def _training_stub(model_id: str, operation: str, data: str | None, json_: bool) -> None:
    from visionservex.core.model import _training_capabilities

    cap = _training_capabilities(model_id)
    key = f"{operation}_supported"
    supported = cap.get(key, False)

    if not supported:
        payload = {
            "status": "TRAINING_NOT_SUPPORTED",
            "operation": operation,
            "model_id": model_id,
            "reason": cap.get("notes", f"{operation} is not supported for {model_id}."),
            "supported_alternatives": cap.get("export_supported", []),
            "docs": cap.get("docs", ""),
            "hint": f"Check visionservex training capabilities --model {model_id}",
        }
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[yellow]TRAINING_NOT_SUPPORTED:[/yellow] {operation} for {model_id}")
            console.print(f"  Reason: {cap.get('notes', '-')}")
            if cap.get("docs"):
                console.print(f"  Docs: {cap['docs']}")
        raise typer.Exit(2)


@training_app.command(
    "train", help="Train a model (returns TRAINING_NOT_SUPPORTED for most models)."
)
def train_model(
    model_id: str,
    data: str | None = typer.Option(None, "--data"),
    epochs: int = typer.Option(50, "--epochs"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Train a model. Most models return a structured TRAINING_NOT_SUPPORTED error."""
    _training_stub(model_id, "train", data, json_)


@training_app.command(
    "finetune", help="Fine-tune a model (returns structured error for most models)."
)
def finetune_model(
    model_id: str,
    data: str | None = typer.Option(None, "--data"),
    epochs: int = typer.Option(20, "--epochs"),
    device: str = typer.Option("auto", "--device"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Fine-tune a model. Most models return a structured TRAINING_NOT_SUPPORTED error."""
    _training_stub(model_id, "finetune", data, json_)


# ---------------------------------------------------------------------------
# Val command
# ---------------------------------------------------------------------------


@training_app.command("val", help="Evaluate model AP on an annotated dataset.")
def val_model(
    model_id: str,
    dataset: str = typer.Option(..., "--dataset", help="'yolo:<path>' or 'coco-json:<img>:<ann>'"),
    max_images: int = typer.Option(100, "--max-images"),
    device: str = typer.Option("auto", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Evaluate detection AP50/mAP50:95 on an annotated dataset."""
    from visionservex.core.model import VisionModel

    try:
        model = VisionModel(model_id, device=device)
        result = model.val(
            dataset=dataset, max_images=max_images, device=device, out=str(out) if out else None
        )
    except Exception as exc:
        payload = {"status": "VAL_FAILED", "model_id": model_id, "error": str(exc)[:200]}
        if json_:
            typer.echo(json.dumps(payload, indent=2))
        else:
            console.print(f"[red]val failed:[/red] {exc}")
        raise typer.Exit(1)

    status = result.get("status", "ok")
    if json_:
        typer.echo(json.dumps(result, indent=2, default=str))
        if status not in ("ok", None):
            raise typer.Exit(2)
        return

    if status not in ("ok", None):
        console.print(f"[yellow]{status}:[/yellow] {result.get('message', '-')}")
        if result.get("hint"):
            console.print(f"  hint: {result['hint']}")
        raise typer.Exit(2)

    console.print(f"[bold]Val results:[/bold] {model_id}")
    console.print(
        f"  AP50:       {result.get('ap50', '-'):.4f}"
        if isinstance(result.get("ap50"), float)
        else f"  AP50: {result.get('ap50', '-')}"
    )
    console.print(
        f"  mAP50:95:   {result.get('map50_95', '-'):.4f}"
        if isinstance(result.get("map50_95"), float)
        else f"  mAP50:95: {result.get('map50_95', '-')}"
    )
    console.print(
        f"  Precision:  {result.get('precision', '-'):.4f}"
        if isinstance(result.get("precision"), float)
        else ""
    )
    console.print(
        f"  Recall:     {result.get('recall', '-'):.4f}"
        if isinstance(result.get("recall"), float)
        else ""
    )
    console.print(f"  Latency P50: {result.get('latency_p50_ms', '-')} ms")
    if out:
        console.print(f"  Results: {out}.json")


# ---------------------------------------------------------------------------
# Video / tracking stubs
# ---------------------------------------------------------------------------


def _video_stub(operation: str, model_id: str, source: str | None, json_: bool) -> None:
    payload = {
        "status": f"{operation.upper()}_NOT_IMPLEMENTED",
        "operation": operation,
        "model_id": model_id,
        "message": f"Video {operation} is not implemented in VisionServeX.",
        "roadmap": "v1.5.0 — video inference and tracking planned.",
        "hint": (
            "For tracking with SAM2: video segmentation is experimental. "
            "For other models: use batch_predict() on extracted frames."
        ),
    }
    if json_:
        typer.echo(json.dumps(payload, indent=2))
    else:
        console.print(f"[yellow]{operation.upper()}_NOT_IMPLEMENTED:[/yellow] {model_id}")
        console.print(f"  Roadmap: {payload['roadmap']}")
        console.print(f"  Hint: {payload['hint']}")
    raise typer.Exit(2)


@video_app.command("predict", help="[ROADMAP v1.5] Video inference — returns NOT_IMPLEMENTED.")
def video_predict(
    model_id: str,
    source: str = typer.Argument(..., help="Video file or webcam/RTSP URL."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    _video_stub("video_predict", model_id, source, json_)


@video_app.command("track", help="[ROADMAP v1.5] Object tracking — returns NOT_IMPLEMENTED.")
def track_video(
    model_id: str,
    source: str = typer.Argument(..., help="Video file or webcam/RTSP URL."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    _video_stub("tracking", model_id, source, json_)


@video_app.command("stream", help="[ROADMAP v1.5] Live stream inference — returns NOT_IMPLEMENTED.")
def stream_source(
    model_id: str,
    source: str = typer.Option("webcam", "--source"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    _video_stub("streaming", model_id, source, json_)


__all__ = ["export_app", "training_app", "video_app"]

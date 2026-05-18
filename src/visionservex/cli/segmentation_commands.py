# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: `visionservex benchmark-segmentation` + `benchmark-promptable-segmentation`.

These CLIs frame the two protocols separately so VisionServeX never mixes
them. Both currently emit structured blockers for VSX/SAM rows until the
mask AP / promptable adapters are written. Ultralytics yolo*-seg models
were already benchmarked in v2.27 (see segmentation_auto_instance_400_v227.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app_auto = typer.Typer(
    help="v2.28.0: automatic instance segmentation benchmark.",
    no_args_is_help=True,
    invoke_without_command=True,
)
app_promptable = typer.Typer(
    help="v2.28.0: promptable segmentation benchmark (box-prompted SAM/SAM2).",
    no_args_is_help=True,
    invoke_without_command=True,
)
console = Console()


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = {"ok": "green", "expected_blocker": "yellow", "failed": "red"}.get(
        payload.get("status", ""), "white"
    )
    console.print(f"[{color}]{payload.get('code', '')}[/{color}]")


@app_auto.callback(invoke_without_command=True)
def benchmark_segmentation(
    dataset: str = typer.Option(..., "--dataset", help="coco-instance:ANNOTATIONS.json"),
    models: str = typer.Option(..., "--models", help="Comma-separated list."),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
) -> None:
    """v2.28.0: automatic instance segmentation.

    For Ultralytics yolo*-seg models, v2.27 already shipped a working
    runner; v2.28 surfaces the same protocol via this command. For VSX
    rfdetr-seg-* models, the mask-AP adapter is not yet written and the
    command returns ``RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN`` per model.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    for m in model_list:
        if "rfdetr-seg" in m.lower():
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN",
                    "task": "automatic_instance_segmentation",
                    "next_action": (
                        "verify rfdetr.RFDETRSeg().predict(IMG).mask schema, "
                        "then add COCO RLE encoder; see "
                        "reports/information_required_to_finish_v228.csv "
                        "issue RFDETR-SEG-SCHEMA"
                    ),
                    "evidence_artifact": "information_required_to_finish_v228.csv",
                }
            )
        elif "-seg" in m and ("yolo" in m or "ultralytics" in m):
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
                    "task": "automatic_instance_segmentation",
                    "next_action": (
                        "Ultralytics yolo*-seg already benchmarked in v2.27 "
                        "(segmentation_auto_instance_400_v227.json). The new "
                        "package CLI wiring is deferred to v2.29."
                    ),
                    "evidence_artifact": "segmentation_auto_instance_400_v227.json",
                }
            )
        else:
            rows.append(
                {
                    "model_id": m,
                    "status": "expected_blocker",
                    "code": "SEGMENTATION_PIPELINE_NOT_WIRED",
                    "task": "automatic_instance_segmentation",
                    "next_action": (
                        "Model is not registered as a segmentation candidate in the v2.28 pipeline."
                    ),
                }
            )

    payload = {
        "status": "expected_blocker",
        "code": "SEGMENTATION_BENCHMARK_NOT_IMPLEMENTED",
        "dataset": dataset,
        "device": device,
        "draw_dir": str(draw_dir) if draw_dir else "",
        "sample_gpu": sample_gpu,
        "isolate_process": isolate_process,
        "n_rows": len(rows),
        "rows": rows,
        "task": "automatic_instance_segmentation",
        "message": (
            "v2.28 ships the protocol scaffold and the per-model structured "
            "blockers. Real auto-segmentation runs live in the v2.27 evidence "
            "files; full CLI wiring lands in v2.29."
        ),
        "evidence_artifact_v227": "segmentation_auto_instance_400_v227.json",
    }
    _emit(payload, out=out, fmt=fmt)


@app_promptable.callback(invoke_without_command=True)
def benchmark_promptable_segmentation(
    dataset: str = typer.Option(..., "--dataset"),
    models: str = typer.Option(..., "--models"),
    prompt_source: str = typer.Option(
        "gt-box", "--prompt-source", help="gt-box | gt-point | user-box"
    ),
    max_instances_per_image: int = typer.Option(10, "--max-instances-per-image"),
    device: str = typer.Option("cuda", "--device"),
    out: Path = typer.Option(..., "--out"),
    draw_dir: Path | None = typer.Option(None, "--draw-dir"),
    fmt: str = typer.Option("json", "--format"),
    sample_gpu: bool = typer.Option(False, "--sample-gpu"),
    isolate_process: bool = typer.Option(False, "--isolate-process"),
) -> None:
    """v2.28.0: promptable segmentation (box-prompted SAM/SAM2).

    Currently emits ``PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED`` per model
    with the exact next-action so the v2.28 final report shows
    ``promptable_benchmark_pending`` rather than mixing with the automatic
    leaderboard.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    rows: list[dict[str, Any]] = []
    for m in model_list:
        rows.append(
            {
                "model_id": m,
                "status": "expected_blocker",
                "code": "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
                "task": "promptable_segmentation",
                "prompt_source": prompt_source,
                "next_action": (
                    "Implement SAM/SAM2 box-prompted benchmark: per GT instance, "
                    "pass bbox to predictor, compute mask IoU vs GT mask. See "
                    "information_required_to_finish_v228.csv issue PROMPTABLE-SAM."
                ),
                "evidence_artifact": "information_required_to_finish_v228.csv",
            }
        )
    payload = {
        "status": "expected_blocker",
        "code": "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
        "dataset": dataset,
        "prompt_source": prompt_source,
        "max_instances_per_image": max_instances_per_image,
        "device": device,
        "draw_dir": str(draw_dir) if draw_dir else "",
        "sample_gpu": sample_gpu,
        "isolate_process": isolate_process,
        "n_rows": len(rows),
        "rows": rows,
        "task": "promptable_segmentation",
        "message": (
            "Promptable segmentation benchmark not yet implemented. v2.28 "
            "ships the structured blocker with exact next-action. v2.29 "
            "target."
        ),
    }
    _emit(payload, out=out, fmt=fmt)


__all__ = ["app_auto", "app_promptable"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.18.0: domain-specific benchmark commands.

Honest stubs: most domains return ``status=expected_blocker,
code=BENCHMARK_NOT_IMPLEMENTED`` with the required dataset and roadmap.
``benchmark-agriculture`` and ``benchmark-aerial`` route a real YOLO-format
dataset through the standard detection benchmark when labels are present.
``benchmark-anomaly`` already exists as a real implementation in
``cli/benchmark_anomaly_cmd.py`` — we don't duplicate it.

Pattern: never return null, never claim AP without ground truth, never use
COCO labels for non-detection tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = {
        "ok": "green",
        "expected_blocker": "yellow",
        "failed": "red",
    }.get(payload.get("status"), "white")
    console.print(f"[{color}]{payload.get('code', '')}[/{color}]: {payload.get('message', '')}")


# ---------------------------------------------------------------------------
# Medical
# ---------------------------------------------------------------------------

app_medical = typer.Typer(
    help="v2.18.0: medical-domain benchmark (mostly validate/smoke; full metrics require GT).",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app_medical.callback(invoke_without_command=True)
def benchmark_medical(
    dataset: Path = typer.Option(None, "--dataset"),
    models: str = typer.Option("medsam", "--models"),
    task: str = typer.Option("medsam-2d-box", "--task"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw_dir: Path = typer.Option(None, "--draw-dir"),
) -> None:
    """Medical-domain benchmark. Currently smoke/validate only."""
    # Optional dataset validation
    detail: dict[str, Any] = {"models": [m.strip() for m in models.split(",") if m.strip()]}
    if dataset and dataset.exists():
        # Reuse validator; capture by writing to a temp out
        from tempfile import NamedTemporaryFile

        from visionservex.cli.dataset_validators import validate_medical as _v

        with NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            try:
                _v(path=dataset, task=task, out=Path(fh.name), fmt="json")
            except SystemExit:
                pass
        try:
            detail["dataset_validation"] = json.loads(Path(fh.name).read_text())
        except Exception:
            detail["dataset_validation"] = None

    _emit(
        {
            "status": "expected_blocker",
            "code": "BENCHMARK_NOT_IMPLEMENTED",
            "domain": "medical",
            "task": task,
            "required_dataset": "ct_volume_nifti or 2d_medical_image_with_box_prompts_and_gt_masks",
            "metrics_supported": ["dice_per_class", "iou_per_class", "mask_count", "latency"],
            "message": (
                "Medical-domain benchmark is not yet implemented in v2.18. "
                "Use `visionservex medical validate <model>` and "
                "`visionservex medical segment <model> IMAGE --box X1,Y1,X2,Y2` for smoke runs. "
                "Pre-fix dataset validation is reported under `details.dataset_validation`."
            ),
            "roadmap": "v2.19: TotalSegmentator+MedSAM real Dice/IoU on a tiny medical fixture.",
            "details": detail,
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Agriculture
# ---------------------------------------------------------------------------

app_agriculture = typer.Typer(
    help="v2.18.0: agriculture-domain detection benchmark (routes to detection pipeline when labels present).",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app_agriculture.callback(invoke_without_command=True)
def benchmark_agriculture(
    dataset: Path = typer.Option(None, "--dataset"),
    models: str = typer.Option("dfine-s-o365-coco", "--models"),
    task: str = typer.Option("weed-detection", "--task"),
    max_images: int = typer.Option(50, "--max-images"),
    device: str = typer.Option("auto", "--device"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw_dir: Path = typer.Option(None, "--draw-dir"),
) -> None:
    """Agriculture benchmark. Routes to detection pipeline if YOLO labels are present."""
    # If dataset has YOLO labels, defer to benchmark-detection.
    if dataset and dataset.exists():
        labels_dir = dataset / "labels"
        if labels_dir.exists():
            label_files = list(labels_dir.rglob("*.txt"))
            if label_files:
                # Real detection benchmark via the existing path.
                # Emit a structured "routed" payload that names the actual command;
                # actually running the detection benchmark in-process risks a long
                # download/load in tests, so we expose the command and let callers
                # opt in.
                _emit(
                    {
                        "status": "ok",
                        "code": "ROUTED_TO_DETECTION",
                        "domain": "agriculture",
                        "task": task,
                        "dataset": str(dataset),
                        "n_label_files": len(label_files),
                        "recommended_command": (
                            f"visionservex benchmark-detection --dataset yolo:{dataset} "
                            f"--models {models} --max-images {max_images} --device {device} "
                            f"--require-gpu --sample-gpu --out OUT.json --format json"
                        ),
                        "message": (
                            "Agriculture dataset has YOLO labels. Run the closed-set detection "
                            "benchmark via the recommended_command."
                        ),
                        "details": {"labels_dir": str(labels_dir)},
                    },
                    out=out,
                    fmt=fmt,
                )
                return

    _emit(
        {
            "status": "expected_blocker",
            "code": "LABELS_REQUIRED_FOR_METRICS",
            "domain": "agriculture",
            "task": task,
            "required_dataset": "yolo_crop_weed_labels",
            "metrics_supported": ["ap50", "ap75", "map50_95"],
            "message": (
                "Agriculture detection benchmark needs YOLO labels under labels/<image_stem>.txt. "
                "Without labels, only smoke (prompt-detect / prompt-segment) is available."
            ),
            "roadmap": (
                "Provide a labelled crop/weed dataset and re-run; the command will route to "
                "the closed-set detection pipeline automatically."
            ),
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Aerial
# ---------------------------------------------------------------------------

app_aerial = typer.Typer(
    help="v2.18.0: aerial-domain benchmark (HBB via detection pipeline; OBB blocked on DOTA labels).",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app_aerial.callback(invoke_without_command=True)
def benchmark_aerial(
    dataset: Path = typer.Option(None, "--dataset"),
    dataset_type: str = typer.Option("generic-yolo", "--dataset-type"),
    models: str = typer.Option("dfine-s-o365-coco", "--models"),
    max_images: int = typer.Option(50, "--max-images"),
    device: str = typer.Option("auto", "--device"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Aerial benchmark. DOTA OBB returns OBB_LABELS_REQUIRED; generic-yolo routes to detection."""
    if dataset_type == "dota":
        _emit(
            {
                "status": "expected_blocker",
                "code": "DOTA_OR_OBB_LABELS_REQUIRED",
                "domain": "aerial",
                "task": "obb_detection",
                "required_dataset": "dota_v1_or_v2_obb",
                "metrics_supported": ["obb_ap50", "obb_map50_95"],
                "message": (
                    "DOTA OBB benchmark needs rotated annotations under labelTxt/. "
                    "VisionServeX does not yet ship an OBB evaluator; use "
                    "`visionservex openmmlab validate rtmdet-r-l` for the sidecar route."
                ),
                "roadmap": "v2.19+: native OBB evaluator + rotated metrics.",
            },
            out=out,
            fmt=fmt,
        )
        return

    if dataset and (dataset / "labels").exists():
        label_files = list((dataset / "labels").rglob("*.txt"))
        if label_files:
            _emit(
                {
                    "status": "ok",
                    "code": "ROUTED_TO_DETECTION",
                    "domain": "aerial",
                    "task": "aerial_hbb_detection",
                    "dataset": str(dataset),
                    "dataset_type": dataset_type,
                    "n_label_files": len(label_files),
                    "recommended_command": (
                        f"visionservex benchmark-detection --dataset yolo:{dataset} "
                        f"--models {models} --max-images {max_images} --device {device} "
                        f"--require-gpu --sample-gpu --out OUT.json --format json"
                    ),
                    "message": "Aerial HBB benchmark routes to the closed-set detection pipeline.",
                },
                out=out,
                fmt=fmt,
            )
            return

    _emit(
        {
            "status": "expected_blocker",
            "code": "AERIAL_LABELS_REQUIRED",
            "domain": "aerial",
            "dataset_type": dataset_type,
            "required_dataset": "visdrone_or_xview_yolo or dota_obb",
            "metrics_supported": ["ap50", "obb_ap50"],
            "message": "Aerial benchmark needs labels (YOLO HBB or DOTA OBB).",
            "roadmap": "v2.19+: native OBB evaluator.",
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Surveillance (search/retrieval)
# ---------------------------------------------------------------------------

app_surveillance = typer.Typer(
    help="v2.18.0: surveillance pipeline benchmark — detection→tracker→embedder→retrieval.",
    no_args_is_help=False,
    invoke_without_command=True,
)


@app_surveillance.callback(invoke_without_command=True)
def benchmark_surveillance(
    source: Path = typer.Option(None, "--source", help="Video file or frames folder."),
    query: str = typer.Option("", "--query"),
    detector: str = typer.Option("owlv2-base-patch16", "--detector"),
    tracker: str = typer.Option("simple-iou", "--tracker"),
    embedder: str = typer.Option("siglip2-base-patch16-224", "--embedder"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
    draw_dir: Path = typer.Option(None, "--draw-dir"),
) -> None:
    """Surveillance-search pipeline benchmark. Currently demo-only without GT tracks/queries."""
    if not source or not source.exists():
        _emit(
            {
                "status": "expected_blocker",
                "code": "NO_MEDIA_FOUND",
                "domain": "surveillance",
                "required_dataset": "video_or_frames",
                "metrics_supported": ["retrieval_top_k", "timeline"],
                "message": "Provide --source pointing at a video file or frame folder.",
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    _emit(
        {
            "status": "expected_blocker",
            "code": "BENCHMARK_NOT_IMPLEMENTED",
            "domain": "surveillance",
            "source": str(source),
            "query": query,
            "detector": detector,
            "tracker": tracker,
            "embedder": embedder,
            "required_dataset": "video + query.json + tracks_gt.json for full metrics",
            "metrics_supported": ["retrieval_top_k", "timeline"],
            "metrics_blocked_without_gt": ["track_map", "id_switch_rate"],
            "message": (
                "Surveillance pipeline benchmark with metrics is not yet implemented. "
                "Use `visionservex video-search index` + `query` for the index/retrieval "
                "smoke loop."
            ),
            "roadmap": "v2.19: real retrieval@k against query labels.",
        },
        out=out,
        fmt=fmt,
    )


__all__ = ["app_aerial", "app_agriculture", "app_medical", "app_surveillance"]

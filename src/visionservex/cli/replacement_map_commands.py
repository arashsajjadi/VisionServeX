# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Replacement map: from Ultralytics/YOLO tasks to VisionServeX models.

Honest and task-specific. Does not claim 'better' unless evidence exists.
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Show recommended VisionServeX replacements for Ultralytics/YOLO tasks.")
console = Console()

# ---------------------------------------------------------------------------
# Replacement data
# ---------------------------------------------------------------------------

_REPLACEMENT_MAP: dict[str, dict[str, Any]] = {
    "detect": {
        "yolo_task": "yolo detect (YOLOv8/v9/v11 detection)",
        "description": ("Object detection with axis-aligned bounding boxes on COCO 80 classes."),
        "replacements": [
            {
                "tier": "fastest_demo",
                "models": ["dfine-n", "rfdetr-nano"],
                "category": "demo_fast",
                "why": "Smallest footprint, CPU-capable, instant download. Not for AP claims.",
                "install": "pip install 'visionservex[hf]'  # dfine-n\npip install 'visionservex[rfdetr]'  # rfdetr-nano",
                "command": "visionservex predict dfine-n image.jpg",
                "ap_claim": False,
                "ap_note": "demo_fast tier — do not use to compare AP against YOLO.",
            },
            {
                "tier": "production",
                "models": ["dfine-s-o365-coco", "rfdetr-small"],
                "category": "production_recommended / accuracy_grade",
                "why": "Accuracy-grade, Objects365+COCO pretraining, CPU-capable (dfine-s). Recommended entry points for AP comparison.",
                "install": "pip install 'visionservex[hf]'  # dfine-s\npip install 'visionservex[rfdetr]'  # rfdetr-small",
                "command": "visionservex predict dfine-s-o365-coco image.jpg\nvisionservex predict rfdetr-small image.jpg --device cuda",
                "ap_claim": False,
                "ap_note": "Upstream claims: D-FINE-S ~48.5 mAP, RF-DETR-small ~48.5 mAP. Run benchmark-competitiveness with ground truth to verify.",
            },
            {
                "tier": "accuracy",
                "models": [
                    "dfine-m-o365-coco",
                    "dfine-l-o365-coco",
                    "dfine-x-o365-coco",
                    "rfdetr-medium",
                    "rfdetr-large",
                ],
                "category": "accuracy_grade",
                "why": "Highest available AP from permissive-license backends. Require GPU.",
                "install": "pip install 'visionservex[hf]'  # dfine-*\npip install 'visionservex[rfdetr]'  # rfdetr-*",
                "command": "visionservex predict dfine-m-o365-coco image.jpg --device cuda",
                "ap_claim": False,
                "ap_note": "Upstream D-FINE-M ~52.3 mAP, D-FINE-L ~54.0, D-FINE-X ~55.8. Use benchmark-competitiveness --dataset to confirm on your data.",
            },
        ],
        "honest_caveats": [
            "Do not use dfine-n or rfdetr-nano as the comparison model for YOLO accuracy.",
            "COCO-only D-FINE variants (dfine-s-coco) point to HF repos that may not exist. Use o365 variants.",
            "DEIM, DEIMv2, RT-DETRv4 are registered but not yet runnable in this build.",
            "Run 'visionservex benchmark benchmark-competitiveness --dataset <path>' with annotated images to get AP50/mAP.",
        ],
    },
    "segment": {
        "yolo_task": "yolo segment (YOLOv8/v9/v11 instance segmentation)",
        "description": "Instance segmentation: bounding boxes + binary masks per object.",
        "replacements": [
            {
                "tier": "fastest_demo",
                "models": ["rfdetr-seg-nano"],
                "category": "demo_fast",
                "why": "Smallest segmentation footprint, CPU-capable.",
                "install": "pip install 'visionservex[rfdetr]'",
                "command": "visionservex predict rfdetr-seg-nano image.jpg",
                "ap_claim": False,
                "ap_note": "demo_fast — not for mask AP comparison.",
            },
            {
                "tier": "production",
                "models": ["rfdetr-seg-small", "rfdetr-seg-medium"],
                "category": "production_recommended / accuracy_grade",
                "why": "RF-DETR-Seg-small is the accuracy entry point. Medium requires GPU.",
                "install": "pip install 'visionservex[rfdetr]'",
                "command": "visionservex predict rfdetr-seg-small image.jpg",
                "ap_claim": False,
                "ap_note": "Mask AP requires COCO segmentation annotations. Run visionservex benchmark-segmentation (roadmap).",
            },
            {
                "tier": "prompt_based",
                "models": ["grounded-sam", "grounded-sam2"],
                "category": "production_recommended",
                "why": "Text-prompted segmentation without class-ID constraints.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict grounded-sam2 image.jpg --prompt 'car, person'",
                "ap_claim": False,
                "ap_note": "Not suitable for closed-set detection/segmentation AP comparison.",
            },
            {
                "tier": "expert",
                "models": ["co-dino-inst-vit-l-coco", "maskdino-r50-coco"],
                "category": "expert_sidecar / experimental_sota",
                "why": "High-accuracy but require OpenMMLab/detectron2. Not pip-installable cleanly.",
                "install": "See docs/openmmlab_expert_models.md",
                "command": "N/A — expert install required",
                "ap_claim": False,
                "ap_note": "Not wired in this build. Implementation is stub.",
            },
        ],
        "honest_caveats": [
            "RF-DETR-Seg-large/xlarge/2xlarge are unavailable (HF checkpoints not published).",
            "SAM/SAM2 are not instance segmentation in the YOLO-seg sense — prompts required.",
            "Mask AP not yet implemented in VisionServeX benchmark. Roadmap for v1.4.",
        ],
    },
    "pose": {
        "yolo_task": "yolo pose (YOLOv8-pose, 17-keypoint COCO)",
        "description": "Human pose estimation: 17 keypoints (COCO format).",
        "replacements": [
            {
                "tier": "expert",
                "models": ["rtmpose-s", "rtmpose-m", "rtmpose-l"],
                "category": "expert_sidecar",
                "why": "RTMPose is the strongest permissive-license pose estimator. Requires MMPose.",
                "install": "pip install openmim && mim install mmengine mmcv mmpose",
                "command": "visionservex openmmlab smoke-test rtmpose-s",
                "ap_claim": False,
                "ap_note": "RTMPose-S COCO AP ~65.2 (OKS). Requires OpenMMLab. Not wired natively.",
            },
        ],
        "honest_caveats": [
            "No pose estimator is wired natively in VisionServeX yet.",
            "RTMPose is registered as expert_sidecar; install MMPose separately.",
            "OKS AP is not the same metric as detection mAP — do not mix.",
            "YOLO-pose may win on pure speed on supported hardware.",
        ],
    },
    "obb": {
        "yolo_task": "yolo obb (YOLOv8-OBB, oriented bounding boxes)",
        "description": "Oriented (rotated) bounding box detection.",
        "replacements": [
            {
                "tier": "expert",
                "models": ["rtmdet-r-s", "rtmdet-r-m", "rtmdet-r-l", "rtmdet-r2-s"],
                "category": "expert_sidecar",
                "why": "RTMDet-R/R2 supports oriented detection via MMRotate.",
                "install": "pip install openmim && mim install mmengine mmcv mmrotate",
                "command": "visionservex openmmlab smoke-test rtmdet-r-s",
                "ap_claim": False,
                "ap_note": "Rotated IoU AP is different from standard mAP. No verified VisionServeX OBB benchmark.",
            },
        ],
        "honest_caveats": [
            "No OBB model is wired natively in VisionServeX.",
            "RTMDet-R/R2 are expert_sidecar — requires MMRotate.",
            "VisionServeX does NOT claim a verified winner over YOLO-OBB.",
            "Rotated IoU AP requires dedicated evaluation code (roadmap).",
        ],
    },
    "classify": {
        "yolo_task": "yolo classify (YOLOv8-cls, ImageNet classification)",
        "description": "Image classification on ImageNet-1k (1000 classes).",
        "replacements": [
            {
                "tier": "production",
                "models": ["swinv2-tiny", "swinv2-small"],
                "category": "production_recommended",
                "why": "SwinV2-Tiny/Small: well-tested, CPU-capable, MIT license.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict swinv2-tiny image.jpg",
                "ap_claim": False,
                "ap_note": "SwinV2-Tiny ImageNet top-1 ~81.8%. Run top-k accuracy eval on your dataset.",
            },
            {
                "tier": "accuracy",
                "models": ["swinv2-base", "swinv2-large"],
                "category": "accuracy_grade",
                "why": "SwinV2-Base: ~86.2% top-1 (ImageNet-22K pretrained). Best classification AP from standard install.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict swinv2-base image.jpg",
                "ap_claim": False,
                "ap_note": "Classification top-1 benchmark planned for v1.4.",
            },
            {
                "tier": "expert",
                "models": ["internimage-t", "internimage-b", "internimage-l"],
                "category": "expert_sidecar",
                "why": "InternImage achieves SOTA classification but requires DCNv3 custom ops.",
                "install": "See docs/openmmlab_expert_models.md",
                "command": "N/A — expert install required",
                "ap_claim": False,
                "ap_note": "InternImage-L ImageNet top-1 ~87.9%. Not pip-installable cleanly.",
            },
        ],
        "honest_caveats": [
            "SwinV2 uses ImageNet-1k classes; fine-tune for custom datasets.",
            "Classification top-1/top-5 benchmark not yet implemented in VisionServeX.",
            "InternImage requires DCNv3 custom CUDA ops — not beginner-friendly.",
        ],
    },
    "open-vocab": {
        "yolo_task": "YOLO-World / YOLOE (open-vocabulary detection)",
        "description": "Text-prompted zero-shot object detection.",
        "replacements": [
            {
                "tier": "fastest_demo",
                "models": ["grounding-dino-tiny"],
                "category": "demo_fast",
                "why": "CPU-capable, easy install, good demo quality.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict grounding-dino-tiny image.jpg --prompt 'cat,dog'",
                "ap_claim": False,
                "ap_note": "GD-Tiny zero-shot COCO AP ~48.4. demo_fast — use swin-b for accuracy.",
            },
            {
                "tier": "accuracy",
                "models": ["grounding-dino-swin-b"],
                "category": "accuracy_grade",
                "why": "Stronger Swin-B backbone. Zero-shot COCO AP ~56.7.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict grounding-dino-swin-b image.jpg --prompt 'cat'",
                "ap_claim": False,
                "ap_note": "Zero-shot AP ~56.7 per upstream. Requires GPU.",
            },
            {
                "tier": "api_gated",
                "models": ["grounding-dino-1.5", "grounding-dino-1.6"],
                "category": "external_api",
                "why": "Strongest Grounding DINO models but API-gated by IDEA-Research.",
                "install": "Requires API token from IDEA-Research",
                "command": "N/A — upstream API token required",
                "ap_claim": False,
                "ap_note": "Closed API. Not self-hostable without upstream agreement.",
            },
            {
                "tier": "prompt_segment",
                "models": ["grounded-sam2"],
                "category": "production_recommended",
                "why": "Text → detect → segment pipeline. Not a detection AP model.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict grounded-sam2 image.jpg --prompt 'person'",
                "ap_claim": False,
                "ap_note": "Not suitable for detection AP comparison — different output type.",
            },
        ],
        "honest_caveats": [
            "Open-vocab AP uses LVIS/COCO zero-shot protocols — different from closed-set mAP.",
            "Do not compare Grounding DINO zero-shot AP directly with YOLO closed-set mAP.",
            "Grounding DINO 1.5/1.6 are API-gated and not self-hostable.",
            "SEEM is registered as expert_sidecar; complex install required.",
        ],
    },
    "sam": {
        "yolo_task": "Ultralytics SAM / FastSAM wrappers",
        "description": "Foundation segmentation with point/box prompts.",
        "replacements": [
            {
                "tier": "production",
                "models": ["sam-vit-base", "sam-vit-large"],
                "category": "production_recommended / accuracy_grade",
                "why": "Official SAM v1 from Meta, no wrapper overhead. Same weights as Ultralytics SAM.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict sam-vit-base image.jpg --point 320,240",
                "ap_claim": False,
                "ap_note": "SAM uses SA-1B metrics. Not comparable to COCO detection mAP.",
            },
            {
                "tier": "newer",
                "models": [
                    "sam2-hiera-tiny",
                    "sam2-hiera-small",
                    "sam2-hiera-base-plus",
                    "sam2-hiera-large",
                ],
                "category": "production_recommended / accuracy_grade",
                "why": "SAM 2 via HF Transformers — no CUDA extensions. Stronger quality than SAM v1.",
                "install": "pip install 'visionservex[hf]'",
                "command": "visionservex predict sam2-hiera-tiny image.jpg",
                "ap_claim": False,
                "ap_note": "SAM2 quality assessed on SA-1B video/image benchmarks.",
            },
        ],
        "honest_caveats": [
            "SAM/SAM2 are not detection models. Do not include in detection mAP tables.",
            "Ultralytics FastSAM uses different architecture — not a 1:1 replacement.",
        ],
    },
}


def _format_map_markdown(task: str | None, from_tool: str | None) -> str:
    items = {}
    if task:
        t = task.lower().replace("-", "_").replace(" ", "_")
        if t in _REPLACEMENT_MAP:
            items[t] = _REPLACEMENT_MAP[t]
        elif t == "open_vocab":
            items["open-vocab"] = _REPLACEMENT_MAP.get("open-vocab", {})
    elif from_tool and from_tool.lower() == "ultralytics":
        items = _REPLACEMENT_MAP
    else:
        items = _REPLACEMENT_MAP

    if not items:
        return f"No replacement map entry for task={task}"

    lines = ["# VisionServeX Replacement Map\n"]
    lines.append(
        "_Honest replacements for Ultralytics/YOLO tasks. No 'better' claims without evidence._\n"
    )

    for key, info in items.items():
        lines += [
            f"## {info.get('yolo_task', key)}",
            "",
            f"_{info.get('description', '')}_",
            "",
            "| Tier | VisionServeX models | Category | Install | AP claim |",
            "|------|---------------------|----------|---------|---------|",
        ]
        for rep in info.get("replacements", []):
            models = ", ".join(f"`{m}`" for m in rep["models"])
            ap = "No — use benchmark-competitiveness" if not rep.get("ap_claim") else "Yes"
            lines.append(
                f"| {rep['tier']} | {models} | {rep['category']} | `{rep['install'][:40]}...` | {ap} |"
            )

        lines += ["", "### Caveats"]
        for c in info.get("honest_caveats", []):
            lines.append(f"- ⚠ {c}")
        lines.append("")

    return "\n".join(lines)


@app.command(
    "map",
    help="Show recommended VisionServeX models for each Ultralytics/YOLO task.",
)
def replacement_map_cmd(
    task: str | None = typer.Option(
        None, "--task", help="Filter to task: detect|segment|pose|obb|classify|open-vocab|sam"
    ),
    from_tool: str | None = typer.Option(
        None, "--from", help="Filter by source tool (e.g. ultralytics)."
    ),
    format_: str = typer.Option("human", "--format", help="Output: human|json|markdown"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show replacement models for Ultralytics/YOLO tasks."""
    if json_:
        format_ = "json"

    # Select entries
    if task:
        t = task.lower().replace("-", "_")
        # Handle aliases
        alias = {"open_vocab": "open-vocab", "seg": "segment", "cls": "classify"}
        t = alias.get(t, t)
        selected = {k: v for k, v in _REPLACEMENT_MAP.items() if k == t}
        if not selected:
            console.print(
                f"[yellow]No replacement map for task={task!r}. "
                f"Available: {', '.join(_REPLACEMENT_MAP)}[/yellow]"
            )
            raise typer.Exit(1)
    else:
        selected = _REPLACEMENT_MAP

    if format_ == "json":
        typer.echo(json.dumps(selected, indent=2))
        return

    if format_ == "markdown":
        typer.echo(_format_map_markdown(task, from_tool))
        return

    # Human-readable
    for key, info in selected.items():
        from rich.panel import Panel

        console.print(
            Panel.fit(
                f"[bold]{info.get('yolo_task', key)}[/bold]\n{info.get('description', '')}",
                border_style="cyan",
            )
        )

        table = Table(show_lines=True)
        for col in ("Tier", "Models", "Category", "Install hint"):
            table.add_column(col)
        for rep in info.get("replacements", []):
            models = "\n".join(rep["models"])
            table.add_row(rep["tier"], models, rep["category"], rep["install"][:50])
        console.print(table)

        for rep in info.get("replacements", []):
            if rep.get("ap_note"):
                console.print(f"  [dim]{rep['tier']}: {rep['ap_note']}[/dim]")

        console.print("\n[bold]Caveats:[/bold]")
        for c in info.get("honest_caveats", []):
            console.print(f"  [yellow]⚠[/yellow] {c}")
        console.print()


__all__ = ["app"]

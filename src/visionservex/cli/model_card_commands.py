# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Model card system — structured, honest per-model documentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Show structured model cards.")
console = Console()

# ---------------------------------------------------------------------------
# Supplementary card data (what the registry doesn't store)
# ---------------------------------------------------------------------------

_CARD_SUPPLEMENTS: dict[str, dict[str, Any]] = {
    # ---- D-FINE ----
    "dfine-n": {
        "recommended_for": ["quick demos", "edge devices", "CPU inference", "first detection test"],
        "not_recommended_for": [
            "accuracy benchmarks",
            "YOLO AP comparison",
            "production detection",
        ],
        "replaces_or_competes_with": ["YOLOv8n (speed only, not accuracy)", "MobileNet-SSD"],
        "input_type": "RGB image (any resolution)",
        "output_type": "DetectionResult (boxes, labels, scores)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "Upstream D-FINE-N reports COCO mAP ~42.8. Do not use this model to claim D-FINE beats YOLO.",
        "visionservex_benchmark_status": "latency tested; AP not validated",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-n image.jpg",
        "known_limitations": [
            "Not suitable for accuracy benchmarks.",
            "Nano checkpoint may miss small/crowded objects.",
        ],
    },
    "dfine-s": {
        "recommended_for": [
            "accuracy-grade CPU detection",
            "competitive COCO mAP",
            "Colab GPU baseline",
        ],
        "not_recommended_for": ["real-time edge inference"],
        "replaces_or_competes_with": ["YOLOv8s (similar scale)", "RT-DETR-R18"],
        "input_type": "RGB image (any resolution)",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "D-FINE-S (Obj365+COCO) upstream reports COCO mAP ~48.5. This uses the ustc-community HF checkpoint.",
        "visionservex_benchmark_status": "wired; AP not yet independently verified in this build",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-s image.jpg",
        "known_limitations": [
            "ustc-community checkpoint — not the official Peterande/D-FINE .pth. AP may differ slightly.",
        ],
    },
    "dfine-s-o365-coco": {
        "recommended_for": [
            "accuracy-grade detection benchmarks",
            "YOLO comparison baseline",
            "CPU-capable AP evaluation",
        ],
        "not_recommended_for": ["real-time inference", "resource-constrained edge"],
        "replaces_or_competes_with": ["YOLOv8s", "RT-DETR-R18", "YOLOv9s"],
        "input_type": "RGB image (any resolution)",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "D-FINE-S (Obj365+COCO) upstream claims COCO mAP ~48.5. Objects365 pretraining improves recall on rare classes.",
        "visionservex_benchmark_status": "wired; use benchmark-competitiveness --dataset <path> to validate AP",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-s-o365-coco image.jpg",
        "known_limitations": [
            "AP claims from upstream; run benchmark-competitiveness with ground truth to confirm.",
            "ustc-community HF checkpoint, not official GitHub release.",
        ],
    },
    "dfine-m-o365-coco": {
        "recommended_for": ["high-accuracy GPU detection", "competitive AP benchmark"],
        "not_recommended_for": ["CPU inference (slow)", "laptops without GPU"],
        "replaces_or_competes_with": ["YOLOv8m", "RT-DETR-R50"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "D-FINE-M (Obj365+COCO) upstream claims COCO mAP ~52.3.",
        "visionservex_benchmark_status": "wired; AP not independently verified",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-m-o365-coco image.jpg --device cuda",
        "known_limitations": ["Requires GPU. 2 GB+ VRAM minimum."],
    },
    "dfine-l-o365-coco": {
        "recommended_for": ["research-grade AP benchmarks", "maximum D-FINE accuracy on GPU"],
        "not_recommended_for": ["CPU inference", "Colab T4 (borderline VRAM)"],
        "replaces_or_competes_with": ["YOLOv8l", "RT-DETR-R101"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": False,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "D-FINE-L (Obj365+COCO) upstream claims COCO mAP ~54.0.",
        "visionservex_benchmark_status": "wired; requires 4+ GB VRAM",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-l-o365-coco image.jpg --device cuda",
        "known_limitations": ["Requires 4+ GB VRAM. T4 borderline."],
    },
    "dfine-x-o365-coco": {
        "recommended_for": ["maximum accuracy detection research"],
        "not_recommended_for": ["production deployment", "resource-limited environments"],
        "replaces_or_competes_with": ["YOLOv8x", "YOLOv9e"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": False,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "D-FINE-X (Obj365+COCO) upstream claims COCO mAP ~55.8.",
        "visionservex_benchmark_status": "wired; requires 8+ GB VRAM",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict dfine-x-o365-coco image.jpg --device cuda",
        "known_limitations": ["Requires 8-12 GB VRAM."],
    },
    # ---- RF-DETR ----
    "rfdetr-nano": {
        "recommended_for": ["quick laptop demos", "CPU detection", "first detection benchmark"],
        "not_recommended_for": ["accuracy comparison with YOLO", "production"],
        "replaces_or_competes_with": ["YOLOv8n (speed benchmark only)"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR Nano is a demo/edge variant. Do not use to claim RF-DETR beats YOLO.",
        "visionservex_benchmark_status": "wired; use rfdetr-small for AP comparison",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-nano image.jpg",
        "known_limitations": ["Demo model. Use rfdetr-small for accuracy benchmarks."],
    },
    "rfdetr-small": {
        "recommended_for": ["accuracy-grade detection", "GPU-accelerated benchmark", "Colab GPU"],
        "not_recommended_for": ["CPU-only environments (slow)"],
        "replaces_or_competes_with": ["YOLOv8s", "D-FINE-S"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR-small is the recommended AP entry point from Roboflow.",
        "visionservex_benchmark_status": "wired; use benchmark-competitiveness with ground truth",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-small image.jpg",
        "known_limitations": ["AP claims require ground-truth evaluation."],
    },
    "rfdetr-medium": {
        "recommended_for": ["high-accuracy GPU detection", "competitive mAP benchmark"],
        "not_recommended_for": ["CPU inference"],
        "replaces_or_competes_with": ["YOLOv8m", "D-FINE-M"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": True,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR-medium accuracy data available at roboflow/rf-detr.",
        "visionservex_benchmark_status": "wired; AP not independently verified",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-medium image.jpg --device cuda",
        "known_limitations": ["3+ GB VRAM recommended."],
    },
    "rfdetr-large": {
        "recommended_for": ["maximum RF-DETR accuracy", "research benchmarks"],
        "not_recommended_for": ["CPU inference", "Colab T4 (borderline)"],
        "replaces_or_competes_with": ["YOLOv8l", "D-FINE-L"],
        "input_type": "RGB image",
        "output_type": "DetectionResult",
        "colab_suitable": False,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR-large is the highest-AP non-Plus variant with Apache-2.0.",
        "visionservex_benchmark_status": "wired; requires 6+ GB VRAM",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-large image.jpg --device cuda",
        "known_limitations": ["6+ GB VRAM required."],
    },
    # ---- Segmentation ----
    "rfdetr-seg-nano": {
        "recommended_for": ["segmentation demo", "CPU-capable masks"],
        "not_recommended_for": ["accuracy benchmarks; use rfdetr-seg-small"],
        "replaces_or_competes_with": ["YOLOv8n-seg (demo only)"],
        "input_type": "RGB image",
        "output_type": "SegmentationResult (masks, boxes, labels, scores)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "Nano variant; not suitable for mask AP comparison.",
        "visionservex_benchmark_status": "wired; demo_fast only",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-seg-nano image.jpg",
        "known_limitations": ["Demo model. Use rfdetr-seg-small for segmentation AP."],
    },
    "rfdetr-seg-small": {
        "recommended_for": ["instance segmentation", "Colab GPU masks"],
        "not_recommended_for": [],
        "replaces_or_competes_with": ["YOLOv8s-seg"],
        "input_type": "RGB image",
        "output_type": "SegmentationResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR-Seg-small is the recommended segmentation entry point.",
        "visionservex_benchmark_status": "wired",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-seg-small image.jpg",
        "known_limitations": ["Mask AP requires ground-truth COCO segmentation evaluation."],
    },
    "rfdetr-seg-medium": {
        "recommended_for": ["high-accuracy instance segmentation"],
        "not_recommended_for": ["CPU inference"],
        "replaces_or_competes_with": ["YOLOv8m-seg"],
        "input_type": "RGB image",
        "output_type": "SegmentationResult",
        "colab_suitable": True,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "RF-DETR-Seg-medium — accuracy data from Roboflow.",
        "visionservex_benchmark_status": "wired; 4+ GB VRAM",
        "install_command": "pip install 'visionservex[rfdetr]'",
        "quick_command": "visionservex predict rfdetr-seg-medium image.jpg --device cuda",
        "known_limitations": ["4+ GB VRAM."],
    },
    # ---- SAM / SAM2 ----
    "sam-vit-base": {
        "recommended_for": ["interactive segmentation", "point/box prompts", "CPU masks"],
        "not_recommended_for": ["closed-set detection AP", "comparing with YOLO detection"],
        "replaces_or_competes_with": ["Ultralytics SAM wrapper (same weights)"],
        "input_type": "RGB image + optional points/boxes",
        "output_type": "FoundationSegmentResult (masks)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "SAM v1 ViT-Base from Meta. COCO mask AP not comparable to closed-set detection AP.",
        "visionservex_benchmark_status": "wired; not included in detection AP tables",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict sam-vit-base image.jpg --point 320,240",
        "known_limitations": [
            "Not a detection model. Do not mix with detection AP.",
            "Point/box prompts required for object-level masks.",
        ],
    },
    "sam2-hiera-tiny": {
        "recommended_for": ["prompt-based segmentation", "fine-grained masks", "Colab demos"],
        "not_recommended_for": ["detection AP benchmarks"],
        "replaces_or_competes_with": ["Ultralytics SAM2 wrapper"],
        "input_type": "RGB image + optional points/boxes",
        "output_type": "FoundationSegmentResult (masks)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "SAM 2 Hiera-Tiny from Meta via HF. No CUDA extensions required.",
        "visionservex_benchmark_status": "wired; demo_fast",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict sam2-hiera-tiny image.jpg",
        "known_limitations": ["SAM 2 metrics use SA-1B, not COCO. Do not mix with detection mAP."],
    },
    # ---- Grounding DINO ----
    "grounding-dino-tiny": {
        "recommended_for": ["open-vocabulary demos", "text-prompted detection"],
        "not_recommended_for": ["closed-set AP comparison", "large-scale open-vocab benchmarks"],
        "replaces_or_competes_with": ["YOLO-World (demo mode)", "YOLOE"],
        "input_type": "RGB image + text prompt",
        "output_type": "DetectionResult (text-conditioned boxes)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "Grounding DINO Tiny from IDEA-Research. COCO zero-shot AP ~48.4.",
        "visionservex_benchmark_status": "wired; demo_fast",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": 'visionservex predict grounding-dino-tiny image.jpg --prompt "cat,dog,car"',
        "known_limitations": ["Zero-shot AP; prompt quality affects results."],
    },
    "grounding-dino-swin-b": {
        "recommended_for": [
            "accuracy-grade open-vocabulary detection",
            "COCO zero-shot evaluation",
        ],
        "not_recommended_for": ["resource-constrained environments"],
        "replaces_or_competes_with": ["YOLO-World", "YOLOE-L"],
        "input_type": "RGB image + text prompt",
        "output_type": "DetectionResult",
        "colab_suitable": False,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "Grounding DINO Base (Swin-B). COCO zero-shot AP ~56.7.",
        "visionservex_benchmark_status": "wired; accuracy_grade",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": 'visionservex predict grounding-dino-swin-b image.jpg --prompt "cat,dog"',
        "known_limitations": ["8 GB VRAM minimum. Prompt format affects AP."],
    },
    # ---- Grounded SAM / SAM2 ----
    "grounded-sam": {
        "recommended_for": ["text-prompted segmentation", "find-and-segment"],
        "not_recommended_for": ["detection AP benchmarks"],
        "replaces_or_competes_with": ["Grounded-SAM (original implementation)"],
        "input_type": "RGB image + text prompt",
        "output_type": "GroundedSegmentResult (boxes + masks)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "Composition: Grounding DINO Tiny + SAM v1. Not a single-checkpoint model.",
        "visionservex_benchmark_status": "wired; beta",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": 'visionservex predict grounded-sam image.jpg --prompt "car"',
        "known_limitations": ["Composed pipeline; errors in detector cascade to segmenter."],
    },
    "grounded-sam2": {
        "recommended_for": ["text-prompted instance masks with SAM2 quality"],
        "not_recommended_for": ["detection AP benchmarks"],
        "replaces_or_competes_with": ["Grounded-SAM-2 (IDEA-Research)"],
        "input_type": "RGB image + text prompt",
        "output_type": "GroundedSegmentResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "Grounding DINO Tiny + SAM 2 Tiny via HF. No CUDA extensions.",
        "visionservex_benchmark_status": "wired; beta",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": 'visionservex predict grounded-sam2 image.jpg --prompt "person"',
        "known_limitations": ["Composed pipeline."],
    },
    # ---- Classification ----
    "swinv2-tiny": {
        "recommended_for": ["ImageNet-1k classification", "CPU classification demo"],
        "not_recommended_for": ["detection AP", "open-vocabulary"],
        "replaces_or_competes_with": ["YOLOv8-cls (tiny variant)"],
        "input_type": "RGB image",
        "output_type": "ClassificationResult (top-k labels, scores)",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "SwinV2-Tiny ImageNet-1k top-1 ~81.8%.",
        "visionservex_benchmark_status": "wired; production_recommended",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict swinv2-tiny image.jpg",
        "known_limitations": ["ImageNet-1k classes only (1000 categories)."],
    },
    "swinv2-base": {
        "recommended_for": ["accuracy-grade classification", "GPU classification benchmark"],
        "not_recommended_for": ["edge inference"],
        "replaces_or_competes_with": ["YOLOv8-cls (medium)", "EfficientNet-B4"],
        "input_type": "RGB image",
        "output_type": "ClassificationResult",
        "colab_suitable": True,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "SwinV2-Base ImageNet-1k top-1 ~86.2% (22k pretrained).",
        "visionservex_benchmark_status": "wired; accuracy_grade",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict swinv2-base image.jpg",
        "known_limitations": ["1.5 GB VRAM minimum."],
    },
    # ---- OneFormer ----
    "oneformer-swin-large": {
        "recommended_for": ["panoptic/semantic/instance segmentation", "multi-task segmentation"],
        "not_recommended_for": ["detection AP comparison with YOLO"],
        "replaces_or_competes_with": ["Mask2Former", "Panoptic-FPN"],
        "input_type": "RGB image",
        "output_type": "SegmentationResult (semantic/panoptic/instance via --task flag)",
        "colab_suitable": False,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "OneFormer Swin-L COCO panoptic PQ ~57.9. Semantic mIoU on ADE20K ~57.0.",
        "visionservex_benchmark_status": "wired; accuracy_grade; do not compare PQ with detection mAP",
        "install_command": "pip install 'visionservex[hf]'",
        "quick_command": "visionservex predict oneformer-swin-large image.jpg --task panoptic",
        "known_limitations": [
            "6 GB VRAM minimum.",
            "Panoptic/semantic/instance AP not directly comparable to closed-set detection mAP.",
        ],
    },
    # ---- Expert sidecars ----
    "rtmpose-s": {
        "recommended_for": ["2D pose estimation", "17-keypoint COCO pose"],
        "not_recommended_for": ["detection", "classification", "segmentation"],
        "replaces_or_competes_with": ["YOLO-Pose (similar scale)"],
        "input_type": "RGB image",
        "output_type": "PoseResult (keypoints, scores)",
        "colab_suitable": False,
        "cpu_suitable": True,
        "cuda_suitable": True,
        "official_benchmark_note": "RTMPose-S COCO AP ~65.2. Requires MMPose.",
        "visionservex_benchmark_status": "expert_sidecar; real inference requires OpenMMLab install",
        "install_command": "pip install openmim && mim install mmengine mmcv mmpose",
        "quick_command": "visionservex openmmlab smoke-test rtmpose-s",
        "known_limitations": [
            "Requires OpenMMLab toolchain.",
            "No verified AP from VisionServeX native path yet.",
        ],
    },
    "internimage-t": {
        "recommended_for": ["expert-level classification research"],
        "not_recommended_for": ["production use", "beginners"],
        "replaces_or_competes_with": ["SwinV2-T (simpler install)"],
        "input_type": "RGB image",
        "output_type": "ClassificationResult",
        "colab_suitable": False,
        "cpu_suitable": False,
        "cuda_suitable": True,
        "official_benchmark_note": "InternImage-T ImageNet-1k top-1 ~83.5%.",
        "visionservex_benchmark_status": "expert_sidecar; requires DCNv3 custom CUDA ops",
        "install_command": "see docs/openmmlab_expert_models.md",
        "quick_command": "N/A — expert install required",
        "known_limitations": [
            "Requires custom DCNv3 CUDA ops; may not build on all platforms.",
            "Not pip-installable cleanly.",
        ],
    },
}


def _get_card(model_id: str) -> dict[str, Any]:
    """Build a model card combining registry data + supplementary info."""
    from visionservex.registry import RegistryError, default_registry

    try:
        entry = default_registry().get(model_id)
    except RegistryError:
        return {"error": f"model '{model_id}' not found in registry"}

    supp = _CARD_SUPPLEMENTS.get(model_id, {})

    return {
        "model_id": model_id,
        "display_name": entry.display_name,
        "family": entry.family,
        "task": entry.task,
        "strength_category": entry.model_category,
        "implementation_status": entry.implementation_status,
        "project_status": entry.status,
        "license": entry.license,
        "license_uncertain": entry.license_uncertain or False,
        "upstream_url": entry.upstream_url,
        "install_extra": entry.install_extra,
        "install_command": supp.get(
            "install_command",
            f"pip install 'visionservex[{entry.install_extra}]'"
            if entry.install_extra
            else "pip install visionservex",
        ),
        "quick_command": supp.get("quick_command", f"visionservex predict {model_id} image.jpg"),
        "input_type": supp.get("input_type", "RGB image"),
        "output_type": supp.get("output_type", "task-dependent result"),
        "recommended_for": supp.get("recommended_for", []),
        "not_recommended_for": supp.get("not_recommended_for", []),
        "replaces_or_competes_with": supp.get("replaces_or_competes_with", []),
        "expected_hardware": {
            "cpu_suitable": supp.get("cpu_suitable", "cpu" in entry.supported_devices),
            "cuda_suitable": supp.get("cuda_suitable", "cuda" in entry.supported_devices),
            "colab_suitable": supp.get("colab_suitable", False),
            "minimum_vram_gb": entry.minimum_vram_gb,
            "recommended_vram_gb": entry.recommended_vram_gb,
            "minimum_ram_gb": entry.minimum_ram_gb,
        },
        "auto_download": entry.auto_download,
        "known_limitations": supp.get(
            "known_limitations", entry.notes.split(". ") if entry.notes else []
        ),
        "official_benchmark_note": supp.get(
            "official_benchmark_note", "No official benchmark note available for this model."
        ),
        "visionservex_benchmark_status": supp.get("visionservex_benchmark_status", "not validated"),
        "unavailable_reason": entry.unavailable_reason,
        "docs_link": f"https://github.com/arashsajjadi/VisionServeX/blob/main/docs/model_zoo.md#{model_id}",
    }


def _card_to_markdown(card: dict[str, Any]) -> str:
    if "error" in card:
        return f"# Error\n\n{card['error']}"

    lines = [
        f"# Model Card: {card['display_name']}",
        f"**ID:** `{card['model_id']}`  ",
        f"**Family:** {card['family']}  ",
        f"**Task:** {card['task']}  ",
        f"**Strength category:** `{card['strength_category']}`  ",
        f"**Implementation:** {card['implementation_status']} / {card['project_status']}  ",
        f"**License:** {card['license']}{'  ⚠ uncertain' if card['license_uncertain'] else ''}  ",
        "",
        "## Install",
        "```bash",
        f"{card['install_command']}",
        "```",
        "",
        "## Quick start",
        "```bash",
        f"{card['quick_command']}",
        "```",
        "",
        "## Input / Output",
        f"- Input: {card['input_type']}",
        f"- Output: {card['output_type']}",
        "",
        "## Recommended for",
    ]
    for r in card["recommended_for"]:
        lines.append(f"- {r}")
    lines += ["", "## Not recommended for"]
    for r in card["not_recommended_for"]:
        lines.append(f"- {r}")

    lines += ["", "## Competes with / replaces"]
    for r in card["replaces_or_competes_with"]:
        lines.append(f"- {r}")

    hw = card["expected_hardware"]
    lines += [
        "",
        "## Hardware requirements",
        f"- CPU suitable: {'yes' if hw['cpu_suitable'] else 'no'}",
        f"- CUDA suitable: {'yes' if hw['cuda_suitable'] else 'no'}",
        f"- Colab suitable: {'yes' if hw['colab_suitable'] else 'no'}",
        f"- Minimum VRAM: {hw['minimum_vram_gb'] or 'N/A'} GB",
        f"- Recommended VRAM: {hw['recommended_vram_gb'] or 'N/A'} GB",
        "",
        "## Official benchmark",
        f"> {card['official_benchmark_note']}",
        "",
        "## VisionServeX benchmark status",
        f"> {card['visionservex_benchmark_status']}",
    ]

    if card["known_limitations"]:
        lines += ["", "## Known limitations"]
        for lim in card["known_limitations"]:
            lines.append(f"- {lim}")

    if card["unavailable_reason"]:
        lines += ["", "## Unavailable reason", f"> {card['unavailable_reason']}"]

    lines += ["", f"[Upstream]({card['upstream_url']}) | [Model zoo]({card['docs_link']})"]
    return "\n".join(lines)


@app.command("show", help="Show a structured model card for a model.")
def show_card(
    model_id: str,
    format_: str = typer.Option("human", "--format", help="Output format: human|json|markdown"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    if json_:
        format_ = "json"

    card = _get_card(model_id)

    if format_ == "json":
        typer.echo(json.dumps(card, indent=2, default=str))
        return

    if format_ == "markdown":
        typer.echo(_card_to_markdown(card))
        return

    # Human-readable
    if "error" in card:
        console.print(f"[red]error:[/red] {card['error']}")
        raise typer.Exit(1)

    from rich.panel import Panel

    cat_color = {
        "accuracy_grade": "green",
        "production_recommended": "cyan",
        "demo_fast": "yellow",
        "experimental_sota": "magenta",
        "expert_sidecar": "grey50",
        "external_api": "grey50",
        "unavailable_with_reason": "red",
    }.get(card["strength_category"], "white")

    console.print(
        Panel.fit(
            f"[bold]{card['display_name']}[/bold]\n"
            f"ID: {card['model_id']} | Task: {card['task']} | "
            f"Category: [{cat_color}]{card['strength_category']}[/{cat_color}]",
            border_style="cyan",
        )
    )

    if card["recommended_for"]:
        console.print("\n[bold]Recommended for:[/bold]")
        for r in card["recommended_for"]:
            console.print(f"  [green]✓[/green] {r}")

    if card["not_recommended_for"]:
        console.print("\n[bold]Not recommended for:[/bold]")
        for r in card["not_recommended_for"]:
            console.print(f"  [red]✗[/red] {r}")

    hw = card["expected_hardware"]
    console.print(
        f"\n[bold]Hardware:[/bold] CPU={'yes' if hw['cpu_suitable'] else 'no'} | "
        f"CUDA={'yes' if hw['cuda_suitable'] else 'no'} | "
        f"Colab={'yes' if hw['colab_suitable'] else 'no'} | "
        f"VRAM min={hw['minimum_vram_gb'] or 'N/A'} GB"
    )
    console.print(f"[bold]License:[/bold] {card['license']}")
    console.print(f"[bold]Install:[/bold] [cyan]{card['install_command']}[/cyan]")
    console.print(f"[bold]Quick start:[/bold] [cyan]{card['quick_command']}[/cyan]")

    console.print("\n[bold]Official benchmark:[/bold]")
    console.print(f"  {card['official_benchmark_note']}")
    console.print("\n[bold]VisionServeX status:[/bold]")
    console.print(f"  {card['visionservex_benchmark_status']}")

    if card["known_limitations"]:
        console.print("\n[bold]Known limitations:[/bold]")
        for lim in card["known_limitations"]:
            console.print(f"  [yellow]•[/yellow] {lim}")

    if card["unavailable_reason"]:
        console.print(f"\n[red]UNAVAILABLE:[/red] {card['unavailable_reason']}")


@app.command("list", help="List all models with available model cards.")
def list_cards(
    task: str | None = typer.Option(None, "--task"),
    category: str | None = typer.Option(None, "--category"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    from visionservex.registry import default_registry

    entries = default_registry().list(task=task)  # type: ignore[arg-type]
    if category:
        entries = [e for e in entries if e.model_category == category]

    has_card = [e for e in entries if e.id in _CARD_SUPPLEMENTS]
    no_card = [e for e in entries if e.id not in _CARD_SUPPLEMENTS]

    if json_:
        typer.echo(
            json.dumps(
                {
                    "with_full_card": [e.id for e in has_card],
                    "registry_only": [e.id for e in no_card],
                    "total": len(entries),
                },
                indent=2,
            )
        )
        return

    table = Table(title=f"Model cards ({len(has_card)} full, {len(no_card)} registry-only)")
    table.add_column("Model ID")
    table.add_column("Task")
    table.add_column("Category")
    table.add_column("Full card")
    for e in entries:
        cat_color = {
            "accuracy_grade": "green",
            "production_recommended": "cyan",
            "demo_fast": "yellow",
        }.get(e.model_category or "", "white")
        has = "[green]yes[/green]" if e.id in _CARD_SUPPLEMENTS else "[grey50]registry[/grey50]"
        table.add_row(e.id, e.task, f"[{cat_color}]{e.model_category}[/{cat_color}]", has)
    console.print(table)


@app.command("export", help="Export all model cards to a markdown file.")
def export_cards(
    out: Path = typer.Option(Path("docs/generated_model_cards.md"), "--out"),
    task: str | None = typer.Option(None, "--task"),
) -> None:
    from visionservex.registry import default_registry

    entries = default_registry().list(task=task)  # type: ignore[arg-type]
    lines = ["# VisionServeX Model Cards\n", "_Auto-generated. Do not edit manually._\n"]

    for e in [e for e in entries if e.id in _CARD_SUPPLEMENTS]:
        card = _get_card(e.id)
        lines.append(_card_to_markdown(card))
        lines.append("\n---\n")

    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    console.print(
        f"[green]Exported {len([e for e in entries if e.id in _CARD_SUPPLEMENTS])} model cards to {out}[/green]"
    )


__all__ = ["app"]

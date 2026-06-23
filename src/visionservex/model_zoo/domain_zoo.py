# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Domain-specific model zoo: goal-driven recipes per vertical.

Each domain has a curated list of recommended models + an exact pipeline
recipe + install commands + known limitations. Recipes are honest: they
state which steps are wired in core vs which require expert sidecars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainRecipe:
    """Pipeline recipe for a specific goal within a domain."""

    domain: str
    goal: str
    description: str
    pipeline: list[str] = field(default_factory=list)
    recommended_models: list[str] = field(default_factory=list)
    install_commands: list[str] = field(default_factory=list)
    quick_commands: list[str] = field(default_factory=list)
    expected_hardware: str = ""
    runnable_today: bool = False
    limitations: list[str] = field(default_factory=list)
    license_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "goal": self.goal,
            "description": self.description,
            "pipeline": self.pipeline,
            "recommended_models": self.recommended_models,
            "install_commands": self.install_commands,
            "quick_commands": self.quick_commands,
            "expected_hardware": self.expected_hardware,
            "runnable_today": self.runnable_today,
            "limitations": self.limitations,
            "license_notes": self.license_notes,
        }


DOMAIN_ZOO: dict[str, dict[str, DomainRecipe]] = {
    "yolo26-competitors": {
        "accuracy": DomainRecipe(
            domain="yolo26-competitors",
            goal="accuracy",
            description=(
                "Best permissive-license detection models comparable to YOLO26-X / YOLO11-X. "
                "Use process-isolated benchmark to measure AP fairly."
            ),
            pipeline=[
                "1. Pull model weights (one-time)",
                "2. Run inference with VisionModel context manager (VRAM-safe)",
                "3. Evaluate AP50/mAP50:95 on annotated dataset",
            ],
            recommended_models=[
                "dfine-x-o365-coco",
                "dfine-l-o365-coco",
                "rfdetr-large",
            ],
            install_commands=[
                "pip install 'visionservex[hf]'",
                "pip install 'visionservex[rfdetr]'",
            ],
            quick_commands=[
                "visionservex predict dfine-x-o365-coco image.jpg --device cuda",
                "visionservex benchmark benchmark-competitiveness --dataset yolo:/path/to/coco128 "
                "--models ultralytics:yolo11x,dfine-x-o365-coco,rfdetr-large --isolate-process",
            ],
            expected_hardware="6+ GB VRAM for L/X; 3+ GB for small/medium variants",
            runnable_today=True,
            limitations=[
                "HF community checkpoints used for D-FINE (not official .pth).",
                "AP not benchmark-verified in CI environment.",
            ],
        ),
    },
    "sam-family": {
        "image-prompt": DomainRecipe(
            domain="sam-family",
            goal="image-prompt-segmentation",
            description="Prompt-based mask generation (point/box) on still images.",
            pipeline=[
                "1. Choose SAM size (sam2-hiera-tiny for speed; sam-vit-large for quality)",
                "2. Provide point or box prompt",
                "3. Receive mask + confidence",
            ],
            recommended_models=[
                "sam-vit-base",
                "sam-vit-large",
                "sam2-hiera-tiny",
                "sam2-hiera-base-plus",
            ],
            install_commands=["pip install 'visionservex[hf]'"],
            quick_commands=[
                "visionservex predict sam2-hiera-tiny image.jpg --point 320,240",
                "visionservex predict sam-vit-base image.jpg --box 50,60,300,400",
            ],
            expected_hardware="CPU OK for tiny; 4-8 GB VRAM for base-plus/large",
            runnable_today=True,
            limitations=[
                "SAM2.1 / SAM3 / SAM3.1 official checkpoints require verification (gated/audit).",
            ],
        ),
        "video-prompt-segmentation": DomainRecipe(
            domain="sam-family",
            goal="video-prompt-segmentation",
            description="Video object tracking with SAM2/SAM2.1 video predictor (not wired here).",
            pipeline=[
                "1. Frame sampling (manual)",
                "2. Initial mask from prompt on first frame",
                "3. Propagate mask through subsequent frames",
            ],
            recommended_models=["sam2-hiera-large", "sam2.1-hiera-large"],
            install_commands=["pip install 'visionservex[hf]'"],
            quick_commands=[],
            runnable_today=False,
            limitations=[
                "Video predictor not yet wired in VisionServeX.",
                "Use facebookresearch/sam2 video API directly for now.",
            ],
        ),
    },
    "promptable": {
        "open-vocab-detection": DomainRecipe(
            domain="promptable",
            goal="open-vocab-detection",
            description="Detect arbitrary objects by text description.",
            pipeline=[
                "1. Choose model (Grounding DINO for accuracy, OWLv2 for cleaner HF integration)",
                "2. Pass image + comma-separated text prompts",
                "3. Receive boxes + matched phrases + scores",
            ],
            recommended_models=[
                "grounding-dino-tiny",
                "grounding-dino-swin-b",
                "owlv2-base-patch16",
                "owlv2-large-patch14",
            ],
            install_commands=["pip install 'visionservex[hf]'"],
            quick_commands=[
                "visionservex open-vocab grounding-dino-swin-b image.jpg --prompt 'car,person'",
                "visionservex predict owlv2-base-patch16 image.jpg --prompt 'cat,dog'",
            ],
            expected_hardware="CPU OK for tiny; GPU for base/large",
            runnable_today=True,
            limitations=[
                "Grounding DINO 1.5/1.6 / DINO-X are API-gated.",
                "YOLO-World excluded due to GPL/AGPL license concerns.",
            ],
        ),
    },
    "feature-intelligence": {
        "retrieval-and-dedup": DomainRecipe(
            domain="feature-intelligence",
            goal="retrieval-and-dedup",
            description=(
                "Image-to-image similarity, dataset deduplication, near-duplicate detection. "
                "Powered by DINOv2 / SigLIP2 embeddings."
            ),
            pipeline=[
                "1. Embed images into N-dim vectors with DINOv2/SigLIP2",
                "2. Build sklearn/FAISS index",
                "3. Query top-k nearest neighbors or deduplicate by similarity threshold",
            ],
            recommended_models=[
                "dinov2-base",
                "dinov2-large",
                "siglip2-base-patch16-224",
            ],
            install_commands=["pip install 'visionservex[hf]'"],
            quick_commands=[
                "visionservex embed dinov2-base image.jpg --out embedding.npy",
                "visionservex index dinov2-base folder/ --out indexes/dinov2_base",
                "visionservex search dinov2-base query.jpg --index indexes/dinov2_base --top-k 10",
                "visionservex deduplicate dinov2-base folder/ --threshold 0.98 --out duplicates.csv",
            ],
            expected_hardware="CPU OK for small/base; GPU recommended for large/giant",
            runnable_today=True,
        ),
        "dataset-intelligence": DomainRecipe(
            domain="feature-intelligence",
            goal="dataset-intelligence",
            description=(
                "Dataset reports, active learning sample selection, train/test domain shift "
                "detection — all driven by DINOv2 features."
            ),
            pipeline=[
                "1. Embed train + test/unlabeled folders",
                "2. Compute mean similarity / cluster structure",
                "3. Recommend active-learning sample budget",
            ],
            recommended_models=["dinov2-base", "dinov2-large"],
            install_commands=["pip install 'visionservex[hf]'"],
            quick_commands=[
                "visionservex dataset-report dinov2-base folder/ --out dataset_report.md",
                "visionservex active-select dinov2-base folder/ --budget 100 --out selected.csv",
                "visionservex domain-shift dinov2-base train/ test/ --out domain_shift.md",
            ],
            runnable_today=True,
        ),
    },
    "surveillance": {
        "person-attribute-search": DomainRecipe(
            domain="surveillance",
            goal="person-attribute-search",
            description=(
                "Search a camera archive for people matching a text description "
                "(e.g. 'person wearing a red shirt'). Pipeline-based, not a single model."
            ),
            pipeline=[
                "1. Detect persons per frame (Grounding DINO / OWLv2 / D-FINE)",
                "2. Track across frames (ByteTrack — expert sidecar)",
                "3. Extract person crops + ReID embeddings (OSNet — expert sidecar)",
                "4. Match crops against text prompts via SigLIP2 / CLIP",
                "5. Aggregate by timestamp, return ranked timeline",
            ],
            recommended_models=[
                "grounding-dino-swin-b  # detection",
                "owlv2-base-patch16  # alternative detection",
                "siglip2-base-patch16-224  # text-image matching",
                "osnet-x1.0  # expert: person re-ID",
                "bytetrack  # expert: tracking",
            ],
            install_commands=[
                "pip install 'visionservex[hf]'",
                "# Expert sidecar (optional):",
                "pip install torchreid  # for ReID — NOT in core",
            ],
            quick_commands=[
                "# Roadmap (not yet implemented):",
                "visionservex video-search index camera_folder/ --detector grounding-dino-swin-b "
                "--embedder siglip2-base-patch16-224 --out indexes/camera01",
                "visionservex video-search query indexes/camera01 --text 'person wearing red shirt' "
                "--top-k 20 --out report.html",
            ],
            expected_hardware="GPU recommended; appearance-based search only (no identity claim)",
            runnable_today=False,
            limitations=[
                "Full video-search pipeline is roadmap v1.7.",
                "Detection + embedding components are runnable; tracker/ReID need expert install.",
                "No face identification. No biometric identity claim.",
                "Local-only by default. No public exposure.",
            ],
            license_notes=[
                "Apache-2.0 for VisionServeX components.",
                "Verify per-model: torchreid (MIT), ByteTrack (MIT).",
            ],
        ),
    },
    "industrial": {
        "defect-detection": DomainRecipe(
            domain="industrial",
            goal="defect-detection",
            description=(
                "Industrial anomaly detection using one-class methods (Anomalib PatchCore, PaDiM, "
                "EfficientAD) plus DINOv2 features as backbone."
            ),
            pipeline=[
                "1. Collect normal images (no defects)",
                "2. Train PatchCore on normal set (memorize feature distribution)",
                "3. Score test images by distance to normal memory bank",
                "4. Output anomaly heatmap + score",
            ],
            recommended_models=[
                "anomalib-patchcore  # expert: anomalib package",
                "dinov2-base  # feature backbone alternative",
            ],
            install_commands=[
                "pip install anomalib  # NOT in core",
                "pip install 'visionservex[hf]'  # for DINOv2-based custom anomaly",
            ],
            quick_commands=[
                "# Anomalib pipeline (roadmap):",
                "visionservex anomaly train patchcore --data normal/ --out runs/patchcore",
                "visionservex anomaly predict runs/patchcore test.jpg --save-heatmap out.png",
            ],
            runnable_today=False,
            limitations=[
                "Anomalib not wired in core. Use the anomalib package directly.",
                "DINOv2 + PatchCore recipe documented but not packaged.",
            ],
        ),
    },
    "medical": {
        "ct-segmentation": DomainRecipe(
            domain="medical",
            goal="ct-segmentation",
            description="3D CT segmentation. Domain-specific; not for general object detection.",
            pipeline=[
                "1. Load NIfTI/DICOM volume",
                "2. Run organ segmentation",
                "3. Export segmentation masks",
            ],
            recommended_models=[
                "totalsegmentator  # non_core_license_optional",
                "nnunet-v2  # framework, expert path",
            ],
            install_commands=[
                "pip install totalsegmentator  # NOT in core",
                "# OR for nnU-Net:",
                "pip install nnunetv2  # NOT in core",
            ],
            quick_commands=[
                "# Direct upstream usage (not via VisionServeX):",
                "TotalSegmentator -i input.nii.gz -o output/",
            ],
            runnable_today=False,
            limitations=[
                "VisionServeX does not bundle medical models due to license/regulatory care.",
                "Use upstream packages directly; do not use VisionServeX for medical diagnosis.",
            ],
            license_notes=[
                "TotalSegmentator: certain submodels have commercial restrictions.",
                "VisionServeX is not a medical device. No diagnostic claim.",
            ],
        ),
        "promptable-medical-segmentation": DomainRecipe(
            domain="medical",
            goal="promptable-medical-segmentation",
            description="Prompt-based segmentation for medical 2D imaging.",
            pipeline=[
                "1. Provide box prompt around region of interest",
                "2. Get binary mask from MedSAM / MedSAM2",
            ],
            recommended_models=["medsam", "medsam2"],
            install_commands=[
                "pip install 'visionservex[hf]'  # MedSAM v1 (wanglab/medsam-vit-base)"
            ],
            runnable_today=True,
            limitations=[
                "MedSAM v1 is wired and runnable: `visionservex medical segment medsam IMG --box x1,y1,x2,y2`.",
                "MedSAM2 is a RESEARCH-ONLY expert sidecar (non-commercial weights); not runnable in core.",
                "Research/education only — not for diagnosis. MedSAM weights are not commercial-safe by default.",
            ],
        ),
    },
    "agriculture": {
        "weed-detection": DomainRecipe(
            domain="agriculture",
            goal="weed-detection",
            description="Detect weeds vs crops. Best approach: fine-tune D-FINE/RF-DETR on dataset.",
            pipeline=[
                "1. Collect annotated crop/weed dataset (YOLO format)",
                "2. Fine-tune RF-DETR or D-FINE",
                "3. Deploy for inference",
            ],
            recommended_models=[
                "rfdetr-small  # fine-tunable",
                "dfine-s-o365-coco  # transfer-learning starting point",
                "grounding-dino-swin-b  # zero-shot with text prompt 'weed'",
            ],
            install_commands=[
                "pip install 'visionservex[rfdetr]'",
                "pip install 'visionservex[hf]'",
            ],
            quick_commands=[
                "# Zero-shot prompt:",
                "visionservex open-vocab grounding-dino-swin-b field.jpg --prompt 'weed,crop'",
                "# Fine-tuning (RF-DETR):",
                "# See https://rfdetr.roboflow.com/ for training recipe",
            ],
            runnable_today=True,
            limitations=[
                "AgriCLIP and SCOLD not yet wired (license/checkpoint audit pending).",
                "Best results require fine-tuning on domain dataset.",
            ],
        ),
    },
    "aerial": {
        "oriented-detection": DomainRecipe(
            domain="aerial",
            goal="oriented-detection",
            description="Rotated bounding boxes for aerial/satellite imagery.",
            pipeline=[
                "1. Install OpenMMLab MMRotate (expert sidecar)",
                "2. Run RTMDet-R or RTMDet-R2 oriented detection",
                "3. Get rotated boxes (xywhr)",
            ],
            recommended_models=[
                "rtmdet-r2-s",
                "rtmdet-r2-l",
            ],
            install_commands=[
                "pip install openmim",
                "mim install mmengine mmcv mmrotate",
            ],
            quick_commands=[
                "# Expert sidecar (roadmap):",
                "visionservex openmmlab pull rtmdet-r2-s",
                "visionservex openmmlab smoke-test rtmdet-r2-s",
            ],
            runnable_today=False,
            limitations=[
                "OpenMMLab native inference path not yet wired in VisionServeX.",
                "No verified YOLO-OBB winner unless benchmark proves it.",
            ],
        ),
    },
}


def list_domains() -> list[str]:
    """Return all domain names."""
    return sorted(DOMAIN_ZOO.keys())


def recommend_for_domain(domain: str, goal: str | None = None) -> list[DomainRecipe]:
    """Return recipes for a domain, optionally filtered by goal.

    Goal matching is fuzzy: substring match against the recipe's goal key.
    """
    domain_l = domain.lower()
    if domain_l not in DOMAIN_ZOO:
        return []
    goals = DOMAIN_ZOO[domain_l]
    if goal:
        goal_l = goal.lower()
        # Fuzzy: keyword match
        matched = [
            recipe
            for key, recipe in goals.items()
            if goal_l in key
            or goal_l in recipe.description.lower()
            or goal_l in recipe.goal.lower()
        ]
        if matched:
            return matched
    return list(goals.values())


__all__ = ["DOMAIN_ZOO", "DomainRecipe", "list_domains", "recommend_for_domain"]

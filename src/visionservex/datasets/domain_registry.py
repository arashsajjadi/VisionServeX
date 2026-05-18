# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.25.0: Specialized domain dataset registry.

The pre-v2.25 audit conflated "running on COCO128" with "benchmarking
medical/agriculture/aerial models". v2.25 ships a first-class registry
that defines, per domain task, the dataset layout we accept, the ground
truth required for each metric, and the legal default (commercial vs
non-commercial).

If a dataset doesn't satisfy a domain's contract, the validator returns a
structured ``expected_blocker`` and the model is classified ``smoke`` or
``demo`` — never ``benchmark`` with a fake AP/Dice number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DOMAIN_REGISTRY",
    "DomainDataset",
    "list_domain_datasets",
    "validate_domain_path",
]


@dataclass
class DomainDataset:
    """One specialized domain dataset contract."""

    domain: str  # medical | agriculture | aerial | industrial | surveillance | video_search | pose | panoptic
    task: str  # e.g. "2d-box", "nifti-seg", "hbb", "obb", "tracking"
    accepted_formats: list[str] = field(default_factory=list)
    required_files: list[str] = field(default_factory=list)
    required_gt: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    legal_default: str = "open"  # open | restricted | non_commercial
    noncommercial_risks: list[str] = field(default_factory=list)
    smoke_dataset_generator: str = ""  # CLI command that creates a tiny smoke
    scientific_dataset_validator: str = ""  # CLI command that validates real data
    benchmark_allowed_default: bool = False
    expected_blocker_if_missing: str = "DATASET_REQUIRED"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task": self.task,
            "accepted_formats": list(self.accepted_formats),
            "required_files": list(self.required_files),
            "required_gt": list(self.required_gt),
            "metrics": list(self.metrics),
            "legal_default": self.legal_default,
            "noncommercial_risks": list(self.noncommercial_risks),
            "smoke_dataset_generator": self.smoke_dataset_generator,
            "scientific_dataset_validator": self.scientific_dataset_validator,
            "benchmark_allowed_default": self.benchmark_allowed_default,
            "expected_blocker_if_missing": self.expected_blocker_if_missing,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, DomainDataset] = {
    "medical/2d-box": DomainDataset(
        domain="medical",
        task="2d-box",
        accepted_formats=["yolo-style images + boxes.json", "DICOM (PNG-rendered)"],
        required_files=["images/*.(jpg|png|tif)", "boxes.json"],
        required_gt=["axis-aligned bounding boxes per image"],
        metrics=["box_count", "mask_iou_per_image (if masks/)", "latency", "visual_overlay"],
        legal_default="open",
        noncommercial_risks=[],
        smoke_dataset_generator="visionservex dataset generate-synthetic medical-nifti --n-samples 3",
        scientific_dataset_validator="visionservex dataset validate-domain medical --path PATH --task 2d-box",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="BOX_PROMPTS_REQUIRED",
        notes="MedSAM/MedSAM2 require box prompts. Without GT masks only visual smoke is reported.",
    ),
    "medical/nifti-seg": DomainDataset(
        domain="medical",
        task="nifti-seg",
        accepted_formats=[".nii", ".nii.gz", ".mha", ".mhd"],
        required_files=["*.(nii|nii.gz|mha|mhd)"],
        required_gt=["volumetric segmentation labels"],
        metrics=["dice_per_class", "iou_per_class"],
        legal_default="restricted",
        noncommercial_risks=["Some TotalSegmentator submodels have commercial restrictions"],
        smoke_dataset_generator="visionservex dataset generate-synthetic medical-nifti --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain medical --path PATH --task nifti-seg",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="NIFTI_REQUIRED",
        notes="TotalSegmentator + nnU-Net v2 require trained weights AND patient NIfTI volumes.",
    ),
    "agriculture/hbb": DomainDataset(
        domain="agriculture",
        task="hbb",
        accepted_formats=["YOLO-format (images/ + labels/ + data.yaml)"],
        required_files=["images/", "labels/", "data.yaml"],
        required_gt=["YOLO box labels per image"],
        metrics=["AP50", "mAP50:95", "precision", "recall"],
        legal_default="open",
        noncommercial_risks=["PlantVillage is non-commercial — do NOT auto-pull"],
        smoke_dataset_generator="visionservex dataset generate-synthetic agriculture-hbb --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain agriculture --path PATH --task hbb",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="AGRI_LABELS_REQUIRED",
        notes="Custom crop/weed datasets only; never use COCO80 detector on PlantVillage scientifically.",
    ),
    "agriculture/segmentation": DomainDataset(
        domain="agriculture",
        task="segmentation",
        accepted_formats=["YOLO-seg / COCO-stuff style"],
        required_files=["images/", "annotations/"],
        required_gt=["per-pixel mask or polygon"],
        metrics=["mIoU", "Dice"],
        legal_default="open",
        noncommercial_risks=[],
        smoke_dataset_generator="visionservex dataset generate-synthetic agriculture-hbb --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain agriculture --path PATH --task segmentation",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="AGRI_SEG_LABELS_REQUIRED",
        notes="Crop/weed segmentation requires real masks; synthetic data is smoke only.",
    ),
    "aerial/hbb": DomainDataset(
        domain="aerial",
        task="hbb",
        accepted_formats=["YOLO-format"],
        required_files=["images/", "labels/"],
        required_gt=["axis-aligned boxes"],
        metrics=["AP50", "mAP50:95"],
        legal_default="open",
        noncommercial_risks=["VisDrone is non-commercial — do NOT auto-pull"],
        smoke_dataset_generator="visionservex dataset generate-synthetic aerial-obb --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain aerial --path PATH --task hbb",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="AERIAL_LABELS_REQUIRED",
        notes="HBB on aerial works on permissive sources; gate VisDrone behind explicit opt-in.",
    ),
    "aerial/obb": DomainDataset(
        domain="aerial",
        task="obb",
        accepted_formats=["DOTA labelTxt (8 floats + class)", "YOLO-OBB"],
        required_files=["labelTxt/*.txt OR labels/*.txt"],
        required_gt=["oriented bounding boxes"],
        metrics=["OBB_AP50", "OBB_mAP50:95"],
        legal_default="non_commercial",
        noncommercial_risks=[
            "DOTA-v1/2 — academic only; license forbids commercial use without permission"
        ],
        smoke_dataset_generator="visionservex dataset generate-synthetic aerial-obb --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain aerial --path PATH --task obb",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="DOTA_OR_OBB_LABELS_REQUIRED",
        notes="OBB AP cannot be computed from axis-aligned labels.",
    ),
    "industrial/anomaly-simple": DomainDataset(
        domain="industrial",
        task="anomaly-simple",
        accepted_formats=["train/normal + test/{normal,defect} folder schema"],
        required_files=["train/normal/", "test/normal/", "test/defect/ (optional)"],
        required_gt=["binary normal/defect labels OR per-image score targets"],
        metrics=["image_AUROC", "pixel_AUROC (if masks)"],
        legal_default="non_commercial",
        noncommercial_risks=[
            "MVTec AD is non-commercial (CC BY-NC-SA 4.0)",
            "VisA is non-commercial",
        ],
        smoke_dataset_generator="visionservex dataset generate-synthetic anomaly-defect --n-samples 3",
        scientific_dataset_validator="visionservex dataset validate-domain anomaly --path PATH --task simple",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="ANOMALY_LABELS_REQUIRED",
        notes="MVTec/VisA gated by license; user must opt-in and acknowledge non-commercial scope.",
    ),
    "surveillance/tracking": DomainDataset(
        domain="surveillance",
        task="tracking",
        accepted_formats=[
            "MOT17/20 (gt.txt) + sequence frames",
            "video files (mp4/avi) + GT tracks",
        ],
        required_files=["seq*/img1/", "seq*/gt/gt.txt"],
        required_gt=["frame_id, track_id, bbox per detection (MOT format)"],
        metrics=["MOTA", "IDF1", "HOTA"],
        legal_default="non_commercial",
        noncommercial_risks=["DukeMTMC retracted (privacy)", "MOT17/20 academic only"],
        smoke_dataset_generator="visionservex dataset generate-synthetic tracking-video --n-samples 2",
        scientific_dataset_validator="visionservex dataset validate-domain surveillance --path PATH --task tracking",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="GT_TRACKS_OR_QUERY_LABELS_REQUIRED",
        notes="Tracking metrics need GT tracks; without them only demo annotation is honest.",
    ),
    "video-search/retrieval": DomainDataset(
        domain="video_search",
        task="retrieval",
        accepted_formats=["video corpus + query/positive pairs"],
        required_files=["video corpus/", "query_pos.csv"],
        required_gt=["per-query positive/negative pair labels"],
        metrics=["recall@K", "mAP"],
        legal_default="open",
        noncommercial_risks=[],
        smoke_dataset_generator="",
        scientific_dataset_validator="visionservex dataset validate-domain video-search --path PATH --task retrieval",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="GT_RETRIEVAL_PAIRS_REQUIRED",
        notes="Retrieval AP requires labeled query/positive pairs; without them only embedding smoke.",
    ),
    "pose/keypoints": DomainDataset(
        domain="pose",
        task="keypoints",
        accepted_formats=["COCO keypoints (annotations/person_keypoints_*.json)"],
        required_files=["images/", "annotations/person_keypoints_val2017.json"],
        required_gt=["17-keypoint COCO format per person"],
        metrics=["OKS_AP", "OKS_AP50", "OKS_AP75"],
        legal_default="open",
        noncommercial_risks=[],
        smoke_dataset_generator="",
        scientific_dataset_validator="visionservex dataset validate-domain pose --path PATH --task keypoints",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="KEYPOINT_LABELS_REQUIRED",
        notes="Pose AP needs OKS-style keypoint GT; visual overlay alone is smoke only.",
    ),
    "panoptic/segmentation": DomainDataset(
        domain="panoptic",
        task="segmentation",
        accepted_formats=["COCO Panoptic (panoptic_*.json + panoptic_*/*.png)"],
        required_files=["panoptic_val2017.json", "panoptic_val2017/"],
        required_gt=["per-pixel segment_info + thing/stuff classes"],
        metrics=["PQ", "PQ_thing", "PQ_stuff"],
        legal_default="open",
        noncommercial_risks=[],
        smoke_dataset_generator="",
        scientific_dataset_validator="visionservex dataset validate-domain panoptic --path PATH --task segmentation",
        benchmark_allowed_default=False,
        expected_blocker_if_missing="PANOPTIC_LABELS_REQUIRED",
        notes="Panoptic Quality requires panoptic annotations; semseg masks alone are insufficient.",
    ),
}


def list_domain_datasets() -> dict[str, DomainDataset]:
    return dict(DOMAIN_REGISTRY)


def _count_files(root: Path, exts: set[str]) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def validate_domain_path(domain: str, task: str, path: Path) -> dict[str, Any]:
    """Validate that ``path`` satisfies ``DOMAIN_REGISTRY[domain/task]``.

    Returns a structured payload with ``status``, ``code``, ``metrics_valid``,
    ``benchmark_or_smoke``, and ``details``. Never raises on bad input.
    """
    key = f"{domain}/{task}"
    spec = DOMAIN_REGISTRY.get(key)
    if spec is None:
        return {
            "status": "failed",
            "code": "DOMAIN_TASK_NOT_REGISTERED",
            "domain": domain,
            "task": task,
            "message": f"No domain dataset spec for {key!r}.",
            "known_keys": sorted(DOMAIN_REGISTRY.keys()),
        }
    if not path.exists():
        return {
            "status": "expected_blocker",
            "code": spec.expected_blocker_if_missing,
            "domain": domain,
            "task": task,
            "path": str(path),
            "spec": spec.to_dict(),
            "metrics_valid": False,
            "benchmark_or_smoke": "expected_blocker",
            "message": f"Path {path} not found.",
        }

    # Domain-specific structural checks (lightweight; never load heavy data).
    details: dict[str, Any] = {}
    if domain == "medical" and task == "2d-box":
        details["n_images"] = _count_files(
            path / "images" if (path / "images").exists() else path,
            {".jpg", ".jpeg", ".png", ".tif", ".tiff"},
        )
        details["has_boxes_json"] = (path / "boxes.json").exists()
        details["has_masks"] = (path / "masks").exists()
        metrics_valid = details["has_boxes_json"] and details["n_images"] > 0
    elif domain == "medical" and task == "nifti-seg":
        n_nifti = 0
        for p in path.rglob("*"):
            if p.is_file() and (
                p.name.endswith(".nii")
                or p.name.endswith(".nii.gz")
                or p.suffix.lower() in {".mha", ".mhd"}
            ):
                n_nifti += 1
        details["n_nifti"] = n_nifti
        metrics_valid = n_nifti > 0
    elif domain == "agriculture":
        details["has_images_dir"] = (path / "images").exists()
        details["has_labels_dir"] = (path / "labels").exists()
        details["has_data_yaml"] = (path / "data.yaml").exists()
        metrics_valid = all(details.values())
    elif domain == "aerial" and task == "obb":
        details["has_labelTxt"] = (path / "labelTxt").exists()
        details["has_labels"] = (path / "labels").exists()
        metrics_valid = details["has_labelTxt"] or details["has_labels"]
    elif domain == "aerial":
        details["has_images_dir"] = (path / "images").exists()
        details["has_labels_dir"] = (path / "labels").exists()
        metrics_valid = details["has_images_dir"] and details["has_labels_dir"]
    elif domain == "industrial":
        details["has_train_normal"] = (path / "train" / "normal").exists()
        details["has_test_normal"] = (path / "test" / "normal").exists()
        details["has_test_defect"] = (path / "test" / "defect").exists()
        metrics_valid = details["has_train_normal"] and details["has_test_normal"]
    elif domain == "surveillance":
        seqs = [p for p in path.iterdir() if p.is_dir() and (p / "gt" / "gt.txt").exists()]
        details["n_seqs"] = len(seqs)
        metrics_valid = len(seqs) > 0
    elif domain == "video_search":
        details["has_query_pos_csv"] = (path / "query_pos.csv").exists()
        metrics_valid = details["has_query_pos_csv"]
    elif domain == "pose":
        details["has_keypoints_json"] = (
            path / "annotations" / "person_keypoints_val2017.json"
        ).exists()
        metrics_valid = details["has_keypoints_json"]
    elif domain == "panoptic":
        details["has_panoptic_json"] = any(path.glob("panoptic_*.json"))
        metrics_valid = details["has_panoptic_json"]
    else:
        metrics_valid = False

    benchmark_or_smoke = "benchmark" if metrics_valid else "smoke"
    code = "OK" if metrics_valid else spec.expected_blocker_if_missing
    status = "ok" if metrics_valid else "expected_blocker"

    return {
        "status": status,
        "code": code,
        "domain": domain,
        "task": task,
        "path": str(path),
        "spec": spec.to_dict(),
        "details": details,
        "metrics_valid": metrics_valid,
        "benchmark_or_smoke": benchmark_or_smoke,
        "message": (
            f"Domain {domain}/{task} dataset {('valid' if metrics_valid else 'incomplete')}."
        ),
    }

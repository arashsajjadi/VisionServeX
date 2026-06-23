# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Medical training-truth: machine-readable matrix + dataset validator + dry-runs.

No fake training. This module exposes (1) a strict, machine-readable capability
matrix that states exactly what is trainable where, (2) a real 2D segmentation
dataset validator (tested on tiny synthetic fixtures), and (3) dry-run generators
that VALIDATE a dataset and emit the EXACT upstream training command — without
running any training. A model is never reported as VSX-fine-tunable.

Statuses:
    NOT_TRAINABLE_BY_DESIGN     - foundation segmenter, inference only
    NOT_TRAINABLE_IN_VSX        - trainable upstream, no VSX path
    EXTERNAL_TRAINING_ONLY      - train via the upstream repo only
    TRAINING_FRAMEWORK_EXTERNAL - a framework (run its own trainer)
    ACTIVE_DRY_RUN              - VSX validates data + emits the exact command
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

#: Machine-readable training capability matrix (single source of truth).
TRAINING_MATRIX: dict[str, dict[str, Any]] = {
    "medsam": {
        "status": "NOT_TRAINABLE_BY_DESIGN",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "upstream": "https://github.com/bowang-lab/MedSAM",
        "notes": "Foundation promptable segmenter — inference only in VisionServeX.",
    },
    "medsam2": {
        "status": "EXTERNAL_TRAINING_ONLY",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "dry_run_supported": True,
        "upstream": "https://github.com/bowang-lab/MedSAM2",
        "notes": "Fine-tune via the upstream MedSAM2 repo (training/finetune scripts). "
        "VisionServeX offers dataset validation + exact command generation only "
        "(ACTIVE_DRY_RUN); it does NOT train MedSAM2 in-process. Weights non-commercial.",
    },
    "nnunet-v2": {
        "status": "TRAINING_FRAMEWORK_EXTERNAL",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "dry_run_supported": True,
        "required_module": "nnunetv2",
        "upstream": "https://github.com/MIC-DKFZ/nnUNet",
        "notes": "Self-configuring framework; train with its own CLI. VSX validates "
        "the dataset and emits the exact nnUNetv2 commands (ACTIVE_DRY_RUN).",
    },
    "monai": {
        "status": "TRAINING_FRAMEWORK_EXTERNAL",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "dry_run_supported": True,
        "required_module": "monai",
        "upstream": "https://github.com/Project-MONAI/MONAI",
        "notes": "Framework (incl. SwinUNETR/UNETR/Auto3DSeg). VSX validates data and "
        "emits the exact MONAI bundle/Auto3DSeg command (ACTIVE_DRY_RUN).",
    },
    "swinunetr": {
        "status": "TRAINING_FRAMEWORK_EXTERNAL",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "upstream": "https://github.com/Project-MONAI/research-contributions",
        "notes": "Only via a real MONAI trainer/config; no standalone VSX trainer.",
    },
    "sam2": {
        "status": "NOT_TRAINABLE_IN_VSX",
        "trainable_in_vsx": False,
        "finetunable_in_vsx": False,
        "upstream": "https://github.com/facebookresearch/sam2",
        "notes": "General promptable segmenter; no VSX training path.",
    },
}

_DRY_RUN_FRAMEWORKS = ("nnunet", "monai", "medsam2")


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def train_doctor() -> dict[str, Any]:
    """Report honest training status + which frameworks are importable."""
    return {
        "matrix": TRAINING_MATRIX,
        "frameworks_present": {
            "nnunetv2": _module_present("nnunetv2"),
            "monai": _module_present("monai"),
            "sam2": _module_present("sam2"),
        },
        "vsx_trains_any_model": False,
        "note": "VisionServeX does not train/fine-tune medical models in-process. "
        "Dry-run = dataset validation + exact upstream command generation.",
    }


_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_segmentation_dataset(
    dataset_dir: str | Path, *, task: str = "segmentation"
) -> dict[str, Any]:
    """Validate a 2D segmentation dataset: images/ + masks/ paired by filename stem.

    Real, side-effect-free contract check (no training). Returns a structured
    report: validity, pair count, missing masks, orphan masks, shape mismatches.
    """
    root = Path(dataset_dir)
    report: dict[str, Any] = {
        "dataset_dir": str(root),
        "task": task,
        "valid": False,
        "n_pairs": 0,
        "errors": [],
        "warnings": [],
    }
    if task != "segmentation":
        report["errors"].append(f"unsupported task {task!r}; only 'segmentation' is validated")
        return report

    img_dir = root / "images"
    mask_dir = root / "masks"
    if not img_dir.is_dir():
        report["errors"].append(f"missing images/ directory under {root}")
    if not mask_dir.is_dir():
        report["errors"].append(f"missing masks/ directory under {root}")
    if report["errors"]:
        return report

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_EXT}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in _IMG_EXT}
    if not images:
        report["errors"].append("no images found in images/")
    missing = sorted(set(images) - set(masks))
    orphans = sorted(set(masks) - set(images))
    if missing:
        report["errors"].append(f"images without a matching mask: {missing[:10]}")
    if orphans:
        report["warnings"].append(f"masks without a matching image: {orphans[:10]}")

    paired = sorted(set(images) & set(masks))
    report["n_pairs"] = len(paired)

    # Sample-check shapes on up to 5 pairs (cheap; avoids loading the whole set).
    try:
        import numpy as np
        from PIL import Image

        for stem in paired[:5]:
            im = np.array(Image.open(images[stem]))
            mk = np.array(Image.open(masks[stem]))
            if im.shape[:2] != mk.shape[:2]:
                report["errors"].append(
                    f"shape mismatch for {stem!r}: image {im.shape[:2]} vs mask {mk.shape[:2]}"
                )
    except Exception as exc:
        report["warnings"].append(f"could not sample-check shapes: {exc}")

    report["valid"] = (report["n_pairs"] > 0) and not report["errors"]
    return report


def dry_run(
    framework: str, dataset_dir: str | Path, out_dir: str | Path, *, dataset_id: int = 101
) -> dict[str, Any]:
    """Validate a dataset and emit the EXACT upstream training command (no training)."""
    framework = framework.lower()
    if framework not in _DRY_RUN_FRAMEWORKS:
        return {
            "status": "failed",
            "code": "UNKNOWN_FRAMEWORK",
            "message": f"unknown framework {framework!r}; one of {list(_DRY_RUN_FRAMEWORKS)}",
        }
    validation = validate_segmentation_dataset(dataset_dir, task="segmentation")
    out = str(out_dir)
    if framework == "nnunet":
        commands = [
            f"nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity",
            f"nnUNetv2_train {dataset_id} 2d 0",
        ]
        matrix_key = "nnunet-v2"
    elif framework == "monai":
        commands = [
            f"python -m monai.apps.auto3dseg AutoRunner run --input {dataset_dir} --output {out}",
        ]
        matrix_key = "monai"
    else:  # medsam2
        commands = [
            "git clone https://github.com/bowang-lab/MedSAM2 && cd MedSAM2 && pip install -e .",
            f"python finetune_sam2_img.py --dataset {dataset_dir} --output {out}  # see upstream README",
        ]
        matrix_key = "medsam2"
    return {
        "status": "dry_run",
        "framework": framework,
        "matrix_status": TRAINING_MATRIX[matrix_key]["status"],
        "trains_in_vsx": False,
        "dataset_validation": validation,
        "generated_commands": commands,
        "out_dir": out,
        "note": "VisionServeX does NOT run this training. Commands are for the upstream tool.",
    }


__all__ = [
    "TRAINING_MATRIX",
    "dry_run",
    "train_doctor",
    "validate_segmentation_dataset",
]

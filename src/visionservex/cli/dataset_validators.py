# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.18.0: domain dataset validators.

The notebook v19/v20 conflated "running on COCO128" with "benchmarking
medical/agriculture/aerial models". v2.18 ships explicit per-domain
validators that inspect a dataset folder and return a structured verdict.

CLI surface::

    visionservex dataset validate-medical     --path DIR --task medsam-2d-box   --format json --out X
    visionservex dataset validate-agriculture --path DIR --task weed-detection  --format yolo --out X
    visionservex dataset validate-aerial      --path DIR --dataset-type dota    --format json --out X
    visionservex dataset validate-anomaly     --path DIR --schema mvtec         --format json --out X
    visionservex dataset validate-surveillance --path DIR_OR_VIDEO              --format json --out X

Every validator returns the same envelope::

    {
      "status": "ok|partial|expected_blocker|failed",
      "dataset_type": "...",
      "n_images": ..., "n_videos": ..., "n_labels": ..., "n_masks": ...,
      "metrics_possible": [...],
      "metrics_blocked": [...],
      "blocker_code": "..." | null,
      "remediation": "...",
      "details": { ... }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    help="v2.18.0: domain dataset validators (medical/agriculture/aerial/anomaly/surveillance).",
    no_args_is_help=True,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_NIFTI_EXTS = {".nii", ".gz", ".mha", ".mhd"}


def _emit(payload: dict[str, Any], *, out: Path | None, fmt: str) -> None:
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
    if fmt == "json":
        typer.echo(json.dumps(payload, indent=2))
        return
    color = {
        "ok": "green",
        "partial": "yellow",
        "expected_blocker": "yellow",
        "failed": "red",
    }.get(payload.get("status", "failed"), "white")
    console.print(
        f"[{color}]{payload.get('status', '').upper()}[/{color}]: {payload.get('remediation', '')}"
    )


def _count_files(root: Path, exts: set[str]) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def _count_nifti(root: Path) -> int:
    if not root.exists() or not root.is_dir():
        return 0
    n = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.endswith(".nii") or name.endswith(".nii.gz") or name.endswith(".mha"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Medical
# ---------------------------------------------------------------------------


@app.command("validate-medical")
def validate_medical(
    path: Path = typer.Option(..., "--path", help="Dataset folder."),
    task: str = typer.Option(
        "medsam-2d-box",
        "--task",
        help="medsam-2d-box | totalsegmentator | nnunet-v2",
    ),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Validate a medical-imaging dataset folder."""
    if not path.exists():
        _emit(
            {
                "status": "failed",
                "code": "PATH_NOT_FOUND",
                "dataset_type": "medical",
                "task": task,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["dice", "iou"],
                "blocker_code": "PATH_NOT_FOUND",
                "remediation": f"Path not found: {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    images_dir = path / "images" if (path / "images").exists() else path
    boxes_file = path / "boxes.json"
    masks_dir = path / "masks" if (path / "masks").exists() else None

    n_images = _count_files(images_dir, _IMAGE_EXTS)
    n_masks = _count_files(masks_dir, _IMAGE_EXTS) if masks_dir is not None else 0
    n_nifti = _count_nifti(path)

    if task in ("totalsegmentator", "nnunet-v2"):
        if n_nifti == 0:
            payload = {
                "status": "expected_blocker",
                "code": "NIFTI_REQUIRED",
                "dataset_type": "medical_3d",
                "task": task,
                "n_images": n_images,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "n_nifti_files": 0,
                "metrics_possible": [],
                "metrics_blocked": ["dice", "iou"],
                "blocker_code": "NIFTI_REQUIRED",
                "remediation": (f"task={task} needs .nii / .nii.gz volumes under {path}; found 0."),
                "details": {"path": str(path)},
            }
            _emit(payload, out=out, fmt=fmt)
            return
        _emit(
            {
                "status": "ok",
                "code": "OK",
                "dataset_type": "medical_3d",
                "task": task,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "n_nifti_files": n_nifti,
                "metrics_possible": ["dice_per_class", "iou_per_class"],
                "metrics_blocked": [],
                "blocker_code": None,
                "remediation": "",
                "details": {"path": str(path), "nifti_count": n_nifti},
            },
            out=out,
            fmt=fmt,
        )
        return

    # 2D MedSAM-style validation
    if n_images == 0:
        _emit(
            {
                "status": "failed",
                "code": "NO_IMAGES_FOUND",
                "dataset_type": "medical_2d",
                "task": task,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["dice", "iou"],
                "blocker_code": "NO_IMAGES_FOUND",
                "remediation": f"No images found under {images_dir}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    boxes_present = boxes_file.exists()
    metrics_possible: list[str] = ["mask_count", "latency", "visual_overlay"]
    metrics_blocked: list[str] = []
    if n_masks > 0:
        metrics_possible.extend(["mask_iou_per_image", "mask_dice_per_image"])
    else:
        metrics_blocked.extend(["mask_iou", "mask_dice"])

    status = "ok" if boxes_present else "partial"
    blocker_code = None if boxes_present else "BOX_PROMPTS_REQUIRED"
    remediation = (
        ""
        if boxes_present
        else (
            "MedSAM needs box prompts. Add boxes.json with "
            '[{"image_id": "...", "boxes": [[x1,y1,x2,y2], ...]}, ...].'
        )
    )

    _emit(
        {
            "status": status,
            "code": "OK" if status == "ok" else "BOX_PROMPTS_REQUIRED",
            "dataset_type": "medical_2d",
            "task": task,
            "n_images": n_images,
            "n_videos": 0,
            "n_labels": 0,
            "n_masks": n_masks,
            "metrics_possible": metrics_possible,
            "metrics_blocked": metrics_blocked,
            "blocker_code": blocker_code,
            "remediation": remediation,
            "details": {
                "path": str(path),
                "boxes_file_present": boxes_present,
                "masks_dir": str(masks_dir) if masks_dir else None,
            },
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Agriculture
# ---------------------------------------------------------------------------


@app.command("validate-agriculture")
def validate_agriculture(
    path: Path = typer.Option(..., "--path"),
    task: str = typer.Option("weed-detection", "--task"),
    fmt_in: str = typer.Option("yolo", "--dataset-format", help="yolo | coco-json | folder"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Validate an agriculture detection/segmentation dataset."""
    if not path.exists():
        _emit(
            {
                "status": "failed",
                "code": "PATH_NOT_FOUND",
                "dataset_type": "agriculture",
                "task": task,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["ap50"],
                "blocker_code": "PATH_NOT_FOUND",
                "remediation": f"Path not found: {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    images_dir = path / "images" if (path / "images").exists() else path
    labels_dir = path / "labels" if (path / "labels").exists() else None
    n_images = _count_files(images_dir, _IMAGE_EXTS)
    n_labels = _count_files(labels_dir, {".txt"}) if labels_dir is not None else 0

    if n_images == 0:
        _emit(
            {
                "status": "failed",
                "code": "NO_IMAGES_FOUND",
                "dataset_type": "agriculture",
                "task": task,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["ap50"],
                "blocker_code": "NO_IMAGES_FOUND",
                "remediation": f"No images found under {images_dir}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    has_labels = n_labels > 0
    status = "ok" if has_labels else "partial"
    blocker = None if has_labels else "LABELS_REQUIRED_FOR_METRICS"
    metrics_possible: list[str] = ["visual_overlay", "detection_count"]
    metrics_blocked: list[str] = []
    if has_labels:
        metrics_possible.extend(["ap50", "ap75", "map50_95"])
    else:
        metrics_blocked.extend(["ap50", "ap75", "map50_95"])

    _emit(
        {
            "status": status,
            "code": "OK" if has_labels else "LABELS_REQUIRED_FOR_METRICS",
            "dataset_type": "agriculture",
            "task": task,
            "dataset_format": fmt_in,
            "n_images": n_images,
            "n_videos": 0,
            "n_labels": n_labels,
            "n_masks": 0,
            "metrics_possible": metrics_possible,
            "metrics_blocked": metrics_blocked,
            "blocker_code": blocker,
            "remediation": (
                ""
                if has_labels
                else f"No YOLO label files in {labels_dir}. Add labels/<image_stem>.txt with "
                "`class_id cx cy w h` (normalized)."
            ),
            "details": {"path": str(path), "labels_dir": str(labels_dir) if labels_dir else None},
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Aerial
# ---------------------------------------------------------------------------


@app.command("validate-aerial")
def validate_aerial(
    path: Path = typer.Option(..., "--path"),
    dataset_type: str = typer.Option(
        "generic-yolo",
        "--dataset-type",
        help="dota | visdrone | xview | generic-yolo",
    ),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Validate an aerial dataset (HBB / OBB)."""
    if not path.exists():
        _emit(
            {
                "status": "failed",
                "code": "PATH_NOT_FOUND",
                "dataset_type": dataset_type,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["ap50", "obb_ap50"],
                "blocker_code": "PATH_NOT_FOUND",
                "remediation": f"Path not found: {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    images_dir = path / "images" if (path / "images").exists() else path
    labels_dir = path / "labels" if (path / "labels").exists() else None
    annfile_obb = path / "labelTxt"  # DOTA convention
    n_images = _count_files(images_dir, _IMAGE_EXTS)
    n_labels = _count_files(labels_dir, {".txt"}) if labels_dir is not None else 0
    n_obb = _count_files(annfile_obb, {".txt"}) if annfile_obb.exists() else 0

    has_obb = n_obb > 0 and dataset_type == "dota"
    has_hbb = n_labels > 0

    metrics_possible: list[str] = ["visual_overlay", "detection_count"]
    metrics_blocked: list[str] = []
    if has_obb:
        metrics_possible.extend(["obb_ap50", "obb_map50_95"])
    elif dataset_type == "dota":
        metrics_blocked.append("obb_ap50")
    if has_hbb:
        metrics_possible.extend(["ap50", "ap75", "map50_95"])
    else:
        metrics_blocked.extend(["ap50", "ap75", "map50_95"])

    if dataset_type == "dota" and not has_obb:
        status = "expected_blocker"
        blocker = "DOTA_OR_OBB_LABELS_REQUIRED"
        remediation = (
            "DOTA OBB labels expected under labelTxt/. Provide rotated annotations or "
            "use --dataset-type visdrone for axis-aligned HBB labels."
        )
    elif not has_hbb and not has_obb:
        status = "expected_blocker"
        blocker = "AERIAL_LABELS_REQUIRED"
        remediation = (
            "No labels found. Provide YOLO HBB labels (labels/*.txt) or DOTA OBB (labelTxt/*.txt)."
        )
    else:
        status = "ok"
        blocker = None
        remediation = ""

    _emit(
        {
            "status": status,
            "code": "OK" if status == "ok" else blocker,
            "dataset_type": dataset_type,
            "n_images": n_images,
            "n_videos": 0,
            "n_labels": n_labels,
            "n_obb_labels": n_obb,
            "n_masks": 0,
            "metrics_possible": metrics_possible,
            "metrics_blocked": metrics_blocked,
            "blocker_code": blocker,
            "remediation": remediation,
            "details": {"path": str(path)},
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Industrial Anomaly
# ---------------------------------------------------------------------------


@app.command("validate-anomaly")
def validate_anomaly(
    path: Path = typer.Option(..., "--path"),
    schema: str = typer.Option("simple", "--schema", help="mvtec | simple"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Validate an anomaly-detection dataset (MVTec or simple)."""
    if not path.exists():
        _emit(
            {
                "status": "failed",
                "code": "PATH_NOT_FOUND",
                "dataset_type": "anomaly",
                "schema": schema,
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["image_auroc"],
                "blocker_code": "PATH_NOT_FOUND",
                "remediation": f"Path not found: {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    n_normal = 0
    n_defect = 0
    n_masks = 0
    if schema == "mvtec":
        train_good = path / "train" / "good"
        test = path / "test"
        n_normal = _count_files(train_good, _IMAGE_EXTS)
        if test.exists():
            for sub in test.iterdir():
                if not sub.is_dir():
                    continue
                count = _count_files(sub, _IMAGE_EXTS)
                if sub.name.lower() == "good":
                    n_normal += count
                else:
                    n_defect += count
        gt = path / "ground_truth"
        n_masks = _count_files(gt, _IMAGE_EXTS) if gt.exists() else 0
    else:  # simple
        normal = path / "normal"
        test_dir = path / "test"
        n_normal = _count_files(normal, _IMAGE_EXTS)
        n_defect = _count_files(test_dir, _IMAGE_EXTS)

    has_normal = n_normal > 0
    has_defect = n_defect > 0
    metrics_possible: list[str] = []
    metrics_blocked: list[str] = []
    blocker = None
    status = "ok"
    if not has_normal:
        blocker = "NORMAL_IMAGES_REQUIRED"
        status = "expected_blocker"
        metrics_blocked.extend(["image_auroc", "pixel_auroc"])
    if not has_defect:
        if blocker is None:
            blocker = "TEST_IMAGES_REQUIRED"
            status = "partial"
        metrics_blocked.append("image_auroc")
    if has_normal and has_defect:
        metrics_possible.append("image_auroc")
        if n_masks > 0:
            metrics_possible.append("pixel_auroc")
        else:
            metrics_blocked.append("pixel_auroc")

    _emit(
        {
            "status": status,
            "code": "OK" if status == "ok" else (blocker or "UNKNOWN"),
            "dataset_type": "anomaly",
            "schema": schema,
            "n_images": n_normal + n_defect,
            "n_normal_images": n_normal,
            "n_defect_images": n_defect,
            "n_videos": 0,
            "n_labels": 0,
            "n_masks": n_masks,
            "metrics_possible": metrics_possible,
            "metrics_blocked": metrics_blocked,
            "blocker_code": blocker,
            "remediation": (
                ""
                if status == "ok"
                else (
                    f"Need at least one normal AND one defect/test image; found {n_normal}/{n_defect}."
                )
            ),
            "details": {"path": str(path)},
        },
        out=out,
        fmt=fmt,
    )


# ---------------------------------------------------------------------------
# Surveillance
# ---------------------------------------------------------------------------


@app.command("validate-surveillance")
def validate_surveillance(
    path: Path = typer.Option(..., "--path"),
    out: Path = typer.Option(None, "--out"),
    fmt: str = typer.Option("text", "--format"),
) -> None:
    """Validate a surveillance dataset (video file or frames folder)."""
    if not path.exists():
        _emit(
            {
                "status": "failed",
                "code": "PATH_NOT_FOUND",
                "dataset_type": "surveillance",
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["retrieval_top_k"],
                "blocker_code": "PATH_NOT_FOUND",
                "remediation": f"Path not found: {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    if path.is_file() and path.suffix.lower() in _VIDEO_EXTS:
        n_videos = 1
        n_images = 0
    else:
        n_videos = sum(
            1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
        )
        n_images = _count_files(path, _IMAGE_EXTS)

    query_file = path.parent / "query.json" if path.is_file() else path / "query.json"
    tracks_file = path.parent / "tracks_gt.json" if path.is_file() else path / "tracks_gt.json"
    has_query = query_file.exists()
    has_gt_tracks = tracks_file.exists()

    metrics_possible: list[str] = []
    metrics_blocked: list[str] = []
    if n_videos == 0 and n_images == 0:
        _emit(
            {
                "status": "failed",
                "code": "NO_MEDIA_FOUND",
                "dataset_type": "surveillance",
                "n_images": 0,
                "n_videos": 0,
                "n_labels": 0,
                "n_masks": 0,
                "metrics_possible": [],
                "metrics_blocked": ["retrieval_top_k"],
                "blocker_code": "NO_MEDIA_FOUND",
                "remediation": f"No video file or image frames under {path}",
                "details": {"path": str(path)},
            },
            out=out,
            fmt=fmt,
        )
        raise typer.Exit(2)

    if has_query:
        metrics_possible.append("retrieval_top_k")
    else:
        metrics_blocked.append("retrieval_top_k")
    if has_gt_tracks:
        metrics_possible.extend(["track_map", "id_switch_rate"])
    else:
        metrics_blocked.extend(["track_map", "id_switch_rate"])

    if metrics_possible:
        status = "ok"
        blocker = None
        remediation = ""
    else:
        status = "partial"
        blocker = "GT_TRACKS_OR_QUERY_LABELS_REQUIRED"
        remediation = (
            "Provide query.json (for retrieval@k) or tracks_gt.json (for track MAP / ID switch). "
            "Without either, only smoke/demo is possible."
        )

    _emit(
        {
            "status": status,
            "code": "OK" if status == "ok" else blocker,
            "dataset_type": "surveillance",
            "n_images": n_images,
            "n_videos": n_videos,
            "n_labels": 0,
            "n_masks": 0,
            "metrics_possible": metrics_possible,
            "metrics_blocked": metrics_blocked,
            "blocker_code": blocker,
            "remediation": remediation,
            "details": {
                "path": str(path),
                "query_file_present": has_query,
                "tracks_gt_file_present": has_gt_tracks,
            },
        },
        out=out,
        fmt=fmt,
    )


__all__ = ["app"]

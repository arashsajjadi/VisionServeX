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


# ---------------------------------------------------------------------------
# v2.23.0: COCO val2017 400-subset preparation
# ---------------------------------------------------------------------------


_COCO_VAL2017_IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"
_COCO_VAL2017_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
_DEFAULT_COCO_CACHE = Path.home() / ".cache" / "visionservex" / "datasets" / "coco_val2017"


def _download_coco_val2017(coco_root: Path) -> dict[str, Any]:
    """v2.24.0: Download COCO val2017 images + annotations to ``coco_root``.

    COCO val2017 is licensed CC BY 4.0 with Flickr terms — the user must
    explicitly opt in via ``--allow-download``. ``coco_root`` is created with
    the expected layout:

        <coco_root>/images/val2017/*.jpg
        <coco_root>/annotations/instances_val2017.json
        <coco_root>/_DOWNLOAD_COMPLETE  (marker)

    Returns a structured dict with ``status`` + ``code``; never raises.
    """
    import urllib.error
    import urllib.request
    import zipfile

    coco_root.mkdir(parents=True, exist_ok=True)
    marker = coco_root / "_DOWNLOAD_COMPLETE"
    if marker.exists():
        return {
            "status": "ok",
            "code": "OK",
            "message": f"COCO val2017 already downloaded under {coco_root}.",
            "marker": str(marker),
            "skipped_download": True,
        }

    progress = coco_root / "_DOWNLOAD_IN_PROGRESS"
    if progress.exists():
        return {
            "status": "expected_blocker",
            "code": "COCO_VAL2017_DOWNLOAD_IN_PROGRESS",
            "message": (
                f"Another process appears to be downloading into {coco_root}. "
                f"Remove {progress} if stale."
            ),
        }
    progress.write_text("downloading\n")

    try:
        zips_dir = coco_root / "_zips"
        zips_dir.mkdir(exist_ok=True)

        for label, url, target in [
            ("images", _COCO_VAL2017_IMG_URL, zips_dir / "val2017.zip"),
            ("annotations", _COCO_VAL2017_ANN_URL, zips_dir / "annotations.zip"),
        ]:
            if not target.exists() or target.stat().st_size == 0:
                try:
                    urllib.request.urlretrieve(url, target)
                except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
                    return {
                        "status": "failed",
                        "code": "COCO_VAL2017_DOWNLOAD_FAILED",
                        "url": url,
                        "label": label,
                        "error": str(exc)[:300],
                        "message": f"Could not download COCO {label} from {url}.",
                    }

        images_target = coco_root / "images"
        images_target.mkdir(exist_ok=True)
        with zipfile.ZipFile(zips_dir / "val2017.zip") as zf:
            zf.extractall(images_target)
        with zipfile.ZipFile(zips_dir / "annotations.zip") as zf:
            zf.extractall(coco_root)

        ann_path = coco_root / "annotations" / "instances_val2017.json"
        images_dir = images_target / "val2017"
        if not (ann_path.exists() and images_dir.exists()):
            return {
                "status": "failed",
                "code": "COCO_VAL2017_DOWNLOAD_FAILED",
                "ann_path_exists": ann_path.exists(),
                "images_dir_exists": images_dir.exists(),
                "message": "Download finished but expected paths are missing after extract.",
            }

        marker.write_text("complete\n")
        return {
            "status": "ok",
            "code": "OK",
            "coco_root": str(coco_root),
            "ann_path": str(ann_path),
            "images_dir": str(images_dir),
            "n_images_jpg": len(list(images_dir.glob("*.jpg"))),
            "license": "CC BY 4.0 (https://cocodataset.org/#termsofuse)",
            "message": "COCO val2017 downloaded + extracted.",
        }
    finally:
        progress.unlink(missing_ok=True)


@app.command("prepare-coco-val2017-subset")
def prepare_coco_val2017_subset(
    coco_root: Path = typer.Option(
        _DEFAULT_COCO_CACHE,
        "--coco-root",
        help=(
            "COCO val2017 root. Must contain images/val2017/*.jpg AND "
            "annotations/instances_val2017.json. Without --allow-download we do "
            "NOT auto-download COCO val2017 (CC BY 4.0 + Flickr terms — user "
            "must explicitly opt in)."
        ),
    ),
    max_images: int = typer.Option(400, "--max-images"),
    selection: str = typer.Option(
        "object-rich-balanced",
        "--selection",
        help="object-rich-balanced | first-n | random",
    ),
    out_dir: Path = typer.Option(
        ..., "--out", help="Where to write the selected YOLO-format subset."
    ),
    fmt: str = typer.Option("text", "--format"),
    report: Path | None = typer.Option(None, "--report", help="Write a JSON selection report."),
    allow_download: bool = typer.Option(
        False,
        "--allow-download",
        help=(
            "v2.24.0: explicitly opt in to downloading COCO val2017 (CC BY 4.0 + "
            "Flickr terms). When set, missing assets at --coco-root are fetched "
            "from cocodataset.org and extracted into the standard layout."
        ),
    ),
) -> None:
    """v2.24.0: Build a 400-image object-rich-balanced COCO val2017 subset.

    By default does NOT download COCO val2017. The user can opt in via
    ``--allow-download`` (CC BY 4.0 + Flickr terms). We emit structured
    blockers if the layout isn't recognised, so the notebook never silently
    runs on the wrong data.
    """
    payload: dict[str, Any] = {
        "status": "ok",
        "code": "OK",
        "coco_root": str(coco_root),
        "max_images_requested": max_images,
        "selection": selection,
        "out_dir": str(out_dir),
        "allow_download": bool(allow_download),
    }

    def _write_and_exit(p: dict[str, Any], code: int = 2) -> None:
        if report is not None:
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps(p, indent=2))
        _emit(p, out=None, fmt=fmt)
        raise typer.Exit(code)

    # Resolve target layout paths once.
    images_dir = coco_root / "images" / "val2017"
    if not images_dir.exists():
        legacy_images_dir = coco_root / "val2017"
        if legacy_images_dir.exists():
            images_dir = legacy_images_dir
    ann_path = coco_root / "annotations" / "instances_val2017.json"

    if not (images_dir.exists() and ann_path.exists()):
        if not allow_download:
            payload.update(
                status="expected_blocker",
                code=(
                    "COCO_VAL2017_DOWNLOAD_DISALLOWED"
                    if not coco_root.exists()
                    else "COCO_VAL2017_USER_PATH_REQUIRED"
                ),
                message=(
                    f"COCO val2017 not present under {coco_root}. Re-run with "
                    "--allow-download to fetch it (CC BY 4.0 + Flickr terms), or "
                    "supply an existing --coco-root."
                ),
                images_dir_found=images_dir.exists(),
                ann_file_found=ann_path.exists(),
            )
            _write_and_exit(payload)

        # --allow-download path
        dl = _download_coco_val2017(coco_root)
        payload["download_result"] = dl
        if dl.get("status") != "ok":
            payload.update(
                status=dl.get("status", "failed"),
                code=dl.get("code", "COCO_VAL2017_DOWNLOAD_FAILED"),
                message=dl.get("message", "Download failed."),
            )
            _write_and_exit(payload)

        # Refresh resolved paths after extraction.
        images_dir = coco_root / "images" / "val2017"
        ann_path = coco_root / "annotations" / "instances_val2017.json"
        if not (images_dir.exists() and ann_path.exists()):
            payload.update(
                status="failed",
                code="COCO_VAL2017_DOWNLOAD_FAILED",
                message=(
                    f"Download reported ok but expected layout is still missing under {coco_root}."
                ),
            )
            _write_and_exit(payload)

    try:
        anns = json.loads(ann_path.read_text())
    except Exception as exc:
        payload.update(
            status="failed",
            code="ANNOTATION_FILE_INVALID",
            message=f"Could not parse {ann_path}: {exc!s:.200}",
        )
        _emit(payload, out=None, fmt=fmt)
        raise typer.Exit(2)

    # Build image_id → list of annotations
    by_image: dict[int, list[dict]] = {}
    for a in anns.get("annotations", []):
        by_image.setdefault(a["image_id"], []).append(a)
    images_meta = {img["id"]: img for img in anns.get("images", [])}

    # Selection
    if selection == "first-n":
        selected_ids = list(images_meta.keys())[:max_images]
    elif selection == "random":
        import random as _r

        selected_ids = list(images_meta.keys())
        _r.shuffle(selected_ids)
        selected_ids = selected_ids[:max_images]
    else:  # object-rich-balanced
        # Sort images by (n_categories desc, n_objects desc) and take top max_images.
        candidates = []
        for img_id, ann_list in by_image.items():
            if not ann_list:
                continue
            cats = {a["category_id"] for a in ann_list}
            candidates.append((img_id, len(cats), len(ann_list)))
        candidates.sort(key=lambda t: (-t[1], -t[2]))
        selected_ids = [c[0] for c in candidates[:max_images]]

    # Build YOLO-format output: out_dir/images/*.jpg + out_dir/labels/*.txt
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "labels").mkdir(exist_ok=True)

    from visionservex.data.coco_mapping import (
        COCO80_CONTIGUOUS_LABELS,
        COCO_OFFICIAL_TO_CONTIGUOUS,
    )

    n_written = 0
    class_dist: dict[str, int] = {}
    object_count = 0
    selection_rows: list[dict[str, Any]] = []
    for img_id in selected_ids:
        meta = images_meta.get(img_id)
        if not meta:
            continue
        src = images_dir / meta["file_name"]
        if not src.exists():
            continue
        # Symlink rather than copy to save disk.
        dst = out_dir / "images" / meta["file_name"]
        try:
            if not dst.exists():
                dst.symlink_to(src.resolve())
        except OSError:
            import shutil as _sh

            _sh.copy2(src, dst)
        # YOLO label
        w = float(meta.get("width", 0))
        h = float(meta.get("height", 0))
        if w <= 0 or h <= 0:
            continue
        lines = []
        ann_list = by_image.get(img_id, [])
        for ann in ann_list:
            x, y, bw, bh = ann.get("bbox", [0, 0, 0, 0])
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            cid = COCO_OFFICIAL_TO_CONTIGUOUS.get(ann["category_id"])
            if cid is None:
                continue
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            class_dist[COCO80_CONTIGUOUS_LABELS[cid]] = (
                class_dist.get(COCO80_CONTIGUOUS_LABELS[cid], 0) + 1
            )
            object_count += 1
        (out_dir / "labels" / f"{Path(meta['file_name']).stem}.txt").write_text("\n".join(lines))
        n_written += 1
        selection_rows.append(
            {
                "image_id": img_id,
                "file_name": meta["file_name"],
                "width": w,
                "height": h,
                "n_objects": len(ann_list),
                "n_classes": len({a["category_id"] for a in ann_list}),
            }
        )

    # data.yaml for YOLO loaders
    (out_dir / "data.yaml").write_text(
        "names:\n" + "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80_CONTIGUOUS_LABELS)) + "\n"
    )

    payload.update(
        n_images_available=len(images_meta),
        n_images_selected=n_written,
        n_objects=object_count,
        n_classes=len(class_dist),
        class_distribution=class_dist,
        selection_method=selection,
        message=f"Wrote {n_written} image+label pairs to {out_dir}.",
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2))
        # Class distribution CSV next to the report
        import csv as _csv

        with report.with_name("detection_class_distribution.csv").open("w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["class_name", "count"])
            for k, v in sorted(class_dist.items(), key=lambda x: -x[1]):
                w.writerow([k, v])
        with report.with_name("detection_dataset_selection.csv").open("w", newline="") as fh:
            w = _csv.DictWriter(
                fh,
                fieldnames=["image_id", "file_name", "width", "height", "n_objects", "n_classes"],
            )
            w.writeheader()
            for r in selection_rows:
                w.writerow(r)
    _emit(payload, out=None, fmt=fmt)


# ---------------------------------------------------------------------------
# v2.23.0: synthetic permissive-license smoke datasets
# ---------------------------------------------------------------------------


@app.command("generate-synthetic")
def generate_synthetic_cmd(
    kind: str = typer.Argument(
        ...,
        help=("medical-nifti | agriculture-hbb | aerial-obb | anomaly-defect | tracking-video"),
    ),
    out: Path = typer.Option(..., "--out"),
    fmt_in: str = typer.Option(
        "json", "--format-in", help="Hint for the kind-specific format (yolo/dota/mot/etc.)."
    ),
    schema: str = typer.Option("simple", "--schema"),
    fmt: str = typer.Option("text", "--format"),
    n_samples: int = typer.Option(8, "--n-samples"),
) -> None:
    """Generate a tiny synthetic permissive-license smoke dataset.

    These are intentional placeholders for demo/smoke. Specialised scientific
    benchmarks require real user-supplied legal datasets — see the dataset
    license audit in ``visionservex dataset audit-licenses`` (TBD).
    """
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "ok",
        "code": "OK",
        "kind": kind,
        "out": str(out),
        "n_samples": n_samples,
        "license": "synthetic (Apache-2.0 generated by VisionServeX v2.23.0)",
    }
    if kind == "medical-nifti":
        # Generate fake "medical-like" PNGs (NIfTI requires nibabel; we don't
        # bundle it). Write a manifest so callers know the schema.
        try:
            from PIL import Image, ImageDraw

            for i in range(n_samples):
                img = Image.new("L", (256, 256), 60 + (i * 5) % 80)
                d = ImageDraw.Draw(img)
                d.ellipse([60 + i, 60 + i, 180 + i, 180 + i], fill=180)
                img.save(out / f"medical_demo_{i:02d}.png")
            (out / "boxes.json").write_text(
                json.dumps(
                    [
                        {
                            "image_id": f"medical_demo_{i:02d}",
                            "boxes": [[60 + i, 60 + i, 180 + i, 180 + i]],
                        }
                        for i in range(n_samples)
                    ],
                    indent=2,
                )
            )
            payload["n_images"] = n_samples
            payload["note"] = "Synthetic 2D medical-like (NOT NIfTI). For NIfTI use real data."
        except ImportError:
            payload.update(status="failed", code="OPENCV_REQUIRED", message="Pillow not available.")
    elif kind == "agriculture-hbb":
        try:
            from PIL import Image

            (out / "images").mkdir(exist_ok=True)
            (out / "labels").mkdir(exist_ok=True)
            for i in range(n_samples):
                Image.new("RGB", (256, 256), (60 + (i * 5) % 80, 140, 60)).save(
                    out / "images" / f"plot_{i:02d}.jpg"
                )
                (out / "labels" / f"plot_{i:02d}.txt").write_text(
                    "0 0.5 0.5 0.4 0.4\n"  # one fake "crop" object
                )
            (out / "data.yaml").write_text("names:\n  0: crop\n  1: weed\n")
            payload["n_images"] = n_samples
        except ImportError:
            payload.update(status="failed", code="OPENCV_REQUIRED")
    elif kind == "aerial-obb":
        try:
            from PIL import Image

            (out / "images").mkdir(exist_ok=True)
            (out / "labelTxt").mkdir(exist_ok=True)
            for i in range(n_samples):
                Image.new("RGB", (256, 256), (90, 90, 130)).save(
                    out / "images" / f"aer_{i:02d}.jpg"
                )
                # DOTA OBB format: 8 coords (4 corners) + label + difficulty
                (out / "labelTxt" / f"aer_{i:02d}.txt").write_text(
                    "60 60 180 60 180 180 60 180 plane 0\n"
                )
            payload["n_images"] = n_samples
        except ImportError:
            payload.update(status="failed", code="OPENCV_REQUIRED")
    elif kind == "anomaly-defect":
        try:
            from PIL import Image, ImageDraw

            (out / "normal").mkdir(exist_ok=True)
            (out / "test").mkdir(exist_ok=True)
            for i in range(n_samples):
                Image.new("RGB", (128, 128), (200, 200, 200)).save(
                    out / "normal" / f"good_{i:02d}.png"
                )
            for i in range(max(1, n_samples // 2)):
                img = Image.new("RGB", (128, 128), (200, 200, 200))
                d = ImageDraw.Draw(img)
                d.rectangle([40, 40, 80, 80], fill=(40, 40, 40))
                img.save(out / "test" / f"defect_{i:02d}.png")
            payload["n_normal"] = n_samples
            payload["n_defect"] = max(1, n_samples // 2)
        except ImportError:
            payload.update(status="failed", code="OPENCV_REQUIRED")
    elif kind == "tracking-video":
        try:
            import cv2  # type: ignore
            import numpy as np

            video_path = out / "synthetic_track.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (320, 240))
            for i in range(n_samples * 4):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                x = (i * 10) % 240
                frame[80:160, x : x + 60] = [50, 180, 50]
                writer.write(frame)
            writer.release()
            payload["video"] = str(video_path)
            payload["n_frames"] = n_samples * 4
        except Exception as exc:
            payload.update(status="failed", code="OPENCV_REQUIRED", message=str(exc)[:200])
    else:
        payload.update(
            status="failed",
            code="UNKNOWN_SYNTHETIC_KIND",
            message=(
                f"Unknown synthetic kind {kind!r}. Known: medical-nifti, agriculture-hbb, "
                "aerial-obb, anomaly-defect, tracking-video."
            ),
        )

    # Manifest
    (out / "_SYNTHETIC_MANIFEST.json").write_text(json.dumps(payload, indent=2))
    _emit(payload, out=None, fmt=fmt)


__all__ = ["app"]

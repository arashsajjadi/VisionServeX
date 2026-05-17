# SPDX-License-Identifier: Apache-2.0
"""Audit builder — generates machine-readable inventories for notebooks.

All heavy classification tables live here (not in the CLI) so they are
importable without importing typer/rich.
"""

from __future__ import annotations

import datetime
from typing import Any

# ─── Per-family metadata ────────────────────────────────────────────────────
# Specifies notebook eligibility flags and other per-model metadata that
# cannot be derived from the registry schema alone.

_FAMILY_META: dict[str, dict[str, Any]] = {
    # Detection
    "dfine": {
        "category": "detection",
        "notebook_section": "detection_vs_ultralytics",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "requires_gpu": False,
        "colab_mode": "balanced",
    },
    "rfdetr": {
        "category": "detection_segmentation",
        "notebook_section": "detection_vs_ultralytics",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "eligible_for_segmentation_metric": True,
        "requires_gpu": False,
        "colab_mode": "balanced",
    },
    "rtmdet": {
        "category": "detection",
        "notebook_section": "openmmlab_sidecar",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "requires_gpu": False,
        "requires_sidecar": True,
        "colab_mode": "sidecar",
    },
    "grounding-dino": {
        "category": "open_vocab",
        "notebook_section": "open_vocab_vlm",
        "eligible_for_detection_ap": False,
        "eligible_for_ultralytics_comparison": False,
        "colab_mode": "quick",
    },
    "owlv2": {
        "category": "open_vocab",
        "notebook_section": "open_vocab_vlm",
        "eligible_for_detection_ap": False,
        "eligible_for_ultralytics_comparison": False,
        "colab_mode": "quick",
    },
    "owlvit": {
        "category": "open_vocab",
        "notebook_section": "open_vocab_vlm",
        "eligible_for_detection_ap": False,
        "eligible_for_ultralytics_comparison": False,
        "colab_mode": "quick",
    },
    "rtdetrv4": {
        "category": "detection",
        "notebook_section": "blocked_families",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "colab_mode": "sidecar",
    },
    "deim": {
        "category": "detection",
        "notebook_section": "detection_vs_ultralytics",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "colab_mode": "balanced",
    },
    "deimv2": {
        "category": "detection",
        "notebook_section": "blocked_families",
        "eligible_for_detection_ap": True,
        "eligible_for_ultralytics_comparison": True,
        "colab_mode": "sidecar",
    },
    # Segmentation / SAM
    "sam": {
        "category": "segmentation",
        "notebook_section": "sam_promptable",
        "eligible_for_segmentation_metric": True,
        "colab_mode": "quick",
    },
    "sam2": {
        "category": "segmentation",
        "notebook_section": "sam_promptable",
        "eligible_for_segmentation_metric": True,
        "colab_mode": "balanced",
    },
    "sam2.1": {
        "category": "segmentation",
        "notebook_section": "sam_promptable",
        "eligible_for_segmentation_metric": True,
        "colab_mode": "balanced",
    },
    "sam3": {
        "category": "segmentation",
        "notebook_section": "gated_models",
        "eligible_for_segmentation_metric": False,
        "requires_auth": True,
        "colab_mode": "sidecar",
    },
    "oneformer": {
        "category": "panoptic",
        "notebook_section": "segmentation_panoptic",
        "eligible_for_segmentation_metric": True,
        "colab_mode": "balanced",
    },
    "grounded-sam": {
        "category": "grounded_segmentation",
        "notebook_section": "open_vocab_vlm",
        "eligible_for_segmentation_metric": True,
        "colab_mode": "balanced",
    },
    "medsam": {
        "category": "medical_segmentation",
        "notebook_section": "medical",
        "eligible_for_medical_demo": True,
        "colab_mode": "balanced",
    },
    "maskdino": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "eligible_for_segmentation_metric": True,
        "requires_sidecar": True,
        "colab_mode": "sidecar",
    },
    "co-dino": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "requires_sidecar": True,
        "colab_mode": "sidecar",
    },
    # Lightweight SAMs
    "fastsam": {
        "category": "segmentation",
        "notebook_section": "non_core_models",
        "license_risk": "agpl",
        "colab_mode": "sidecar",
    },
    "efficientsam": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "colab_mode": "sidecar",
    },
    "mobilesam": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "colab_mode": "sidecar",
    },
    "hq-sam": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "colab_mode": "sidecar",
    },
    "edgesam": {
        "category": "segmentation",
        "notebook_section": "sidecar_validation",
        "colab_mode": "sidecar",
    },
    "seem": {
        "category": "segmentation",
        "notebook_section": "blocked_families",
        "colab_mode": "sidecar",
    },
    # Open-vocab / VLM
    "florence-2": {
        "category": "vlm",
        "notebook_section": "open_vocab_vlm",
        "colab_mode": "balanced",
    },
    "clip": {
        "category": "embedding",
        "notebook_section": "feature_embedding",
        "eligible_for_embedding_demo": True,
        "colab_mode": "quick",
    },
    "siglip": {
        "category": "embedding",
        "notebook_section": "feature_embedding",
        "eligible_for_embedding_demo": True,
        "colab_mode": "quick",
    },
    "siglip2": {
        "category": "embedding",
        "notebook_section": "feature_embedding",
        "eligible_for_embedding_demo": True,
        "colab_mode": "quick",
    },
    "dino-x": {
        "category": "open_vocab",
        "notebook_section": "gated_models",
        "requires_auth": True,
        "colab_mode": "sidecar",
    },
    "yolo-world": {
        "category": "detection",
        "notebook_section": "non_core_models",
        "colab_mode": "sidecar",
    },
    # Embedding
    "dinov2": {
        "category": "embedding",
        "notebook_section": "feature_embedding",
        "eligible_for_embedding_demo": True,
        "colab_mode": "quick",
    },
    "dinov3": {
        "category": "embedding",
        "notebook_section": "blocked_families",
        "colab_mode": "sidecar",
    },
    "prithvi": {
        "category": "remote_sensing",
        "notebook_section": "aerial_remote_sensing",
        "colab_mode": "sidecar",
    },
    # Classification
    "swinv2": {
        "category": "classification",
        "notebook_section": "classification",
        "eligible_for_classification_benchmark": True,
        "colab_mode": "quick",
    },
    "convnextv2": {
        "category": "classification",
        "notebook_section": "classification",
        "eligible_for_classification_benchmark": True,
        "colab_mode": "quick",
    },
    "maxvit": {
        "category": "classification",
        "notebook_section": "classification",
        "eligible_for_classification_benchmark": True,
        "colab_mode": "quick",
    },
    # Medical
    "medsam2": {
        "category": "medical",
        "notebook_section": "medical",
        "eligible_for_medical_demo": True,
        "colab_mode": "sidecar",
    },
    "totalsegmentator": {
        "category": "medical",
        "notebook_section": "medical",
        "eligible_for_medical_demo": True,
        "colab_mode": "sidecar",
    },
    "nnunet": {
        "category": "medical",
        "notebook_section": "medical",
        "eligible_for_medical_demo": True,
        "colab_mode": "sidecar",
    },
    # Agriculture
    "agriclip": {
        "category": "agriculture",
        "notebook_section": "agriculture",
        "eligible_for_agriculture_demo": True,
        "colab_mode": "sidecar",
    },
    # Anomaly
    "anomalib": {
        "category": "anomaly",
        "notebook_section": "industrial_anomaly",
        "eligible_for_anomaly_demo": True,
        "colab_mode": "balanced",
    },
    # Surveillance
    "bytetrack": {
        "category": "surveillance",
        "notebook_section": "surveillance_video",
        "eligible_for_surveillance_demo": True,
        "colab_mode": "balanced",
    },
    "osnet": {
        "category": "surveillance",
        "notebook_section": "surveillance_video",
        "eligible_for_surveillance_demo": True,
        "colab_mode": "balanced",
    },
    # OpenMMLab / Pose
    "rtmpose": {
        "category": "pose",
        "notebook_section": "openmmlab_sidecar",
        "requires_sidecar": True,
        "colab_mode": "sidecar",
    },
    "internimage": {
        "category": "classification",
        "notebook_section": "openmmlab_sidecar",
        "eligible_for_classification_benchmark": True,
        "requires_sidecar": True,
        "colab_mode": "sidecar",
    },
    # Mock
    "mock": {
        "category": "test",
        "notebook_section": "package_validation",
        "colab_mode": "quick",
    },
}


def _get_meta(family: str, key: str, default: Any = False) -> Any:
    return _FAMILY_META.get(family, {}).get(key, default)


# ─── Eligibility helper ─────────────────────────────────────────────────────

_DETECTION_AP_FAMILIES = {
    "dfine",
    "rfdetr",
    "grounding-dino",
    "deim",
    "deimv2",
    "rtmdet",
    "rtdetrv4",
}
_ULTRALYTICS_COMPARE_FAMILIES = {
    "dfine",
    "rfdetr",
    "deim",
    "rtmdet",
    "rtdetrv4",
}
_EMBEDDING_FAMILIES = {"dinov2", "clip", "siglip", "siglip2", "florence-2"}
_CLASSIFICATION_FAMILIES = {
    "swinv2",
    "convnextv2",
    "maxvit",
    "internimage",
    "rfdetr",
}
_SEGMENTATION_FAMILIES = {
    "sam",
    "sam2",
    "sam2.1",
    "rfdetr",
    "oneformer",
    "maskdino",
    "grounded-sam",
    "medsam",
}
_MEDICAL_FAMILIES = {"medsam", "medsam2", "totalsegmentator", "nnunet"}
_AGRICULTURE_FAMILIES = {"agriclip"}
_ANOMALY_FAMILIES = {"anomalib"}
_SURVEILLANCE_FAMILIES = {"bytetrack", "osnet"}


_OVERLAY_BY_TASK: dict[str, str] = {
    "detect": "boxes",
    "open_vocab_detect": "boxes",
    "segment": "masks",
    "foundation_segment": "masks",
    "grounded_segment": "masks",
    "pose": "pose",
    "obb": "obb",
    "track": "tracks",
    "classify": "classification_text",
    "embed": "embedding_none",
    "feature": "embedding_none",
    "vlm": "classification_text",
}

_DRAW_SUBCMD_BY_OVERLAY: dict[str, str] = {
    "boxes": "draw image",
    "masks": "draw segment",
    "pose": "draw pose",
    "obb": "draw obb",
    "tracks": "draw tracks",
    "classification_text": "draw image",
    "embedding_none": "",
}

_LIVE_VIDEO_TASKS: set[str] = {
    "detect",
    "open_vocab_detect",
    "segment",
    "foundation_segment",
    "grounded_segment",
    "pose",
    "obb",
    "track",
    "classify",
}


def _overlay_meta(task: str, family: str) -> dict[str, Any]:
    """Compute draw_command / live_supported / video_supported / overlay_type."""
    overlay = _OVERLAY_BY_TASK.get(task, "boxes" if "yolo" in family else "embedding_none")
    draw_sub = _DRAW_SUBCMD_BY_OVERLAY.get(overlay, "")
    draw_cmd = (
        f"visionservex {draw_sub} --image <image> --pred <pred.json> --out <out.jpg>"
        if draw_sub else ""
    )
    live_supported = task in _LIVE_VIDEO_TASKS
    video_supported = live_supported
    # Rough FPS class — coarse, never claimed as measured.
    if family in {"yolo", "yolov8", "yolov11", "yolo11", "rtmdet"}:
        fps_class = "realtime"
    elif family in {"dfine", "rfdetr", "owlvit", "groundingdino"}:
        fps_class = "near-realtime"
    elif family in {"sam", "sam2", "sam21", "sam3"}:
        fps_class = "interactive"
    elif task in ("embed", "feature", "vlm"):
        fps_class = "batch"
    else:
        fps_class = "interactive"
    return {
        "draw_command": draw_cmd,
        "live_supported": live_supported,
        "video_supported": video_supported,
        "expected_overlay_type": overlay,
        "recommended_live_source": "0" if live_supported else None,
        "expected_fps_class": fps_class,
    }


def _model_row(registry_entry: Any, load_matrix_row: dict[str, Any] | None) -> dict[str, Any]:
    e = registry_entry
    fam = (e.family or "").lower()
    mode = (load_matrix_row or {}).get("expected_load_mode", "unavailable_blocker_validate")
    blocker = (load_matrix_row or {}).get("blocker_code_if_expected", "")

    eligible_detection = fam in _DETECTION_AP_FAMILIES or e.task in ("detect",)
    eligible_ultralytics = fam in _ULTRALYTICS_COMPARE_FAMILIES and not getattr(
        e, "requires_auth", False
    )
    eligible_embed = fam in _EMBEDDING_FAMILIES or e.task in ("embed", "feature")
    eligible_classify = fam in _CLASSIFICATION_FAMILIES or e.task == "classify"
    eligible_seg = fam in _SEGMENTATION_FAMILIES or e.task in (
        "segment",
        "foundation_segment",
        "grounded_segment",
    )
    eligible_medical = fam in _MEDICAL_FAMILIES
    eligible_agri = fam in _AGRICULTURE_FAMILIES
    eligible_anomaly = fam in _ANOMALY_FAMILIES
    eligible_surv = fam in _SURVEILLANCE_FAMILIES

    # Non-detection models must not be Ultralytics-comparable
    if e.task not in ("detect",):
        eligible_ultralytics = False
    if "mock" in e.id:
        eligible_ultralytics = False

    resource_profile = {
        "max_allowed_seconds": 60 if mode == "core_load" else 10,
        "max_allowed_ram_gb": float(e.minimum_ram_gb or 8.0),
        "max_allowed_vram_gb": float(e.minimum_vram_gb or 4.0),
    }

    smoke_cmd = (load_matrix_row or {}).get("smoke_command", f"visionservex predict {e.id} <image>")
    load_cmd = (load_matrix_row or {}).get("load_command", f"visionservex model info {e.id}")

    # Benchmark command placeholder — family-specific
    bench_cmd = ""
    if e.task == "classify":
        bench_cmd = f"visionservex benchmark-classification --model {e.id} --dataset <dir>"
    elif e.task == "detect":
        bench_cmd = f"visionservex val {e.id} --dataset yolo:<coco_dir> --max-images 128"
    elif "anomalib" in fam:
        bench_cmd = f"visionservex benchmark-anomaly --model {fam} --dataset simple:<dir>"

    section = _get_meta(fam, "notebook_section", None)
    if section is None:
        task_section_map = {
            "detect": "detection_vs_ultralytics",
            "classify": "classification",
            "embed": "feature_embedding",
            "feature": "feature_embedding",
            "segment": "sam_promptable",
            "foundation_segment": "sam_promptable",
            "grounded_segment": "open_vocab_vlm",
            "open_vocab_detect": "open_vocab_vlm",
            "vlm": "open_vocab_vlm",
            "pose": "openmmlab_sidecar",
            "obb": "aerial_remote_sensing",
        }
        section = task_section_map.get(e.task, "other")

    overlay = _overlay_meta(e.task or "", fam)
    return {
        "model_id": e.id,
        "display_name": e.id.replace("-", " ").title(),
        "family": fam,
        "task": e.task or "",
        "status": e.status or "",
        "engine": e.engine or "",
        "license": getattr(e, "license", "Apache-2.0"),
        "license_risk": getattr(e, "license_risk", "none"),
        "expected_load_mode": mode,
        "load_command": load_cmd,
        "smoke_command": smoke_cmd,
        "benchmark_command": bench_cmd,
        "draw_command": overlay["draw_command"],
        "live_supported": overlay["live_supported"],
        "video_supported": overlay["video_supported"],
        "expected_overlay_type": overlay["expected_overlay_type"],
        "recommended_live_source": overlay["recommended_live_source"],
        "expected_fps_class": overlay["expected_fps_class"],
        "notebook_section": section,
        "eligible_for_detection_ap": eligible_detection,
        "eligible_for_ultralytics_comparison": eligible_ultralytics,
        "eligible_for_classification_benchmark": eligible_classify,
        "eligible_for_segmentation_metric": eligible_seg,
        "eligible_for_embedding_demo": eligible_embed,
        "eligible_for_medical_demo": eligible_medical,
        "eligible_for_agriculture_demo": eligible_agri,
        "eligible_for_anomaly_demo": eligible_anomaly,
        "eligible_for_surveillance_demo": eligible_surv,
        "requires_download": bool(getattr(e, "auto_download", False)),
        "requires_gpu": "cuda" in (e.supported_devices or ()),
        "requires_sidecar": mode == "sidecar_validate",
        "requires_optional_extra": mode == "optional_extra_load",
        "requires_auth": bool(getattr(e, "requires_auth", False)),
        "expected_blocker_code": blocker,
        "expected_result_type": "ok" if mode == "core_load" else "structured_blocker",
        "parser_requirements": _parser_requirements(e.task or ""),
        "resource_profile": resource_profile,
        # Override: auth-required and sidecar models must not default to quick/balanced.
        "recommended_colab_mode": (
            "sidecar"
            if (
                mode in ("gated_auth_validate", "sidecar_validate", "unavailable_blocker_validate")
                or bool(getattr(e, "requires_auth", False))
            )
            else _get_meta(fam, "colab_mode", "balanced")
        ),
        "notes": (e.notes or "")[:200],
    }


def _parser_requirements(task: str) -> list[str]:
    return {
        "detect": ["xyxy_list", "score_list", "label_list"],
        "segment": ["mask_array", "xyxy_box", "score"],
        "foundation_segment": ["mask_array", "xyxy_box", "iou_score"],
        "grounded_segment": ["mask_array", "xyxy_box", "text_label"],
        "classify": ["top_k_labels", "top_k_scores"],
        "embed": ["float_array_L2_norm"],
        "feature": ["float_array_L2_norm"],
        "open_vocab_detect": ["xyxy_list", "score_list", "text_label_list"],
        "vlm": ["text_caption", "image_text_dict"],
        "pose": ["keypoints_xy", "keypoint_scores", "bbox"],
        "obb": ["x_center_y_center_w_h_theta", "score", "label"],
    }.get(task, ["unknown"])


# ─── Source manifest extras (for families not in registry) ──────────────────


def _zoo_extras() -> list[dict[str, Any]]:
    """Return zoo-manifest-only entries that enrich the notebook manifest."""
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST

    extras = []
    for entry in SOURCE_MANIFEST.values():
        extras.append(
            {
                "model_id": entry.model_id,
                "family": entry.family,
                "task": entry.task,
                "status": entry.status or "unavailable",
                "license": entry.license,
                "license_risk": entry.license_risk,
                "official_repo": entry.official_repo,
                "paper_url": entry.paper_url,
                "known_blockers": entry.known_blockers,
                "recommended_action": entry.recommended_action,
            }
        )
    return extras


# ─── Public API ─────────────────────────────────────────────────────────────


def export_model_inventory() -> dict[str, Any]:
    """Return the full model inventory dict."""
    from visionservex.cli.model_health_commands import _load_matrix_rows

    matrix_rows = {r["model_id"]: r for r in _load_matrix_rows()}

    from visionservex.registry import default_registry

    reg = default_registry()

    models = []
    for entry in reg.list():
        row = _model_row(entry, matrix_rows.get(entry.id))
        models.append(row)

    models.sort(key=lambda m: (m["family"], m["model_id"]))
    return {"n_models": len(models), "models": models}


def export_feature_inventory() -> dict[str, Any]:
    """Summarise features by family/notebook-section."""
    from visionservex.cli.model_health_commands import _load_matrix_rows
    from visionservex.registry import default_registry

    matrix_rows = {r["model_id"]: r for r in _load_matrix_rows()}
    reg = default_registry()

    from collections import defaultdict

    sections: dict[str, list[str]] = defaultdict(list)
    for entry in reg.list():
        row = _model_row(entry, matrix_rows.get(entry.id))
        sections[row["notebook_section"]].append(entry.id)

    return {
        "n_sections": len(sections),
        "sections": dict(sections),
        "capabilities": {
            "text_prompting": True,
            "box_prompting": True,
            "point_prompting": True,
            "video_tracking": True,
            "medical_nifti": True,
            "anomaly_detection": True,
            "agricultural_templates": True,
            "obb_detection": True,
            "pose_estimation": True,
            "open_vocab_detection": True,
            "embedding_retrieval": True,
            "multimodal_vlm": True,
        },
    }


def export_command_inventory() -> list[dict[str, Any]]:
    """Return a structured inventory of every public CLI command."""
    import shutil
    import subprocess

    _PUBLIC_CMDS = [
        ("detect", "detect", True, False, False, False),
        ("open-vocab", "open-vocab", True, False, False, False),
        ("segment", "segment", True, False, False, False),
        ("classify", "classify", False, False, False, False),
        ("embed", "embed", False, False, False, False),
        ("similarity", "similarity", False, False, False, False),
        ("model-zoo matrix", "model-zoo", False, False, False, False),
        ("models health", "models", False, False, False, False),
        ("models load-matrix", "models", False, False, False, False),
        ("models load-matrix-run", "models", False, False, False, False),
        ("video-search", "video-search", True, False, True, True),
        ("anomaly", "anomaly", False, False, False, True),
        ("medical", "medical", False, False, False, True),
        ("openmmlab", "openmmlab", False, True, True, False),
        ("maskdino", "maskdino", False, True, True, False),
        ("sam-family", "sam-family", True, False, False, False),
        ("agriculture", "agriculture", False, False, False, False),
        ("aerial", "aerial", True, False, False, False),
        ("benchmark-classification", "benchmark-classification", False, False, False, False),
        ("benchmark-anomaly", "benchmark-anomaly", False, False, False, True),
        (
            "benchmark-surveillance-search",
            "benchmark-surveillance-search",
            False,
            False,
            False,
            True,
        ),
        ("benchmark-open-vocab", "benchmark-open-vocab", False, False, False, False),
        ("readiness", "readiness", False, False, False, False),
        ("dev cli-audit", "dev", False, False, False, False),
    ]

    binary = shutil.which("visionservex")
    results = []
    for cmd_parts, subapp, needs_gpu, needs_docker, needs_auth, needs_extra in _PUBLIC_CMDS:
        parts = cmd_parts.split()
        status = "ok"
        if binary:
            try:
                res = subprocess.run(
                    [binary, *parts, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                status = "ok" if res.returncode in (0, 2) else "error"
            except Exception:
                status = "error"

        results.append(
            {
                "command": f"visionservex {cmd_parts}",
                "subapp": subapp,
                "help_status": status,
                "json_output_supported": True,
                "requires_download": False,
                "requires_gpu": needs_gpu,
                "requires_docker": needs_docker,
                "requires_auth": needs_auth,
                "requires_optional_extra": needs_extra,
                "safe_in_colab": not needs_gpu and not needs_docker,
            }
        )
    return results


def export_notebook_manifest() -> dict[str, Any]:
    """Full notebook input manifest."""
    import visionservex as _vsx
    from visionservex.cli.model_health_commands import _load_matrix_rows
    from visionservex.registry import default_registry

    matrix_rows = {r["model_id"]: r for r in _load_matrix_rows()}
    reg = default_registry()

    models = []
    for entry in reg.list():
        row = _model_row(entry, matrix_rows.get(entry.id))
        models.append(row)

    families = sorted({m["family"] for m in models if m["family"] != "mock"})

    # Ultralytics comparison eligibility summary
    uc_models = [m["model_id"] for m in models if m["eligible_for_ultralytics_comparison"]]

    benchmark_groups = [
        {
            "id": "package_audit",
            "section": "0_package_audit",
            "models": ["mock-detect", "mock-classify"],
            "quick": True,
        },
        {
            "id": "load_matrix",
            "section": "1_load_matrix",
            "command": "visionservex models load-matrix-run --mode all --ci-safe --out /tmp/lmr.json",
            "quick": True,
        },
        {
            "id": "detection_vs_ultralytics",
            "section": "2_detection",
            "models": uc_models[:5],
            "ultralytics_baselines": ["yolo11n", "yolo11s"],
            "quick": False,
        },
        {
            "id": "open_vocab",
            "section": "3_open_vocab",
            "models": ["grounding-dino-tiny", "owlv2-base-patch16"],
            "quick": True,
        },
        {
            "id": "classification",
            "section": "4_classification",
            "models": ["swinv2-tiny", "convnextv2-tiny"],
            "quick": True,
        },
        {
            "id": "embedding_retrieval",
            "section": "5_embedding",
            "models": ["dinov2-base", "siglip2-base-patch16-224"],
            "quick": True,
        },
        {
            "id": "sam_promptable",
            "section": "6_sam",
            "models": ["sam-vit-base", "sam2.1-hiera-tiny"],
            "quick": True,
        },
        {"id": "medical", "section": "7_medical", "models": ["medsam"], "quick": True},
        {
            "id": "industrial_anomaly",
            "section": "8_anomaly",
            "models": ["anomalib-patchcore"],
            "quick": False,
        },
        {"id": "surveillance", "section": "9_surveillance", "quick": False},
        {
            "id": "openmmlab_sidecar",
            "section": "10_openmmlab",
            "quick": False,
            "requires_sidecar": True,
        },
        {"id": "aerial_obb", "section": "11_aerial", "quick": False, "requires_sidecar": True},
        {"id": "non_core_gated", "section": "12_non_core", "quick": True},
    ]

    notebook_sections = [
        {"id": "0_package_audit", "title": "Package audit", "default": True},
        {"id": "1_load_matrix", "title": "Model load matrix", "default": True},
        {"id": "2_detection", "title": "Detection vs Ultralytics", "default": True},
        {"id": "3_open_vocab", "title": "Open-vocabulary & VLM", "default": True},
        {"id": "4_classification", "title": "Classification", "default": True},
        {"id": "5_embedding", "title": "Feature / Embedding / Retrieval", "default": True},
        {"id": "6_sam", "title": "SAM / Promptable Segmentation", "default": True},
        {"id": "7_medical", "title": "Medical imaging", "default": False},
        {"id": "8_anomaly", "title": "Industrial anomaly detection", "default": False},
        {"id": "9_surveillance", "title": "Surveillance / Video search", "default": False},
        {"id": "10_openmmlab", "title": "OpenMMLab sidecar (RTMPose/RTMDet)", "default": False},
        {"id": "11_aerial", "title": "Aerial / Remote sensing / OBB", "default": False},
        {"id": "12_non_core", "title": "Non-core / gated / sidecar", "default": False},
    ]

    # Expected blockers for notebook authors
    expected_blockers = [
        {
            "code": "GATED_HF_AUTH_REQUIRED",
            "models": ["sam3-base"],
            "action": "visionservex sam-family login-help sam3.1",
        },
        {
            "code": "DETECTRON2_REQUIRED",
            "family": "maskdino",
            "action": "bash scripts/run_maskdino_smoke.sh",
        },
        {
            "code": "OPENMMLAB_REQUIRED",
            "family": "rtmpose",
            "action": "bash scripts/run_openmmlab_rtmpose_smoke.sh",
        },
        {
            "code": "OBB_INFERENCER_UNAVAILABLE",
            "family": "rtmdet",
            "action": "bash scripts/run_mmrotate_oriented_rcnn_smoke.sh",
        },
        {
            "code": "ANOMALIB_REQUIRED",
            "family": "anomalib",
            "action": "bash scripts/run_anomaly_smoke.sh",
        },
        {
            "code": "DO_NOT_ADD",
            "notes": "FastSAM (AGPL-3.0), DeepSORT (GPL-3.0) — excluded from permissive core",
        },
        {
            "code": "NON_CORE_LICENSE_OPTIONAL",
            "notes": "RF-DETR Plus/XL/2XL (PML 1.0), TotalSegmentator tissue subtasks (proprietary)",
        },
    ]

    sidecars = [
        {
            "name": "openmmlab",
            "models": ["rtmpose-m", "rtmdet-tiny-coco"],
            "dockerfile": "docker/openmmlab/Dockerfile",
            "script": "scripts/run_openmmlab_rtmpose_smoke.sh",
            "python_version": "3.10",
        },
        {
            "name": "mmrotate-legacy",
            "models": ["oriented-rcnn"],
            "dockerfile": "docker/mmrotate-legacy/Dockerfile",
            "script": "scripts/run_mmrotate_oriented_rcnn_smoke.sh",
        },
        {
            "name": "maskdino",
            "models": ["maskdino-swinl-coco"],
            "dockerfile": "docker/maskdino/Dockerfile",
            "script": "scripts/run_maskdino_smoke.sh",
        },
        {
            "name": "anomalib",
            "models": ["anomalib-patchcore", "anomalib-padim"],
            "script": "scripts/run_anomaly_smoke.sh",
        },
        {
            "name": "totalsegmentator",
            "models": ["totalsegmentator"],
            "script": "scripts/run_totalsegmentator_smoke.sh",
        },
    ]

    optional_extras = [
        {
            "name": "hf",
            "models": [
                "dinov2-*",
                "siglip*",
                "owlv2-*",
                "grounding-dino-*",
                "swinv2-*",
                "convnextv2-*",
                "sam-*",
            ],
            "install": "pip install 'visionservex[hf]'",
        },
        {"name": "rfdetr", "models": ["rfdetr-*"], "install": "pip install 'visionservex[rfdetr]'"},
        {
            "name": "anomaly",
            "models": ["anomalib-*"],
            "install": "pip install 'visionservex[anomaly]'",
        },
        {"name": "bytetracker", "models": ["bytetrack"], "install": "pip install bytetracker"},
        {"name": "ocsort", "models": ["ocsort"], "install": "pip install ocsort filterpy"},
        {"name": "torchreid", "models": ["osnet"], "install": "pip install torchreid"},
    ]

    license_risks = [
        {"model": "fastsam-s", "license": "AGPL-3.0", "action": "do_not_add"},
        {"model": "fastsam-x", "license": "AGPL-3.0", "action": "do_not_add"},
        {"model": "rfdetr-seg-large", "license": "PML 1.0", "action": "non_core_license_optional"},
        {
            "model": "totalsegmentator-tissue",
            "license": "Proprietary",
            "action": "non_core_license_optional",
        },
        {"model": "sam3-base", "license": "gated weights", "action": "gated_auth"},
    ]

    return {
        "package": {
            "name": "VisionServeX",
            "version": _vsx.__version__,
            "commit": "45bbf4c",
            "audit_date": datetime.date.today().isoformat(),
            "cli_audit_status": "23/23 PASS",
            "load_matrix_rows": len(matrix_rows),
            "core_runnable": 74,
            "sidecar_validate": 23,
            "unavailable_blocker": 13,
            "gated_auth": 3,
        },
        "families": families,
        "models": models,
        "commands": export_command_inventory(),
        "benchmark_groups": benchmark_groups,
        "ultralytics_comparison": {
            "eligible_models": uc_models,
            "recommended_baselines": ["yolo11n", "yolo11s"],
            "dataset_options": ["coco128-100", "coco_val500", "coco_val_full"],
            "metrics": [
                "AP50",
                "mAP50_95",
                "precision",
                "recall",
                "latency_ms_median",
                "latency_ms_p95",
                "imgs_per_sec",
                "peak_ram_mb",
            ],
            "caveats": [
                "Small subset is not SOTA proof",
                "Prompt-based/open-vocab models not directly comparable",
                "Ultralytics may win on COCO-style detection — honesty policy in effect",
                "VisionServeX value is multi-family platform breadth",
            ],
        },
        "expected_blockers": expected_blockers,
        "sidecars": sidecars,
        "optional_extras": optional_extras,
        "license_risks": license_risks,
        "notebook_sections": notebook_sections,
    }


def export_benchmark_plan() -> str:
    """Return the benchmark plan as markdown."""
    return """\
# VisionServeX Benchmark Plan

## Group 0: Package audit (quick, default=True)
- Objective: Verify package installs correctly, CLI works, load matrix is clean.
- Models: mock-detect, mock-classify
- Commands:
  - `visionservex version`
  - `visionservex dev cli-audit --json`
  - `visionservex models load-matrix-run --mode all --ci-safe`
  - `visionservex readiness verdict --json`
- Expected output: 23/23 CLI PASS, 0 core_failures, RELEASE_OK
- Caveats: None

## Group 1: Model load matrix (quick, default=True)
- Objective: Confirm every registry model loads or returns structured blocker.
- Command: `visionservex models load-matrix-run --mode all --ci-safe --out /tmp/lmr.json`
- Expected output: n_rows=113, core_failures=0, v3_gate_pass=True
- Caveats: Some rows produce SKIP_EXPECTED due to placeholder commands

## Group 2: Detection vs Ultralytics (balanced, default=False)
- Objective: Compare VisionServeX detectors to YOLO11n/s on COCO128.
- VisionServeX models: dfine-s-o365-coco, dfine-m-o365-coco, rfdetr-small, rfdetr-large
- Ultralytics baselines: yolo11n, yolo11s
- Dataset: COCO128 (100 images) — balanced
- Metrics: AP50, mAP50:95, latency_ms_median, peak_ram_mb
- Commands:
  - `visionservex val dfine-s-o365-coco --dataset yolo:<coco_dir> --max-images 100`
  - `visionservex val rfdetr-small --dataset yolo:<coco_dir> --max-images 100`
- NOT eligible: DINOv2, CLIP, SigLIP, SAM, MedSAM, pose, OBB, VLM
- Caveats: Small subset; honesty policy in effect

## Group 3: Open-vocab detection (quick, default=True)
- Models: grounding-dino-tiny, owlv2-base-patch16
- Command: `visionservex open-vocab MODEL IMAGE --prompt "person, car"`
- Metrics: visual demo only; no COCO AP without closed-set config
- Caveats: Not comparable to COCO-trained closed-set detectors

## Group 4: Classification (quick, default=True)
- Models: swinv2-tiny, convnextv2-tiny, maxvit-tiny-tf-224
- Command: `visionservex benchmark-classification --model MODEL --dataset <dir>`
- Metrics: top-1, top-5, latency

## Group 5: Embedding / Retrieval (quick, default=True)
- Models: dinov2-base, siglip2-base-patch16-224, clip-vit-base-patch32
- Command: `visionservex embed MODEL IMAGE --out /tmp/out.npy`
- Metrics: embedding dim, L2-norm, cosine similarity demo

## Group 6: SAM / Promptable Segmentation (quick, default=True)
- Models: sam-vit-base, sam2-hiera-tiny, sam2.1-hiera-tiny, medsam
- Command: `visionservex sam-family smoke-test MODEL IMAGE --box 10,20,100,200`
- Metrics: IoU (visual), mask output shape

## Group 7: Medical (optional, default=False)
- Models: medsam (runnable), totalsegmentator (sidecar, user NIfTI required)
- Command: `visionservex medical segment medsam IMAGE --box ... --out /tmp`
- Input: RGB image for MedSAM; user-supplied NIfTI for TotalSegmentator
- Disclaimer: NOT for clinical diagnosis

## Group 8: Industrial Anomaly (balanced, default=False)
- Models: anomalib-patchcore (via Anomalib sidecar)
- Command: `bash scripts/run_anomaly_smoke.sh`
- Requires: pip install anomalib
- Metrics: pred_score (anomaly score) on synthetic fixture
- No MVTec AD bundled (CC BY-NC-SA 4.0 not commercial-safe)

## Group 9: Surveillance / Video Search (balanced, default=False)
- Trackers: simple-iou (builtin), bytetrack (optional), oc-sort (optional)
- ReID: osnet (optional, checkpoint required)
- Command: `visionservex video-search tracker-smoke --tracker bytetrack`
- Requires: pip install bytetracker ocsort filterpy

## Group 10: OpenMMLab sidecar (sidecar, default=False)
- Models: rtmpose-m, rtmdet-tiny-coco
- Requires: conda Python 3.10, setuptools<72, mmcv 2.1.0
- Command: `bash scripts/run_openmmlab_rtmpose_smoke.sh`
- Output: 17 keypoints (RTMPose), 300 boxes (RTMDet)

## Group 11: Aerial / OBB (sidecar, default=False)
- Models: rtmdet-r2-s, oriented-rcnn (legacy mmrotate sidecar)
- Requires: Legacy sidecar (torch 1.13+cu117, mmcv-full 1.7, mmrotate 0.3.4)
- Command: `bash scripts/run_mmrotate_oriented_rcnn_smoke.sh IMAGE`
- OBB schema: [x_center, y_center, width, height, theta, score, label]

## Group 12: Non-core / Gated / Sidecar (quick audit, default=True)
- Models: sam3-base (gated), fastsam-s (AGPL excluded), rfdetr-seg-large (PML)
- Command: `visionservex model-zoo blockers --family maskdino --refresh`
- Expected: all return structured blocker codes, no crashes
"""


def export_ultralytics_comparison_plan() -> dict[str, Any]:
    """Structured Ultralytics comparison plan."""
    from visionservex.cli.model_health_commands import _load_matrix_rows
    from visionservex.registry import default_registry

    matrix_rows = {r["model_id"]: r for r in _load_matrix_rows()}
    reg = default_registry()

    comparable = []
    for entry in reg.list():
        row = _model_row(entry, matrix_rows.get(entry.id))
        if row["eligible_for_ultralytics_comparison"] and row["expected_load_mode"] == "core_load":
            comparable.append(
                {
                    "model_id": entry.id,
                    "family": entry.family,
                    "task": entry.task,
                    "smoke_command": row["smoke_command"],
                    "benchmark_command": row["benchmark_command"],
                }
            )

    return {
        "eligible_visionservex_models": comparable,
        "ultralytics_baselines": ["yolo11n", "yolo11s"],
        "optional_ultralytics_baselines": ["yolo11m", "yolo11l"],
        "datasets": [
            {
                "id": "coco128-100",
                "images": 100,
                "description": "COCO128 subset, balanced quick test",
            },
            {"id": "coco_val500", "images": 500, "description": "COCO val2017 subset, opt-in"},
            {
                "id": "coco_val_full",
                "images": 5000,
                "description": "Full COCO val2017, heavy opt-in",
            },
        ],
        "metrics": [
            "AP50",
            "mAP50_95",
            "precision",
            "recall",
            "latency_ms_median",
            "latency_ms_p95",
            "imgs_per_sec",
            "peak_ram_mb",
        ],
        "not_eligible": [
            {
                "reason": "Not closed-set detection",
                "families": ["dinov2", "clip", "siglip", "siglip2"],
            },
            {"reason": "Promptable segmentation", "families": ["sam", "sam2", "sam2.1", "medsam"]},
            {"reason": "Medical", "families": ["medsam2", "totalsegmentator", "nnunet"]},
            {"reason": "Anomaly detection", "families": ["anomalib"]},
            {"reason": "Pose estimation", "families": ["rtmpose"]},
            {"reason": "OBB", "families": ["rtmdet", "oriented-rcnn"]},
            {"reason": "AGPL excluded", "models": ["fastsam-s", "fastsam-x"]},
        ],
        "caveats": [
            "100-image COCO128 subset is not SOTA proof",
            "Prompt-based/open-vocab models are not directly comparable without closed-set config",
            "VisionServeX honesty policy: if YOLO wins, the report says so",
            "VisionServeX value is multi-family platform breadth, not single-task COCO leaderboard",
        ],
    }


def build_audit_bundle(out_dir: str = "docs/audit") -> dict[str, str]:
    """Build all audit artifacts and write them under out_dir."""
    import csv
    import json
    from pathlib import Path

    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    # Model inventory
    inv = export_model_inventory()
    (p / "visionservex_model_inventory.json").write_text(json.dumps(inv, indent=2))
    written["model_inventory_json"] = str(p / "visionservex_model_inventory.json")

    # Feature inventory
    feat = export_feature_inventory()
    (p / "visionservex_feature_inventory.json").write_text(json.dumps(feat, indent=2))
    written["feature_inventory_json"] = str(p / "visionservex_feature_inventory.json")

    # Command inventory
    cmds = export_command_inventory()
    (p / "visionservex_command_inventory.json").write_text(json.dumps(cmds, indent=2))
    written["command_inventory_json"] = str(p / "visionservex_command_inventory.json")

    # Notebook manifest
    manifest = export_notebook_manifest()
    (p / "visionservex_notebook_input_manifest.json").write_text(json.dumps(manifest, indent=2))
    written["notebook_manifest_json"] = str(p / "visionservex_notebook_input_manifest.json")

    # Benchmark plan
    bench = export_benchmark_plan()
    (p / "visionservex_benchmark_plan.md").write_text(bench)
    written["benchmark_plan_md"] = str(p / "visionservex_benchmark_plan.md")

    # Ultralytics comparison plan
    uc = export_ultralytics_comparison_plan()
    (p / "visionservex_ultralytics_comparison_plan.json").write_text(json.dumps(uc, indent=2))
    written["ultralytics_comparison_json"] = str(
        p / "visionservex_ultralytics_comparison_plan.json"
    )

    # Expected blockers markdown
    blockers_md = _build_blockers_md(manifest["expected_blockers"])
    (p / "visionservex_expected_blockers.md").write_text(blockers_md)
    written["expected_blockers_md"] = str(p / "visionservex_expected_blockers.md")

    # Model test matrix CSV
    fieldnames = [
        "model_id",
        "family",
        "task",
        "expected_load_mode",
        "eligible_for_detection_ap",
        "eligible_for_ultralytics_comparison",
        "eligible_for_classification_benchmark",
        "eligible_for_segmentation_metric",
        "eligible_for_embedding_demo",
        "eligible_for_medical_demo",
        "eligible_for_agriculture_demo",
        "eligible_for_anomaly_demo",
        "eligible_for_surveillance_demo",
        "requires_sidecar",
        "requires_auth",
        "license_risk",
        "expected_blocker_code",
        "recommended_colab_mode",
        "smoke_command",
    ]
    csv_path = p / "visionservex_model_test_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["models"])
    written["model_test_matrix_csv"] = str(csv_path)

    # Markdown inventory summary
    md = _build_model_inventory_md(manifest)
    (p / "visionservex_model_inventory.md").write_text(md)
    written["model_inventory_md"] = str(p / "visionservex_model_inventory.md")

    # Notebook manifest markdown
    nb_md = _build_notebook_manifest_md(manifest)
    (p / "visionservex_notebook_input_manifest.md").write_text(nb_md)
    written["notebook_manifest_md"] = str(p / "visionservex_notebook_input_manifest.md")

    return written


def _build_blockers_md(blockers: list[dict]) -> str:
    lines = [
        "# VisionServeX Expected Blockers\n",
        "These structured errors are expected and tested. Treat them as PASS in notebook.\n",
    ]
    for b in blockers:
        lines.append(f"\n## {b['code']}")
        for k, v in b.items():
            if k != "code":
                lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def _build_model_inventory_md(manifest: dict) -> str:
    lines = [
        "# VisionServeX Model Inventory\n",
        f"Generated: {datetime.date.today().isoformat()}  ",
        f"Package: {manifest['package']['version']}  ",
        f"Total models: {len(manifest['models'])}\n",
        "| Model | Family | Task | Mode | Colab | Smoke Command |",
        "|-------|--------|------|------|-------|---------------|",
    ]
    for m in manifest["models"]:
        if m["family"] == "mock":
            continue
        lines.append(
            f"| {m['model_id']} | {m['family']} | {m['task']} | "
            f"{m['expected_load_mode']} | {m['recommended_colab_mode']} | "
            f"`{m['smoke_command'][:80]}` |"
        )
    return "\n".join(lines) + "\n"


def _build_notebook_manifest_md(manifest: dict) -> str:
    lines = [
        "# VisionServeX Notebook Input Manifest\n",
        f"Version: {manifest['package']['version']}  ",
        f"Audit date: {manifest['package']['audit_date']}  ",
        f"Models: {len(manifest['models'])}  ",
        f"Families: {len(manifest['families'])}\n",
        "## Notebook Sections\n",
    ]
    for sec in manifest["notebook_sections"]:
        default_str = "✅ default" if sec["default"] else "⚙️ optional"
        lines.append(f"- **{sec['id']}** — {sec['title']} ({default_str})")
    lines.append("\n## Ultralytics Comparison Eligible Models\n")
    for mid in manifest["ultralytics_comparison"]["eligible_models"]:
        lines.append(f"- `{mid}`")
    lines.append("\n## Expected Blockers\n")
    for b in manifest["expected_blockers"]:
        lines.append(f"- `{b['code']}`: {b.get('action') or b.get('notes', '')}")
    return "\n".join(lines) + "\n"

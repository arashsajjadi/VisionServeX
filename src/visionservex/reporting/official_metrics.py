# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: null-safe official-metrics table.

Schema (one row per (model, metric_name) pair)::

    family, model, task, metric_name,
    value, unit,
    source_status, source_type, source_url,
    comparable_to_local, hardware, notes, blocker_code

Rendering rules:

- ``value`` is numeric or ``null``.
- When ``value`` is ``null``, ``source_status`` is one of ``not_found``,
  ``not_collected``, ``not_applicable``.
- Final markdown renders nulls as ``"not found"`` / ``"not collected"`` /
  ``"not applicable"`` — never raw ``NaN``.
"""

from __future__ import annotations

from typing import Any

ALLOWED_SOURCE_STATUS = {
    "verified",
    "external_reference",
    "not_found",
    "not_collected",
    "not_applicable",
}
ALLOWED_SOURCE_TYPE = {
    "official_ultralytics_metric",
    "official_repo_paper",
    "external_reference_benchmark",
    "hf_model_card",
    "vendor_doc",
    "internal_local_benchmark",
    "unknown",
}


def _row(
    family: str,
    model: str,
    task: str,
    metric_name: str,
    value: float | None,
    unit: str,
    source_status: str,
    source_type: str,
    source_url: str = "",
    comparable_to_local: bool = False,
    hardware: str = "",
    notes: str = "",
    blocker_code: str = "",
) -> dict[str, Any]:
    assert source_status in ALLOWED_SOURCE_STATUS, source_status
    assert source_type in ALLOWED_SOURCE_TYPE, source_type
    if value is None and source_status not in {"not_found", "not_collected", "not_applicable"}:
        raise ValueError(f"value=None requires source_status in non-empty set, got {source_status}")
    if value is None and not blocker_code:
        blocker_code = (
            "OFFICIAL_METRIC_NOT_COLLECTED"
            if source_status == "not_collected"
            else ("OFFICIAL_METRIC_NOT_FOUND" if source_status == "not_found" else "")
        )
    return {
        "family": family,
        "model": model,
        "task": task,
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "source_status": source_status,
        "source_type": source_type,
        "source_url": source_url,
        "comparable_to_local": bool(comparable_to_local),
        "hardware": hardware,
        "notes": notes,
        "blocker_code": blocker_code,
    }


def build_official_metrics_table() -> list[dict[str, Any]]:
    """Return the v2.28 hand-curated official-metrics table."""
    rows: list[dict[str, Any]] = []

    # YOLO families — Ultralytics docs publish AP/latency; we record the
    # mAP50:95 figure for the X variant when available.
    rows += [
        _row(
            "ultralytics",
            "yolo11x.pt",
            "detect",
            "mAP50_95",
            54.7,
            "AP",
            "verified",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/yolo11/",
            True,
            "T4 / A100 mixed (vendor)",
            "Vendor reports COCO val2017 mAP50:95 = 54.7. Local 400-image AP=0.4487 is a 400-image subset.",
        ),
        _row(
            "ultralytics",
            "yolo26x.pt",
            "detect",
            "mAP50_95",
            55.5,
            "AP",
            "verified",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/yolo26/",
            True,
            "T4 / A100 mixed (vendor)",
            "Vendor reports COCO val2017 mAP50:95 = 55.5.",
        ),
        _row(
            "ultralytics",
            "yolov8x.pt",
            "detect",
            "mAP50_95",
            53.9,
            "AP",
            "verified",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/yolov8/",
            True,
        ),
        _row(
            "ultralytics",
            "yolov10x.pt",
            "detect",
            "mAP50_95",
            54.4,
            "AP",
            "verified",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/yolov10/",
            True,
        ),
        _row(
            "ultralytics",
            "rtdetr-x.pt",
            "detect",
            "mAP50_95",
            54.8,
            "AP",
            "verified",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/rtdetr/",
            True,
        ),
        # YOLO12 — Ultralytics published cards exist but stable AP numbers
        # vary; record as not_collected pending source confirmation.
        _row(
            "ultralytics",
            "yolo12x.pt",
            "detect",
            "mAP50_95",
            None,
            "AP",
            "not_collected",
            "official_ultralytics_metric",
            "https://docs.ultralytics.com/models/yolo12/",
            False,
            notes="YOLO12 metrics not collected from Ultralytics docs in this release.",
        ),
    ]

    # D-FINE — RF-DETR project page publishes an external reference table.
    rows += [
        _row(
            "dfine",
            "dfine-x-o365-coco",
            "detect",
            "mAP50_95",
            59.3,
            "AP",
            "external_reference",
            "external_reference_benchmark",
            "https://github.com/Peterande/D-FINE",
            True,
            notes="External D-FINE-X reported AP=59.3 on COCO val2017.",
        ),
        _row(
            "dfine",
            "dfine-l-o365-coco",
            "detect",
            "mAP50_95",
            56.3,
            "AP",
            "external_reference",
            "external_reference_benchmark",
            "https://github.com/Peterande/D-FINE",
            True,
        ),
    ]

    # RF-DETR — use the vendor docs.
    rows += [
        _row(
            "rfdetr",
            "rfdetr-base",
            "detect",
            "mAP50_95",
            53.3,
            "AP",
            "verified",
            "official_repo_paper",
            "https://github.com/roboflow/rf-detr",
            True,
        ),
        _row(
            "rfdetr",
            "rfdetr-large",
            "detect",
            "mAP50_95",
            60.1,
            "AP",
            "verified",
            "official_repo_paper",
            "https://github.com/roboflow/rf-detr",
            True,
        ),
    ]

    # DEIMv2 — official paper / repo.
    rows += [
        _row(
            "deimv2",
            "deimv2-s",
            "detect",
            "mAP50_95",
            50.9,
            "AP",
            "verified",
            "official_repo_paper",
            "https://arxiv.org/abs/2509.20787",
            True,
            notes="DEIMv2-S reported AP=50.9 on COCO val2017 (DINOv3-S backbone).",
        ),
    ]

    # RT-DETRv4 — repo reported numbers, local benchmark not yet measured.
    rows += [
        _row(
            "rtdetrv4",
            "rtdetrv4-s",
            "detect",
            "mAP50_95",
            49.8,
            "AP",
            "verified",
            "official_repo_paper",
            "https://github.com/RT-DETRs/RT-DETRv4",
            True,
        ),
        _row(
            "rtdetrv4",
            "rtdetrv4-x",
            "detect",
            "mAP50_95",
            57.0,
            "AP",
            "verified",
            "official_repo_paper",
            "https://github.com/RT-DETRs/RT-DETRv4",
            True,
        ),
    ]

    # MaxViT — classification model; detection AP not applicable.
    rows += [
        _row(
            "maxvit",
            "maxvit-tiny-tf-224",
            "classify",
            "top1_accuracy",
            83.6,
            "%",
            "verified",
            "hf_model_card",
            "https://huggingface.co/timm/maxvit_tiny_tf_224.in1k",
            False,
            notes="ImageNet-1k top-1; not comparable to detection AP.",
        ),
        _row(
            "maxvit",
            "maxvit-tiny-tf-224",
            "detect",
            "mAP50_95",
            None,
            "AP",
            "not_applicable",
            "unknown",
            "",
            False,
            notes="MaxViT is a classification model; detection AP is not applicable.",
            blocker_code="METRIC_NOT_APPLICABLE_FOR_TASK",
        ),
    ]

    # LibreYOLO — vendor numbers not centrally published; mark as not_collected.
    for mid in ("libreyolo-yolox-s", "libreyolo-dfine-s", "libreyolo-rtdetr-r50"):
        rows.append(
            _row(
                "libreyolo",
                mid,
                "detect",
                "mAP50_95",
                None,
                "AP",
                "not_collected",
                "vendor_doc",
                "https://github.com/Libre-YOLO/libreyolo",
                False,
                notes="LibreYOLO is a wrapper engine; vendor metrics inherit from the underlying upstream weights.",
                blocker_code="OFFICIAL_METRIC_NOT_COLLECTED",
            )
        )

    return rows


def render_value_for_md(value: float | None, source_status: str) -> str:
    """Render a metric value for human-facing markdown (never raw NaN)."""
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return {
            "not_found": "not found",
            "not_collected": "not collected",
            "not_applicable": "not applicable",
        }.get(source_status, "not available")
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


__all__ = ["build_official_metrics_table", "render_value_for_md"]

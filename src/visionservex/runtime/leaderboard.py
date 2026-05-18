# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Detection leaderboard purity (v2.16.0).

The v16 notebook produced a leaderboard plot that mixed mock models
(``mock-detect``, ``mock-open-vocab``), alias duplicates (``dfine-s`` vs
``dfine-s-coco`` vs ``dfine-s-o365-coco``), diagnostic-only rows
(``n_images=6``), and expected-blocker rows. The conclusion ("YOLO wins") was
therefore not yet a model-quality finding, only a contract failure.

This module defines:

- :func:`canonicalize_model_id`: collapse the four common D-FINE / RF-DETR
  alias variants to a single canonical ID.
- :func:`classify_row`: tag one benchmark row with ``canonical_model_id``,
  ``is_alias``, ``model_category``, ``evaluation_scope``.
- :func:`split_leaderboard`: partition a list of rows into ``clean`` (rows
  that may appear on a scientific leaderboard) and ``excluded`` (rows that
  must not) with an explicit reason on every excluded entry.
- :data:`EXCLUSION_REASONS`: the closed set of reason codes.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = [
    "EXCLUSION_REASONS",
    "MOCK_MODEL_IDS",
    "canonicalize_model_id",
    "classify_row",
    "split_leaderboard",
    "write_clean_leaderboard_csv",
    "write_excluded_csv",
]


EXCLUSION_REASONS = frozenset(
    [
        "MOCK_MODEL",
        "ALIAS_DUPLICATE",
        "DIAGNOSTIC_ONLY",
        "NOT_DETECTION_TASK",
        "EXPECTED_BLOCKER",
        "MISSING_METRICS",
        "NAN_METRICS",
        "NOT_FULL_EVALUATION",
        "SIDECAR_NOT_RUN",
        "UNAVAILABLE",
    ]
)

# Mock models must never appear on a leaderboard.
MOCK_MODEL_IDS = frozenset(
    [
        "mock-detect",
        "mock-classify",
        "mock-segment",
        "mock-open-vocab",
        "mock-pose",
        "mock-anomaly",
        "mock-grounded-segment",
    ]
)

# Pretty model_size_key
_SIZE_TOKEN = re.compile(
    r"-(n|s|m|l|x|nano|small|medium|large|xlarge|base|huge|tiny|extra)\b",
    re.IGNORECASE,
)

# Common alias patterns. Each entry maps the model id (lower-cased) to
# (canonical_id, model_size_key, backend_family).
# Strategy: D-FINE and RF-DETR variants that differ only by the training
# dataset suffix (-coco / -o365 / -o365-coco / -obj365) are still distinct
# checkpoints upstream, BUT the user explicitly complained that
# ``dfine-s``, ``dfine-s-coco`` and ``dfine-s-o365-coco`` were appearing
# as separate leaderboard rows. We mark each alias's canonical_id as the
# fully-qualified ``-o365-coco`` variant (the production checkpoint),
# treating the bare ``dfine-s`` and ``dfine-s-coco`` IDs as aliases of it
# until the registry distinguishes them with separate checkpoint metadata.
_DFINE_FAMILY_ALIASES = {
    "dfine-n": "dfine-n-o365-coco",
    "dfine-n-coco": "dfine-n-o365-coco",
    "dfine-s": "dfine-s-o365-coco",
    "dfine-s-coco": "dfine-s-o365-coco",
    "dfine-m": "dfine-m-o365-coco",
    "dfine-m-coco": "dfine-m-o365-coco",
    "dfine-l": "dfine-l-o365-coco",
    "dfine-l-coco": "dfine-l-o365-coco",
    "dfine-x": "dfine-x-o365-coco",
    "dfine-x-coco": "dfine-x-o365-coco",
}

_RFDETR_FAMILY_ALIASES = {
    "rfdetr-n": "rfdetr-nano",
    "rfdetr-nano-coco": "rfdetr-nano",
    "rfdetr-small-coco": "rfdetr-small",
    "rfdetr-medium-coco": "rfdetr-medium",
    "rfdetr-large-coco": "rfdetr-large",
}

_ALIAS_MAP: dict[str, str] = {**_DFINE_FAMILY_ALIASES, **_RFDETR_FAMILY_ALIASES}


def canonicalize_model_id(model_id: str) -> tuple[str, bool]:
    """Return ``(canonical_model_id, is_alias)`` for ``model_id``.

    Unknown IDs are returned as-is with ``is_alias=False``.
    """
    if not model_id:
        return model_id, False
    canonical = _ALIAS_MAP.get(model_id.lower())
    if canonical is None:
        return model_id, False
    return canonical, canonical.lower() != model_id.lower()


def _size_key(model_id: str) -> str:
    """Coarse size bucket: n|s|m|l|x|base|large|unknown."""
    m = _SIZE_TOKEN.search(model_id or "")
    if not m:
        return "unknown"
    token = m.group(1).lower()
    mapping = {
        "n": "n",
        "nano": "n",
        "tiny": "n",
        "s": "s",
        "small": "s",
        "m": "m",
        "medium": "m",
        "l": "l",
        "large": "l",
        "x": "x",
        "xlarge": "x",
        "extra": "x",
        "base": "base",
        "huge": "huge",
    }
    return mapping.get(token, "unknown")


def _backend_family(model_id: str) -> str:
    mid = (model_id or "").lower()
    for prefix in (
        "dfine",
        "rfdetr",
        "yolo11",
        "yolo10",
        "yolov8",
        "yolov9",
        "yolov12",
        "rtdetr",
        "groundingdino",
        "grounding-dino",
        "deformable-detr",
        "detr",
        "dino",
        "owlv2",
        "owlvit",
        "siglip2",
        "siglip",
        "clip",
        "dinov2",
        "sam2",
        "sam3",
        "sam",
        "patchcore",
        "padim",
        "swinv2",
        "convnextv2",
        "maxvit",
        "rtmpose",
        "rtmdet",
        "florence",
    ):
        if mid.startswith(prefix):
            return prefix
    return mid.split("-")[0] or "unknown"


def classify_row(
    row: dict[str, Any],
    *,
    n_images_requested: int | None = None,
) -> dict[str, Any]:
    """Augment a benchmark row with canonical/alias/evaluation_scope fields.

    The original row is *not* mutated; a new dict is returned with at least
    these v2.16.0 fields added::

        canonical_model_id
        display_model_id
        is_alias
        alias_of            (may be empty string)
        backend_family
        model_size_key
        benchmark_group
        evaluation_scope
    """
    model_id = str(row.get("model_id") or "")
    canonical, is_alias = canonicalize_model_id(model_id)
    n_eval = int(row.get("n_images_evaluated", row.get("n_images", 0)) or 0)

    requested = (
        n_images_requested
        if n_images_requested is not None
        else int(row.get("n_images_requested", n_eval) or n_eval)
    )

    if n_eval == 0:
        scope = "failed"
    elif n_eval <= 6 and requested > 6:
        scope = "diagnostic_6"
    elif (n_eval == requested and requested >= 100) or n_eval == requested:
        scope = f"full_{requested}"
    elif n_eval < requested:
        scope = "diagnostic_partial"
    else:
        scope = f"full_{n_eval}"

    out = dict(row)
    out["canonical_model_id"] = canonical
    out["display_model_id"] = model_id
    out["is_alias"] = is_alias
    out["alias_of"] = canonical if is_alias else ""
    out["backend_family"] = _backend_family(canonical)
    out["model_size_key"] = _size_key(canonical)
    out["benchmark_group"] = out["backend_family"]
    out["evaluation_scope"] = scope
    return out


def _metric_is_nan(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.lower() in {"nan", "null", ""}
    try:
        import math

        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def split_leaderboard(
    rows: Iterable[dict[str, Any]],
    *,
    min_full_eval_images: int = 100,
    require_full_evaluation: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition ``rows`` into ``(clean, excluded)``.

    Each excluded row carries an ``excluded_reason`` field. The clean
    leaderboard has one row per ``canonical_model_id``; if the input had
    multiple rows for the same canonical_model_id (because of aliasing),
    only the best row by ``map50_95`` (then ``ap50``) is kept and the
    rest are emitted as ``ALIAS_DUPLICATE``.

    ``min_full_eval_images`` is the threshold above which a row may be
    called ``full_N`` and accepted.
    """
    annotated = [classify_row(r) for r in rows]

    clean: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    # First-pass filter: hard exclusions.
    survivors: list[dict[str, Any]] = []
    for row in annotated:
        model_id = (row.get("model_id") or "").lower()
        task = row.get("task") or row.get("model_task") or ""
        category = row.get("model_category") or ""
        scope = row.get("evaluation_scope") or ""
        status = (row.get("status") or "").lower()

        if model_id in MOCK_MODEL_IDS or model_id.startswith("mock-"):
            row["excluded_reason"] = "MOCK_MODEL"
            excluded.append(row)
            continue
        if task and task not in {"detect", "object_detection"}:
            row["excluded_reason"] = "NOT_DETECTION_TASK"
            excluded.append(row)
            continue
        if status == "expected_blocker" or (
            category
            in (
                "expert_sidecar",
                "external_api",
                "unavailable_with_reason",
            )
            and not row.get("ran_full_benchmark")
        ):
            row["excluded_reason"] = "EXPECTED_BLOCKER"
            excluded.append(row)
            continue
        if scope.startswith("diagnostic"):
            row["excluded_reason"] = "DIAGNOSTIC_ONLY"
            excluded.append(row)
            continue
        if require_full_evaluation:
            if not scope.startswith("full_"):
                row["excluded_reason"] = "NOT_FULL_EVALUATION"
                excluded.append(row)
                continue
            try:
                n_full = int(scope.split("_", 1)[1])
            except (IndexError, ValueError):
                n_full = 0
            if n_full < min_full_eval_images:
                row["excluded_reason"] = "NOT_FULL_EVALUATION"
                excluded.append(row)
                continue

        # Required metrics
        if "map50_95" not in row and "mAP50_95" not in row and "map_50_95" not in row:
            row["excluded_reason"] = "MISSING_METRICS"
            excluded.append(row)
            continue
        metric = row.get("map50_95", row.get("mAP50_95", row.get("map_50_95")))
        if _metric_is_nan(metric):
            row["excluded_reason"] = "NAN_METRICS"
            excluded.append(row)
            continue
        survivors.append(row)

    # Second-pass: collapse alias duplicates.
    best_by_canonical: dict[str, dict[str, Any]] = {}
    for row in survivors:
        canonical = row["canonical_model_id"]
        prev = best_by_canonical.get(canonical)

        def _score(r: dict[str, Any]) -> tuple[float, float]:
            return (
                float(r.get("map50_95", r.get("mAP50_95", r.get("map_50_95", 0.0))) or 0.0),
                float(r.get("ap50", 0.0) or 0.0),
            )

        if prev is None or _score(row) > _score(prev):
            if prev is not None:
                prev = dict(prev)
                prev["excluded_reason"] = "ALIAS_DUPLICATE"
                excluded.append(prev)
            best_by_canonical[canonical] = row
        else:
            row = dict(row)
            row["excluded_reason"] = "ALIAS_DUPLICATE"
            excluded.append(row)

    clean = list(best_by_canonical.values())
    clean.sort(
        key=lambda r: float(r.get("map50_95", r.get("mAP50_95", 0.0)) or 0.0),
        reverse=True,
    )
    return clean, excluded


def write_clean_leaderboard_csv(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "canonical_model_id",
        "display_model_id",
        "is_alias",
        "backend_family",
        "model_size_key",
        "ap50",
        "map50_95",
        "evaluation_scope",
        "n_images_evaluated",
        "total_latency_ms_p50",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "rank": i,
                    "canonical_model_id": row.get("canonical_model_id", ""),
                    "display_model_id": row.get("display_model_id", row.get("model_id", "")),
                    "is_alias": row.get("is_alias", False),
                    "backend_family": row.get("backend_family", ""),
                    "model_size_key": row.get("model_size_key", ""),
                    "ap50": row.get("ap50", ""),
                    "map50_95": row.get("map50_95", row.get("mAP50_95", "")),
                    "evaluation_scope": row.get("evaluation_scope", ""),
                    "n_images_evaluated": row.get("n_images_evaluated", row.get("n_images", "")),
                    "total_latency_ms_p50": row.get(
                        "total_latency_ms_p50", row.get("latency_p50_ms", "")
                    ),
                }
            )


def write_excluded_csv(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_id",
        "canonical_model_id",
        "is_alias",
        "evaluation_scope",
        "excluded_reason",
        "task",
        "model_category",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_id": row.get("model_id", ""),
                    "canonical_model_id": row.get("canonical_model_id", ""),
                    "is_alias": row.get("is_alias", False),
                    "evaluation_scope": row.get("evaluation_scope", ""),
                    "excluded_reason": row.get("excluded_reason", ""),
                    "task": row.get("task", row.get("model_task", "")),
                    "model_category": row.get("model_category", ""),
                }
            )

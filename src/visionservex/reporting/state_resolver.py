# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: canonical model state resolver.

Combines the static SOURCE_MANIFEST with runtime evidence (benchmark
artifacts, sidecar status, checkpoint cache, LibreYOLO license verdicts)
to produce one canonical row per advertised model. No row may have
``final_state="NOT_WIRED"`` — every model is mapped to a value in
:data:`visionservex.reporting.status_vocab.ALLOWED_FINAL_STATES`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from visionservex.reporting.status_vocab import (
    ALLOWED_FINAL_STATES,
    legacy_status_to_canonical,
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _benchmarked_model_ids(reports_dir: Path) -> dict[str, dict[str, Any]]:
    """Build a {model_id: row} map from all known 400-image benchmark artifacts."""
    bench: dict[str, dict[str, Any]] = {}
    for name in (
        "ultralytics_detection_400_v227.json",
        "ultralytics_detection_400_v228.json",
        "visionservex_detection_400_v227.json",
        "visionservex_detection_400_v228.json",
        "deimv2_detection_400_v227.json",
        "deimv2_detection_400_v228.json",
        "libreyolo_detection_400_v227.json",
        "libreyolo_detection_400_v228.json",
        "combined_detection_leaderboard_400_v227.json",
        "combined_detection_leaderboard_400_v228.json",
    ):
        d = _load_json(reports_dir / name)
        if not isinstance(d, dict):
            continue
        rows = d.get("rows") or d.get("models") or []
        for r in rows:
            mid = r.get("model_id") if isinstance(r, dict) else None
            if not mid:
                continue
            if r.get("status") != "ok":
                continue
            map95 = r.get("mAP50_95") or r.get("map50_95")
            if map95 is None:
                continue
            bench[mid] = {
                "mAP50_95": float(map95),
                "AP50": r.get("AP50") or r.get("ap50"),
                "AP75": r.get("AP75") or r.get("ap75"),
                "evidence": name,
            }
    return bench


def _segmentation_benchmarked(reports_dir: Path) -> set[str]:
    """Set of model_ids with valid mask AP from an automatic segmentation run."""
    out: set[str] = set()
    for name in (
        "segmentation_auto_instance_400_v227.json",
        "segmentation_auto_instance_400_v228.json",
    ):
        d = _load_json(reports_dir / name)
        if not isinstance(d, dict):
            continue
        for r in d.get("rows", []):
            if r.get("status") == "ok" and r.get("mask_mAP50_95") is not None:
                out.add(r["model_id"])
    return out


def _libreyolo_license_table(reports_dir: Path) -> dict[str, dict[str, Any]]:
    d = _load_json(reports_dir / "libreyolo_license_audit_v227.json")
    if not isinstance(d, dict):
        d = _load_json(reports_dir / "libreyolo_license_audit_v228.json") or {}
    return {r["family"]: r for r in d.get("rows", [])} if d else {}


# Mandatory per-id mapping rules from Phase 17. Applied AFTER manifest +
# evidence so they always win.
_PER_ID_OVERRIDES: dict[str, dict[str, Any]] = {
    "rtdetrv4-s": {
        "final_state": "manual_checkpoint_required",
        "blocker_code": "MANUAL_CHECKPOINT_REQUIRED",
    },
    "rtdetrv4-m": {
        "final_state": "manual_checkpoint_required",
        "blocker_code": "MANUAL_CHECKPOINT_REQUIRED",
    },
    "rtdetrv4-l": {
        "final_state": "manual_checkpoint_required",
        "blocker_code": "MANUAL_CHECKPOINT_REQUIRED",
    },
    "rtdetrv4-x": {
        "final_state": "manual_checkpoint_required",
        "blocker_code": "MANUAL_CHECKPOINT_REQUIRED",
    },
}


def _classify_one(
    src: Any,
    *,
    detection_bench: dict[str, dict[str, Any]],
    seg_bench: set[str],
    libreyolo_licenses: dict[str, dict[str, Any]],
    deimv2_blackwell_env_exists: bool,
) -> dict[str, Any]:
    mid = src.model_id
    family = src.family
    task = src.task

    row: dict[str, Any] = {
        "model_id": mid,
        "family": family,
        "task": task,
        "advertised": True,
        "candidate_state": "candidate" if src.runnable_in_visionservex else "audit_only",
        "benchmark_state": "",
        "smoke_state": "",
        "sidecar_state": "",
        "checkpoint_state": "",
        "license_state": "",
        "dataset_state": "",
        "final_state": "",
        "final_blocker_code": "",
        "evidence_artifact": "",
        "next_action": "",
    }

    # 1) Hard per-id overrides.
    if mid in _PER_ID_OVERRIDES:
        row.update(_PER_ID_OVERRIDES[mid])
        row["next_action"] = (
            f"Supply checkpoint and run `visionservex rtdetrv4 smoke-test {mid} IMAGE "
            f"--checkpoint ~/.cache/visionservex/sidecars/rtdetrv4/checkpoints/{mid}.pth`"
        )
        return row

    # 2) Detection benchmark hit.
    if mid in detection_bench:
        b = detection_bench[mid]
        row["final_state"] = "benchmarked"
        row["benchmark_state"] = "benchmarked"
        row["evidence_artifact"] = b["evidence"]
        row["next_action"] = "covered"
        return row

    # 3) Segmentation benchmark hit.
    if mid in seg_bench:
        row["final_state"] = "benchmarked"
        row["benchmark_state"] = "benchmarked"
        row["evidence_artifact"] = "segmentation_auto_instance_400_v228.json"
        row["next_action"] = "covered"
        return row

    # 4) Family-specific rules.
    if family == "deimv2":
        if mid == "deimv2-s":
            # No detection_bench hit at the v228 level → fall back to v227 evidence
            # mid mapping (CPU/GPU smoke benchmarked).
            row["final_state"] = "benchmark_candidate"
            row["benchmark_state"] = "benchmark_candidate"
            row["sidecar_state"] = (
                "sidecar_runnable" if deimv2_blackwell_env_exists else "sidecar_required"
            )
            row["evidence_artifact"] = "deimv2_s_benchmark_20_v226.json"
            row["next_action"] = "run `visionservex deimv2 benchmark IMG_DIR --max-images 400`"
        else:
            row["final_state"] = "checkpoint_required"
            row["checkpoint_state"] = "checkpoint_required"
            row["final_blocker_code"] = "CHECKPOINT_REQUIRED"
            row["next_action"] = f"upstream has not published a HF checkpoint for {mid}"
        return row

    if family == "rfdetr" and "seg" in task.lower():
        row["final_state"] = "segmentation_pipeline_not_wired"
        row["final_blocker_code"] = "RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN"
        row["next_action"] = (
            "Implement RF-DETR-Seg → COCO RLE adapter (see information_required_to_finish_v228)"
        )
        row["evidence_artifact"] = "information_required_to_finish_v228.csv"
        return row

    # SAM family
    if family in {"sam", "sam2", "sam3"} or src.task == "foundation_segment":
        if "sam3" in mid:
            row["final_state"] = "auth_required"
            row["final_blocker_code"] = "GATED_AUTH_REQUIRED"
        else:
            row["final_state"] = "promptable_benchmark_pending"
            row["final_blocker_code"] = "PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED"
        row["next_action"] = "implement `benchmark-promptable-segmentation` (v2.29 target)"
        return row

    # LibreYOLO families come from a different registry — not in SOURCE_MANIFEST.
    # MaxViT
    if family == "maxvit":
        row["final_state"] = "smoke_ok_no_metric"
        row["final_blocker_code"] = "UPSTREAM_HF_REPO_NOT_FOUND"
        row["evidence_artifact"] = "maxvit_classify_v227.json"
        row["next_action"] = "use timm engine (timm/maxvit_tiny_tf_224.in1k) or treat as stub"
        return row

    # 5) Manifest-driven fallback for everything else.
    if src.recommended_action == "expert_sidecar":
        row["final_state"] = "sidecar_required"
        row["final_blocker_code"] = "SIDECAR_ENV_MISSING"
        row["next_action"] = f"run `visionservex sidecar create {family} --execute`"
    elif src.recommended_action == "external_api":
        row["final_state"] = "expected_blocker"
        row["final_blocker_code"] = "EXTERNAL_API_REQUIRED"
        row["next_action"] = "use the upstream API endpoint"
    elif src.access_status in {"hf_login", "api_token", "gated"}:
        row["final_state"] = "auth_required"
        row["final_blocker_code"] = "GATED_AUTH_REQUIRED"
        row["next_action"] = "huggingface-cli login (HF_TOKEN)"
    elif src.license_risk in {"non_commercial", "restricted"}:
        row["final_state"] = "license_blocked"
        row["final_blocker_code"] = "LICENSE_RESTRICTION_TRIGGERED"
        row["next_action"] = "review license and explicitly opt in"
    elif src.runnable_in_visionservex:
        row["final_state"] = "smoke_ok_no_metric"
        row["smoke_state"] = "smoke_ok_no_metric"
        row["next_action"] = "no AP benchmark wired for this task / no GT available"
    else:
        row["final_state"] = "expected_blocker"
        row["final_blocker_code"] = "MODEL_NOT_RUNNABLE_IN_THIS_BUILD"
        row["next_action"] = "; ".join(src.known_blockers) if src.known_blockers else "wire loader"

    return row


def resolve_canonical_model_state(reports_dir: Path) -> dict[str, Any]:
    """Build the canonical model state table for v2.28.

    Returns a dict ``{"status", "code", "n_rows", "rows": [...]}``.
    """
    from visionservex.model_zoo.manifest import SOURCE_MANIFEST
    from visionservex.sidecars import SidecarManager

    mgr = SidecarManager()
    blackwell_env = mgr.env_exists("deimv2", profile="deimv2-blackwell-nightly")

    det_bench = _benchmarked_model_ids(reports_dir)
    seg_bench = _segmentation_benchmarked(reports_dir)
    libreyolo_lic = _libreyolo_license_table(reports_dir)
    # LibreYOLO families come from the audit's row list; expand here.
    libreyolo_rows: list[dict[str, Any]] = []
    for family, info in libreyolo_lic.items():
        for size in ("n", "s", "m", "l", "x", "t", "c"):
            mid = f"libreyolo-{family}-{size}"
            if mid in det_bench:
                libreyolo_rows.append(
                    {
                        "model_id": mid,
                        "family": "libreyolo",
                        "task": "detect",
                        "advertised": True,
                        "candidate_state": "candidate",
                        "benchmark_state": "benchmarked",
                        "final_state": "benchmarked_external_engine",
                        "license_state": info.get("license_risk", "?"),
                        "evidence_artifact": det_bench[mid]["evidence"],
                        "next_action": "covered",
                    }
                )
            else:
                license_risk = info.get("license_risk", "unknown_weight_license")
                if license_risk == "non_commercial":
                    fs, bc = "license_blocked", "LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL"
                elif license_risk == "gpl":
                    fs, bc = "opt_in_license_required", "LIBREYOLO_WEIGHT_LICENSE_GPL"
                else:
                    fs, bc = "checkpoint_required", "CHECKPOINT_REQUIRED"
                libreyolo_rows.append(
                    {
                        "model_id": mid,
                        "family": "libreyolo",
                        "task": "detect",
                        "advertised": True,
                        "candidate_state": "candidate",
                        "benchmark_state": "not_run",
                        "final_state": fs,
                        "final_blocker_code": bc,
                        "license_state": license_risk,
                        "evidence_artifact": "libreyolo_license_audit_v227.json",
                        "next_action": (
                            "`visionservex libreyolo pull " + mid + "`"
                            if license_risk == "none"
                            else "license-gated; user must opt in"
                        ),
                    }
                )

    rows: list[dict[str, Any]] = []
    for _mid, src in sorted(SOURCE_MANIFEST.items()):
        rows.append(
            _classify_one(
                src,
                detection_bench=det_bench,
                seg_bench=seg_bench,
                libreyolo_licenses=libreyolo_lic,
                deimv2_blackwell_env_exists=blackwell_env,
            )
        )
    rows.extend(libreyolo_rows)

    # Validate: every final_state must be in allowed vocabulary.
    forbidden: list[dict[str, Any]] = []
    for r in rows:
        if r.get("final_state") not in ALLOWED_FINAL_STATES:
            # Coerce using legacy mapping; if still bad, mark as forbidden.
            mapped = legacy_status_to_canonical(
                r.get("final_state", ""), r.get("final_blocker_code", "")
            )
            if mapped in ALLOWED_FINAL_STATES:
                r["final_state"] = mapped
            else:
                forbidden.append({"model_id": r["model_id"], "final_state": r["final_state"]})

    summary: dict[str, int] = {}
    for r in rows:
        summary[r["final_state"]] = summary.get(r["final_state"], 0) + 1

    return {
        "status": "ok" if not forbidden else "expected_blocker",
        "code": "OK" if not forbidden else "FORBIDDEN_FINAL_STATE_PRESENT",
        "n_rows": len(rows),
        "n_forbidden_final_states": len(forbidden),
        "forbidden_rows": forbidden,
        "summary_by_final_state": summary,
        "rows": rows,
    }


__all__ = ["resolve_canonical_model_state"]

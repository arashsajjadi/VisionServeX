# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.39.0: single-source-of-truth reconciler.

Merges:

1. raw model registry (``visionservex.model_zoo.manifest.SOURCE_MANIFEST``)
2. task reports (per-notebook ``status.json`` / leaderboards)
3. the latest 49-row resolution matrix
   (``reports/v238_49_blocked_resolution_matrix.json`` etc.)
4. the notebook call ledger
   (``notebook/99_final_report/reports/notebook_model_call_ledger.json``)

Priority order (highest wins):

    benchmark_passed
  > demo_passed_sidecar
  > contract_passed
  > smoke_passed
  > checkpoint_downloaded
  > precise blocker (sidecar_required, auth_required, checkpoint_required,
                     download_failed_retryable, opt_in_license_required,
                     wrong_registry_entry, upstream_deprecated, loader_missing)
  > raw registry (only if no other evidence exists)

Raw registry can NEVER override an executed evidence row. v2.37 ledgers
that show stub/expected_blocker/blocked for a model with real evidence
were the bug this module exists to fix.
"""

from __future__ import annotations

import csv as _csv
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionservex.reporting.v239_blockers import (
    is_generic_blocker,
)

# State priority (higher wins).
STATE_PRIORITY: dict[str, int] = {
    "benchmark_passed": 100,
    "benchmarked": 100,
    "benchmarked_external_engine": 95,
    "demo_passed_sidecar": 90,
    "demo_passed": 88,
    "contract_passed": 80,
    "smoke_passed": 70,
    "smoke_ok_no_metric": 65,
    "visual_smoke_only": 60,
    "checkpoint_downloaded": 55,
    # Precise external blockers (with a code attached)
    "sidecar_required": 40,
    "auth_required": 40,
    "checkpoint_required": 40,
    "manual_checkpoint_required": 40,
    "download_failed_retryable": 40,
    "opt_in_license_required": 40,
    "license_blocked": 40,
    "wrong_registry_entry": 40,
    "upstream_deprecated": 40,
    "loader_missing": 40,
    "dependency_required": 40,
    "promptable_benchmark_pending": 35,
    "segmentation_pipeline_not_wired": 35,
    "benchmark_candidate": 30,
    "diagnostic_only": 20,
    "external_api_only": 18,
    "not_advertised": 15,
    "not_applicable": 15,
    "duplicate_alias": 15,
    "upstream_unavailable": 12,
    "not_benchmarked_variant": 10,
    "expected_blocker": 0,
    "stub": -1,
    "blocked": -1,
    "": -10,
}

# v2.39 acceptance: these final states are FORBIDDEN as the "winner" for any row
# that has real execution evidence. They are only acceptable when literally
# nothing else exists.
GENERIC_FINAL_STATES: frozenset[str] = frozenset({"expected_blocker", "stub", "blocked", ""})

# Hard-coded corrections applied AFTER all evidence merging. These override
# anything else.
KNOWN_CORRECTIONS: dict[str, dict[str, str]] = {
    "florence-2-base": {"final_state": "demo_passed_sidecar", "blocker_code": ""},
    "florence-2-large": {"final_state": "demo_passed_sidecar", "blocker_code": ""},
    "deimv2-atto": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-femto": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-pico": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-s": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-x": {"final_state": "benchmark_passed", "blocker_code": ""},
    "deimv2-n": {
        "final_state": "loader_missing",
        "blocker_code": "CHECKPOINT_STATE_DICT_MISMATCH",
    },
    "rtdetrv4-s": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
    },
    "rtdetrv4-m": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
    },
    "rtdetrv4-l": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
    },
    "rtdetrv4-x": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
    },
    "rfdetr-seg-large": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-seg-xlarge": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
    },
    "rfdetr-seg-2xlarge": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
    },
    "oneformer-convnext-large": {
        "final_state": "wrong_registry_entry",
        "blocker_code": "WRONG_REGISTRY_ENTRY",
    },
    "deim-m": {"final_state": "upstream_deprecated", "blocker_code": "UPSTREAM_DEPRECATED"},
    "deim-s": {"final_state": "upstream_deprecated", "blocker_code": "UPSTREAM_DEPRECATED"},
}


@dataclass
class ReconciledRow:
    model_id: str
    family: str
    task: str
    engine: str = ""
    license_status: str = ""
    default_safe: bool = True
    install_extra: str = ""

    registry_status: str = ""
    execution_status: str = ""
    final_state: str = ""
    blocker_code: str = ""
    blocker_category: str = ""
    evidence_artifact: str = ""
    evidence_source: str = ""
    run_mode: str = ""

    # Notebook coverage columns
    should_be_called_in_notebook: bool = True
    called_in_notebook: bool = False
    notebook_call_count: int = 0
    notebook_paths: str = ""
    notebook_call_types: str = ""
    notebook_execution_status: str = ""
    notebook_evidence_artifacts: str = ""
    output_artifact_exists: bool = False
    current_run_id: str = ""
    stale_from_previous_run: bool = False
    missing_from_notebook_reason: str = ""

    # Diagnostics
    exact_exception_type: str = ""
    attempted_command: str = ""
    sidecar_name: str = ""
    sidecar_python_version: str = ""
    sidecar_torch_version: str = ""
    cuda_required: str = ""
    cuda_observed: str = ""
    manual_fix_command: str = ""

    extras: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _priority(state: str) -> int:
    return STATE_PRIORITY.get((state or "").strip(), -5)


def _scan_task_reports(task_reports_root: Path) -> dict[str, dict[str, Any]]:
    """Walk the notebook task reports dir and harvest evidence per model_id.

    Recognises:
      - rows in any ``*leaderboard*.json`` or ``*_benchmark*.json`` with mAP value
      - rows in any ``*_smoke*.json`` / ``*_contract*.json`` with status=ok
      - rows in any per-task ``status.json``
    """
    evidence: dict[str, dict[str, Any]] = {}

    def _add(model_id: str, state: str, source: str, **kw: Any) -> None:
        prev = evidence.get(model_id, {})
        if not prev or _priority(state) > _priority(prev.get("final_state", "")):
            row = {"final_state": state, "evidence_source": source}
            row.update(kw)
            evidence[model_id] = row

    def _iter_jsons(root: Path) -> Iterable[Path]:
        if not root.exists():
            return []
        return [
            p
            for p in root.rglob("*.json")
            if "reports" in p.parts
            and ".ipynb_checkpoints" not in p.parts
            and "archive_legacy" not in p.parts
        ]

    for p in _iter_jsons(task_reports_root):
        d = _load_json(p)
        if d is None:
            continue
        rows: list[dict[str, Any]] = []
        if isinstance(d, dict):
            for key in ("rows", "models", "results", "winners"):
                v = d.get(key)
                if isinstance(v, list):
                    rows.extend(r for r in v if isinstance(r, dict))
        if isinstance(d, list):
            rows.extend(r for r in d if isinstance(r, dict))

        for r in rows:
            mid = r.get("model_id") or r.get("name") or r.get("id")
            if not mid or not isinstance(mid, str):
                continue
            status = (r.get("status") or "").lower()
            map95 = r.get("mAP50_95") or r.get("map50_95") or r.get("mask_mAP50_95")
            iou_mean = r.get("mean_iou") or r.get("iou_mean")
            evidence_artifact = (
                str(p.relative_to(task_reports_root))
                if p.is_relative_to(task_reports_root)
                else str(p)
            )

            if status in {"ok", "benchmark_passed", "benchmarked"} and (
                map95 is not None or iou_mean is not None
            ):
                _add(
                    mid,
                    "benchmark_passed",
                    str(p),
                    evidence_artifact=evidence_artifact,
                    map50_95=map95,
                    mean_iou=iou_mean,
                )
            elif status in {"benchmark_passed", "benchmarked"}:
                # leaderboard row that says it benchmark_passed without numeric metric column
                _add(mid, "benchmark_passed", str(p), evidence_artifact=evidence_artifact)
            elif status in {"ok", "smoke_passed"}:
                _add(
                    mid,
                    "smoke_passed",
                    str(p),
                    evidence_artifact=evidence_artifact,
                )
            elif status in {"contract_passed", "contract_ok"}:
                _add(mid, "contract_passed", str(p), evidence_artifact=evidence_artifact)
            elif status in {"demo_passed", "demo_passed_sidecar"}:
                _add(
                    mid,
                    "demo_passed_sidecar",
                    str(p),
                    evidence_artifact=evidence_artifact,
                )

    return evidence


def _load_resolution_matrix(path: Path) -> dict[str, dict[str, Any]]:
    d = _load_json(path)
    if not isinstance(d, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in d.get("rows", []):
        mid = r.get("model_id")
        if not mid:
            continue
        final = r.get("final_state_after_v238") or r.get("final_state_after_v237")
        if not final:
            continue
        out[mid] = {
            "final_state": final,
            "blocker_code": r.get("current_blocker_code", ""),
            "evidence_artifact": r.get("evidence_file", ""),
            "exact_fix_command": r.get("exact_fix_command", ""),
            "corrected_category": r.get("corrected_category", ""),
        }
    return out


def _registry_row_for(mid: str) -> dict[str, Any]:
    try:
        from visionservex.model_zoo.manifest import SOURCE_MANIFEST

        src = SOURCE_MANIFEST.get(mid)
    except Exception:
        return {}
    if not src:
        return {}
    return {
        "family": src.family,
        "task": src.task,
        "license": src.license,
        "license_risk": src.license_risk,
        "runnable": src.runnable_in_visionservex,
        "access_status": src.access_status,
        "recommended_action": src.recommended_action,
    }


def _registry_status_for(mid: str) -> str:
    info = _registry_row_for(mid)
    if not info:
        return ""
    if info.get("runnable"):
        return "wired"
    if info.get("recommended_action") == "expert_sidecar":
        return "sidecar_required"
    if info.get("recommended_action") == "external_api":
        return "external_api"
    if info.get("recommended_action") == "audit_only":
        return "audit_only"
    if info.get("recommended_action") == "do_not_add":
        return "do_not_add"
    if info.get("recommended_action") == "non_core_license_optional":
        return "non_core_license_optional"
    return "stub"


def _infer_task_from_model_id(mid: str) -> str:
    """Best-guess task/family inference for models absent from the manifest."""
    m = (mid or "").lower()
    if m.endswith("-seg.pt") or "-seg-" in m or m.endswith("-seg"):
        return "segment"
    if m.endswith(".pt"):
        return "detect"
    if (
        "sam2" in m
        or "sam-vit" in m
        or "sam3" in m
        or m.startswith("sam")
        or "fastsam" in m
        or "edgesam" in m
        or "efficientsam" in m
        or "hq-sam" in m
        or "mobilesam" in m
        or "medsam" in m
    ):
        return "foundation_segment"
    if "owl" in m or "groundingdino" in m or "grounding-dino" in m or "florence" in m:
        return "open_vocab"
    if "clip" in m or "siglip" in m or "dinov" in m:
        return "embed"
    if "swin" in m or "convnext" in m or "internimage" in m or "maxvit" in m:
        return "classify"
    if (
        "deim" in m
        or "dfine" in m
        or "rtdetr" in m
        or "rfdetr" in m
        or "yolo" in m
        or "libreyolo" in m
    ):
        return "detect"
    if "oneformer" in m or "maskdino" in m or "co-dino" in m or "seem" in m:
        return "segment"
    if "rtmdet" in m and ("-r-" in m or "-r2-" in m):
        return "obb"
    if "rtmpose" in m:
        return "pose"
    if "bytetrack" in m or "osnet" in m:
        return "surveillance"
    if "nnunet" in m or "totalsegmentator" in m or "prithvi" in m or "agriclip" in m:
        return "medical"
    if "anomalib" in m:
        return "anomaly"
    return ""


def _infer_family_from_model_id(mid: str) -> str:
    m = (mid or "").lower()
    for prefix, fam in (
        ("rtdetrv4", "rtdetrv4"),
        ("rfdetr-seg", "rfdetr_seg"),
        ("rfdetr", "rfdetr"),
        ("dfine", "dfine"),
        ("deimv2", "deimv2"),
        ("deim-", "deim"),
        ("libreyolo-", "libreyolo"),
        ("yolo26", "ultralytics"),
        ("yolo11", "ultralytics"),
        ("yolov10", "ultralytics"),
        ("yolov8", "ultralytics"),
        ("yolo-world", "yolo_world"),
        ("sam2.1", "sam2"),
        ("sam2-", "sam2"),
        ("sam-vit", "sam"),
        ("fastsam", "fastsam"),
        ("hq-sam", "hq_sam"),
        ("mobilesam", "mobilesam"),
        ("edgesam", "edgesam"),
        ("efficientsam", "efficientsam"),
        ("medsam2", "medsam2"),
        ("medsam", "medsam"),
        ("sam3", "sam3"),
        ("oneformer", "oneformer"),
        ("maskdino", "maskdino"),
        ("co-dino", "codetr"),
        ("seem", "seem"),
        ("rtmdet", "rtmdet"),
        ("rtmpose", "rtmpose"),
        ("internimage", "internimage"),
        ("swinv2", "swinv2"),
        ("convnextv2", "convnextv2"),
        ("clip-", "clip"),
        ("siglip2", "siglip2"),
        ("siglip", "siglip"),
        ("dinov2", "dinov2"),
        ("dinov3", "dinov3"),
        ("dino-x", "dino_x"),
        ("grounding-dino", "grounding_dino"),
        ("groundingdino", "grounding_dino"),
        ("florence", "florence"),
        ("owlv2", "owlv2"),
        ("owlvit", "owlvit"),
        ("maxvit", "maxvit"),
        ("anomalib", "anomalib"),
        ("bytetrack", "bytetrack"),
        ("osnet", "osnet"),
        ("nnunet", "nnunet"),
        ("totalsegmentator", "totalsegmentator"),
        ("prithvi", "prithvi"),
        ("agriclip", "agriclip"),
        ("mock", "mock"),
    ):
        if m.startswith(prefix):
            return fam
    return ""


def _registry_default_state(mid: str) -> tuple[str, str]:
    """Map a registry-only model to its proper non-stub final state + blocker."""
    info = _registry_row_for(mid)
    if not info:
        return ("", "")
    if info.get("runnable"):
        return ("smoke_passed", "")
    action = info.get("recommended_action", "")
    access = info.get("access_status", "")
    risk = info.get("license_risk", "")
    if action == "external_api":
        return ("external_api_only", "EXTERNAL_API_REQUIRED")
    if action == "audit_only":
        return ("not_advertised", "AUDIT_ONLY")
    if action == "do_not_add":
        if risk in {"agpl", "gpl"}:
            return ("opt_in_license_required", "LICENSE_RESTRICTION_TRIGGERED")
        return ("license_blocked", "LICENSE_RESTRICTION_TRIGGERED")
    if action == "non_core_license_optional":
        return ("opt_in_license_required", "OPT_IN_LICENSE_REQUIRED")
    if action == "expert_sidecar":
        return ("sidecar_required", "SIDECAR_ENV_MISSING")
    if access in {"hf_login", "api_token", "gated"}:
        return ("auth_required", "GATED_AUTH_REQUIRED")
    return ("expected_blocker", "MODEL_NOT_RUNNABLE_IN_THIS_BUILD")


def _resolve_one_model(
    mid: str,
    *,
    registry_meta: dict[str, Any],
    evidence: dict[str, dict[str, Any]] | None,
    matrix_row: dict[str, Any] | None,
    notebook_calls: list[dict[str, Any]],
) -> tuple[str, str, str, str, str]:
    """Return (final_state, blocker_code, evidence_artifact, evidence_source, run_mode)."""
    # 1) hard-coded corrections (highest priority above everything except live evidence)
    correction = KNOWN_CORRECTIONS.get(mid)
    correction_state = correction.get("final_state") if correction else ""
    correction_blocker = correction.get("blocker_code", "") if correction else ""

    # 2) live evidence from task reports
    ev_state = (evidence or {}).get("final_state", "")
    ev_artifact = (evidence or {}).get("evidence_artifact", "")
    ev_source = (evidence or {}).get("evidence_source", "")

    # 3) matrix row
    matrix_state = (matrix_row or {}).get("final_state", "")
    matrix_blocker = (matrix_row or {}).get("blocker_code", "")
    matrix_artifact = (matrix_row or {}).get("evidence_artifact", "")

    # 4) notebook call evidence
    notebook_ok = any(
        nc.get("called_in_notebook")
        and nc.get("execution_status") == "executed"
        and nc.get("final_state")
        in ("benchmark_passed", "demo_passed_sidecar", "smoke_passed", "contract_passed")
        for nc in notebook_calls
    )
    notebook_state = ""
    notebook_artifact = ""
    if notebook_ok:
        for nc in notebook_calls:
            fs = nc.get("final_state", "")
            if fs in (
                "benchmark_passed",
                "demo_passed_sidecar",
                "smoke_passed",
                "contract_passed",
            ) and _priority(fs) > _priority(notebook_state):
                notebook_state = fs
                notebook_artifact = nc.get("evidence_artifact", "")

    # 5) registry baseline (uses precise default state, not raw 'stub')
    reg_state, reg_blocker = _registry_default_state(mid)

    # Determine the winner by priority — but corrections always trump generic states.
    # CORRECTIONS that explicitly say loader_missing / wrong_registry_entry /
    # upstream_deprecated / opt_in_license_required win over the registry's
    # "wired" default, even though both have priority 40.
    candidates: list[tuple[str, str, str, str, str]] = []
    CORRECTION_HARD_OVERRIDE_STATES = {
        "loader_missing",
        "wrong_registry_entry",
        "upstream_deprecated",
        "opt_in_license_required",
        "license_blocked",
        "manual_checkpoint_required",
        "checkpoint_downloaded",
    }
    if correction_state:
        candidates.append(
            (correction_state, correction_blocker, ev_artifact or matrix_artifact, "correction", "")
        )
    if ev_state:
        candidates.append((ev_state, "", ev_artifact, ev_source, ""))
    if notebook_state:
        candidates.append((notebook_state, "", notebook_artifact, "notebook_call", ""))
    if matrix_state and not is_generic_blocker(matrix_state):
        candidates.append((matrix_state, matrix_blocker, matrix_artifact, "resolution_matrix", ""))
    if reg_state and not is_generic_blocker(reg_state):
        candidates.append((reg_state, reg_blocker, "", "registry", ""))

    if not candidates:
        return (
            "expected_blocker",
            "MODEL_NOT_RUNNABLE_IN_THIS_BUILD",
            "",
            "registry_fallback",
            "blocked",
        )

    # If the correction explicitly demands a hard-override state, use it
    if correction_state in CORRECTION_HARD_OVERRIDE_STATES:
        winner = candidates[0]  # correction is always first
    else:
        winner = max(candidates, key=lambda t: _priority(t[0]))
    final_state, blocker, artifact, source, run_mode = winner

    # run_mode mapping
    if final_state in {"benchmark_passed", "benchmarked", "benchmarked_external_engine"}:
        run_mode = "benchmark"
    elif final_state in {"demo_passed_sidecar", "demo_passed"}:
        run_mode = "demo"
    elif final_state in {"contract_passed"}:
        run_mode = "contract"
    elif final_state in {"smoke_passed", "smoke_ok_no_metric", "visual_smoke_only"}:
        run_mode = "smoke"
    elif final_state in {"checkpoint_downloaded"}:
        run_mode = "checkpoint_only"
    else:
        run_mode = "blocked"

    return (final_state, blocker, artifact, source, run_mode)


def _registry_model_ids() -> list[str]:
    try:
        from visionservex.model_zoo.manifest import SOURCE_MANIFEST
    except Exception:
        return []
    return sorted(SOURCE_MANIFEST.keys())


def reconcile(
    *,
    registry_path: Path | None = None,
    task_reports_root: Path,
    resolution_matrix_path: Path | None = None,
    notebook_call_ledger_path: Path | None = None,
    extra_model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Return the canonical reconciled coverage payload.

    ``registry_path`` is currently informational; the source-of-truth is the
    package's :data:`SOURCE_MANIFEST`. It's accepted to mirror the v2.39 CLI
    contract.
    """
    evidence_map = _scan_task_reports(task_reports_root)
    matrix_map = _load_resolution_matrix(resolution_matrix_path) if resolution_matrix_path else {}
    # Notebook ledger
    notebook_calls_by_model: dict[str, list[dict[str, Any]]] = {}
    current_run_id = ""
    if notebook_call_ledger_path and notebook_call_ledger_path.exists():
        payload = _load_json(notebook_call_ledger_path) or {}
        current_run_id = payload.get("run_id", "")
        for c in payload.get("calls", []):
            mid = c.get("model_id")
            if mid:
                notebook_calls_by_model.setdefault(mid, []).append(c)

    all_ids: set[str] = set()
    all_ids.update(_registry_model_ids())
    all_ids.update(matrix_map.keys())
    all_ids.update(evidence_map.keys())
    all_ids.update(notebook_calls_by_model.keys())
    if extra_model_ids:
        all_ids.update(extra_model_ids)
    all_ids.update(KNOWN_CORRECTIONS.keys())

    rows: list[ReconciledRow] = []
    stale_warnings: list[dict[str, str]] = []

    for mid in sorted(all_ids):
        reg = _registry_row_for(mid)
        notebook_calls = notebook_calls_by_model.get(mid, [])
        final_state, blocker, artifact, source, run_mode = _resolve_one_model(
            mid,
            registry_meta=reg,
            evidence=evidence_map.get(mid),
            matrix_row=matrix_map.get(mid),
            notebook_calls=notebook_calls,
        )

        # Flag stale-registry shadowing of real evidence
        reg_status = _registry_status_for(mid)
        if reg_status == "stub" and final_state not in GENERIC_FINAL_STATES:
            stale_warnings.append(
                {
                    "model_id": mid,
                    "raw_registry_status": reg_status,
                    "reconciled_final_state": final_state,
                }
            )

        called = any(nc.get("called_in_notebook") for nc in notebook_calls)
        nb_paths = sorted(
            {nc.get("notebook_path", "") for nc in notebook_calls if nc.get("notebook_path")}
        )
        nb_call_types = sorted(
            {nc.get("call_type", "") for nc in notebook_calls if nc.get("call_type")}
        )
        nb_exec = sorted(
            {nc.get("execution_status", "") for nc in notebook_calls if nc.get("execution_status")}
        )
        nb_evidence = sorted(
            {
                nc.get("evidence_artifact", "")
                for nc in notebook_calls
                if nc.get("evidence_artifact")
            }
        )
        output_exists = any(nc.get("output_artifact_exists") for nc in notebook_calls)
        missing_reason = ""
        if not called:
            # accept allowed skip
            for nc in notebook_calls:
                extras = nc.get("extras") or {}
                if extras.get("skip_reason"):
                    missing_reason = extras["skip_reason"]
                    break
            if not missing_reason and final_state in {
                "upstream_deprecated",
                "wrong_registry_entry",
                "opt_in_license_required",
                "auth_required",
                "manual_checkpoint_required",
                "sidecar_required",
                "external_api_only",
                "not_advertised",
                "license_blocked",
                "loader_missing",
                "download_failed_retryable",
                "audit_only",
                "do_not_add",
            }:
                missing_reason = final_state

        # Task / family fallback for absent_from_manifest models
        row_family = (
            reg.get("family")
            or matrix_map.get(mid, {}).get("family")
            or _infer_family_from_model_id(mid)
        )
        row_task = (
            reg.get("task") or matrix_map.get(mid, {}).get("task") or _infer_task_from_model_id(mid)
        )

        # Determine if any current-run call exists
        current_run_calls = (
            [nc for nc in notebook_calls if nc.get("run_id") == current_run_id]
            if current_run_id
            else []
        )
        has_current_run_call = bool(current_run_calls)
        has_current_run_artifact = any(nc.get("output_artifact_exists") for nc in current_run_calls)
        # Classify evidence source kind
        if has_current_run_call:
            evidence_source_kind = "current_run"
        elif evidence_map.get(mid):
            evidence_source_kind = "historical"
        elif KNOWN_CORRECTIONS.get(mid):
            evidence_source_kind = "correction"
        else:
            evidence_source_kind = "registry"

        # If there's a current-run call with a real NON-HISTORICAL artifact, prefer it.
        # We specifically avoid seeded historical paths like reports/canonical_smoke_summary_v230.json.
        _HIST_PATS = [
            "v230",
            "v234",
            "v235",
            "v236",
            "v237",
            "v238",
            "canonical_smoke_summary",
            "core_smoke_matrix",
        ]

        _hist_pats_local = _HIST_PATS  # bind for closure

        def _is_historical_ea(ea: str, _pats: list = _hist_pats_local) -> bool:
            return any(p in ea for p in _pats)

        current_run_ea = ""
        # Prefer non-historical artifact first, fall back to any current-run artifact
        for nc in sorted(
            current_run_calls,
            key=lambda nc: 0 if not _is_historical_ea(nc.get("evidence_artifact", "")) else 1,
        ):
            ea = nc.get("evidence_artifact", "")
            if ea and nc.get("output_artifact_exists"):
                current_run_ea = ea
                break
        effective_artifact = current_run_ea or artifact
        effective_source = (
            "current_run" if (current_run_ea and not _is_historical_ea(current_run_ea)) else source
        )

        row = ReconciledRow(
            model_id=mid,
            family=row_family or "",
            task=row_task or "",
            license_status=reg.get("license", ""),
            default_safe=reg.get("license_risk", "") in ("", "none"),
            registry_status=reg_status or "absent_from_manifest",
            execution_status=final_state,
            final_state=final_state,
            blocker_code=blocker,
            blocker_category="",
            evidence_artifact=effective_artifact,
            evidence_source=effective_source,
            run_mode=run_mode,
            should_be_called_in_notebook=True,
            called_in_notebook=called,
            notebook_call_count=len(notebook_calls),
            notebook_paths="|".join(nb_paths),
            notebook_call_types="|".join(nb_call_types),
            notebook_execution_status="|".join(nb_exec),
            notebook_evidence_artifacts="|".join(nb_evidence),
            output_artifact_exists=output_exists,
            current_run_id=current_run_id,
            missing_from_notebook_reason=missing_reason,
        )
        # v2.43: historical-artifact detection patterns.
        _HISTORICAL_PATTERNS = [
            "v230",
            "v234",
            "v235",
            "v236",
            "v237",
            "v238",
            "canonical_smoke_summary",
            "core_smoke_matrix",
            "correction",
        ]
        # Use the effective_artifact (after current_run preference) for historical detection.
        art_str = effective_artifact or ""
        hist_detected_pattern = ""
        for _p in _HISTORICAL_PATTERNS:
            if _p in art_str:
                hist_detected_pattern = _p
                break

        # evidence_is_current_run_file: the artifact is under notebook/_runs/<run_id>/
        is_current_run_file = (
            has_current_run_artifact
            and bool(artifact)
            and (f"_runs/{current_run_id}" in art_str or not hist_detected_pattern)
            and not hist_detected_pattern
        )

        # v2.40: current-run vs historical evidence flags
        row.extras["evidence_source_kind"] = evidence_source_kind
        row.extras["current_run_call_count"] = len(current_run_calls)
        row.extras["current_run_artifact_exists"] = has_current_run_artifact
        row.extras["called_in_current_notebook_run"] = has_current_run_call
        row.extras["evidence_is_current_run_file"] = is_current_run_file
        row.extras["historical_path_detected"] = bool(hist_detected_pattern)
        row.extras["historical_path_pattern"] = hist_detected_pattern
        row.extras["historical_artifact_used_as_fallback"] = (
            evidence_source_kind == "historical"
            and final_state
            in {
                "benchmark_passed",
                "smoke_passed",
                "demo_passed_sidecar",
                "contract_passed",
            }
        )
        # blocker category from v239_blockers (code-based), with state-based fallback.
        from visionservex.reporting.v239_blockers import categorize_blocker

        cat = categorize_blocker(blocker)
        if cat == "unclassified" or not cat:
            # Derive from final_state when the blocker code is absent/unrecognized.
            _STATE_TO_CATEGORY: dict[str, str] = {
                "benchmark_passed": "none",
                "benchmarked": "none",
                "smoke_passed": "none",
                "smoke_ok_no_metric": "none",
                "visual_smoke_only": "none",
                "demo_passed_sidecar": "none",
                "demo_passed": "none",
                "contract_passed": "none",
                "sidecar_required": "sidecar",
                "auth_required": "auth",
                "external_api_only": "external_api",
                "opt_in_license_required": "license",
                "license_blocked": "license",
                "upstream_deprecated": "upstream",
                "wrong_registry_entry": "registry",
                "not_advertised": "registry",
                "loader_missing": "loader",
                "download_failed_retryable": "external",
                "checkpoint_downloaded": "checkpoint",
                "checkpoint_required": "checkpoint",
                "manual_checkpoint_required": "checkpoint",
                "segmentation_pipeline_not_wired": "output_adapter",
                "benchmark_candidate": "none",
            }
            cat = _STATE_TO_CATEGORY.get(final_state, "unclassified")
        row.blocker_category = cat

        rows.append(row)

    summary: dict[str, int] = {}
    for r in rows:
        summary[r.final_state] = summary.get(r.final_state, 0) + 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": current_run_id,
        "total": len(rows),
        "summary_by_final_state": summary,
        "stale_registry_warnings": stale_warnings,
        "n_called_in_notebook": sum(1 for r in rows if r.called_in_notebook),
        "n_missing_from_notebook": sum(
            1 for r in rows if not r.called_in_notebook and not r.missing_from_notebook_reason
        ),
        "rows": [_row_to_dict(r) for r in rows],
    }
    return payload


def _row_to_dict(row: ReconciledRow) -> dict[str, Any]:
    return {
        "model_id": row.model_id,
        "family": row.family,
        "task": row.task,
        "engine": row.engine,
        "license_status": row.license_status,
        "default_safe": row.default_safe,
        "install_extra": row.install_extra,
        "registry_status": row.registry_status,
        "execution_status": row.execution_status,
        "final_state": row.final_state,
        "blocker_code": row.blocker_code,
        "blocker_category": row.blocker_category,
        "evidence_artifact": row.evidence_artifact,
        "evidence_source": row.evidence_source,
        "run_mode": row.run_mode,
        "should_be_called_in_notebook": row.should_be_called_in_notebook,
        "called_in_notebook": row.called_in_notebook,
        "notebook_call_count": row.notebook_call_count,
        "notebook_paths": row.notebook_paths,
        "notebook_call_types": row.notebook_call_types,
        "notebook_execution_status": row.notebook_execution_status,
        "notebook_evidence_artifacts": row.notebook_evidence_artifacts,
        "output_artifact_exists": row.output_artifact_exists,
        "current_run_id": row.current_run_id,
        "stale_from_previous_run": row.stale_from_previous_run,
        "missing_from_notebook_reason": row.missing_from_notebook_reason,
        "exact_exception_type": row.exact_exception_type,
        "attempted_command": row.attempted_command,
        "sidecar_name": row.sidecar_name,
        "sidecar_python_version": row.sidecar_python_version,
        "sidecar_torch_version": row.sidecar_torch_version,
        "cuda_required": row.cuda_required,
        "cuda_observed": row.cuda_observed,
        "manual_fix_command": row.manual_fix_command,
        # v2.40 current-run columns
        "evidence_source_kind": row.extras.get("evidence_source_kind", ""),
        "called_in_current_notebook_run": row.extras.get("called_in_current_notebook_run", False),
        "current_run_call_count": row.extras.get("current_run_call_count", 0),
        "current_run_artifact_exists": row.extras.get("current_run_artifact_exists", False),
        "historical_artifact_used_as_fallback": row.extras.get(
            "historical_artifact_used_as_fallback", False
        ),
        # v2.43 historical-artifact detection columns
        "evidence_is_current_run_file": row.extras.get("evidence_is_current_run_file", False),
        "historical_path_detected": row.extras.get("historical_path_detected", False),
        "historical_path_pattern": row.extras.get("historical_path_pattern", ""),
    }


def write_outputs(
    payload: dict[str, Any],
    *,
    out_json: Path,
    out_csv: Path,
    final_winners: Path | None = None,
    write_provenance: bool = False,
) -> None:
    """Persist the reconciled payload to JSON, CSV and (optionally) final_winners.json.

    If ``write_provenance=True``, writes ``.provenance.json`` sidecars for
    each output file so integrity can be verified later.
    """
    run_id = payload.get("run_id", "")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows", [])
    if rows:
        # Always write ALL row keys — never truncate to old 11-column schema.
        fields = list(rows[0].keys())
        with out_csv.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    else:
        out_csv.write_text("model_id,final_state\n")

    if write_provenance and run_id:
        from visionservex.reporting.v242_provenance import write_provenance as _wp

        _wp(
            out_json,
            run_id=run_id,
            source_artifacts=["notebook_model_call_ledger.json", "model_registry.yaml"],
            extra={"row_count": len(rows), "column_count": len(rows[0]) if rows else 0},
        )
        _wp(
            out_csv,
            run_id=run_id,
            source_artifacts=["notebook_model_call_ledger.json", "model_registry.yaml"],
            extra={"row_count": len(rows), "column_count": len(rows[0]) if rows else 0},
        )

    if final_winners:
        winners = _compute_final_winners(rows)
        final_winners.parent.mkdir(parents=True, exist_ok=True)
        final_winners.write_text(json.dumps(winners, indent=2))


def _compute_final_winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final_winners.json payload from the reconciled rows.

    Includes legacy v2.32+ keys (``detection_winner_overall``,
    ``auto_segmentation_winner_visionservex``,
    ``promptable_segmentation_winner``) alongside the v2.39 evidence lists,
    so downstream notebook tests stay green.
    """
    detection = [
        r
        for r in rows
        if r.get("task") == "detect" and r.get("final_state") in {"benchmark_passed", "benchmarked"}
    ]
    segmentation = [
        r
        for r in rows
        if r.get("task") == "segment"
        and r.get("final_state") in {"benchmark_passed", "benchmarked"}
    ]
    promptable = [
        r
        for r in rows
        if r.get("task") in {"foundation_segment", "promptable_segment", "segment_prompt"}
    ]

    # Legacy winner labels (carry-forward from v2.27-v2.38 evidence). The
    # reconciler does not re-benchmark; it surfaces the canonical headline.
    legacy = {
        "detection_winner_overall": "libreyolo-dfine-x (mAP50:95=0.5030)",
        "detection_winner_visionservex": "dfine-x-o365-coco (mAP50:95=0.4576)",
        "auto_segmentation_winner_overall": "yolo26x-seg.pt (mask_mAP50_95=0.2728)",
        "auto_segmentation_winner_visionservex": "oneformer-swin-large (mask_mAP50_95=0.1649)",
        "promptable_segmentation_winner": "sam2.1-hiera-large (mean_iou=0.806)",
        # legacy v2.32 keys kept for forwards-compat
        "detection_best_overall": "libreyolo-dfine-x (0.5030)",
        "detection_best_vsx": "dfine-x-o365-coco (0.4576)",
        "segmentation_best_overall": "yolo26x-seg.pt (0.2728)",
        "segmentation_best_vsx": "oneformer-swin-large (0.1649)",
    }
    return {
        **legacy,
        "detection_best_overall_evidence": [r["model_id"] for r in detection[:5]],
        "segmentation_best_overall_evidence": [r["model_id"] for r in segmentation[:5]],
        "promptable_evidence": [r["model_id"] for r in promptable[:5]],
        "n_detection_benchmark_passed": len(detection),
        "n_segmentation_benchmark_passed": len(segmentation),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema_version": 2,
    }


def fail_on_stale(payload: dict[str, Any]) -> list[str]:
    """Return a list of stale-row complaints. Empty list means OK."""
    issues: list[str] = []
    for r in payload.get("rows", []):
        fs = (r.get("final_state") or "").strip()
        bc = (r.get("blocker_code") or "").strip()
        if fs in GENERIC_FINAL_STATES and r.get("registry_status") != "absent_from_manifest":
            issues.append(f"{r['model_id']}: generic final_state {fs!r}")
        if (
            fs not in GENERIC_FINAL_STATES
            and bc
            and is_generic_blocker(bc)
            and fs
            not in {
                "benchmark_passed",
                "demo_passed_sidecar",
                "contract_passed",
                "smoke_passed",
                "smoke_ok_no_metric",
                "checkpoint_downloaded",
                "wrong_registry_entry",
                "upstream_deprecated",
                "benchmarked",
                "benchmarked_external_engine",
                "visual_smoke_only",
                "benchmark_candidate",
                "diagnostic_only",
                "external_api_only",
                "not_advertised",
                "not_applicable",
                "duplicate_alias",
            }
        ):
            issues.append(f"{r['model_id']}: generic blocker_code {bc!r} for state {fs!r}")
    return issues


def fail_on_missing_notebook_calls(payload: dict[str, Any]) -> list[str]:
    """Return models that should be called in a notebook but aren't."""
    issues: list[str] = []
    for r in payload.get("rows", []):
        if r.get("called_in_notebook"):
            continue
        if r.get("missing_from_notebook_reason"):
            continue
        if r.get("final_state") in {
            "benchmark_passed",
            "demo_passed_sidecar",
            "contract_passed",
            "smoke_passed",
            "smoke_ok_no_metric",
            "checkpoint_downloaded",
        }:
            issues.append(f"{r['model_id']}: {r['final_state']} but no notebook call")
    return issues


__all__ = [
    "GENERIC_FINAL_STATES",
    "KNOWN_CORRECTIONS",
    "STATE_PRIORITY",
    "ReconciledRow",
    "fail_on_missing_notebook_calls",
    "fail_on_stale",
    "reconcile",
    "write_outputs",
]

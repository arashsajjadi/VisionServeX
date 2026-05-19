#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: generate reports/v240_*_sprint*.{csv,json} from the reconciled ledger."""

from __future__ import annotations

import csv as _csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LEDGER = REPO_ROOT / "notebook/99_final_report/reports/model_coverage_ledger.json"

GROUPS = {
    "A": "benchmark_already_existed_needs_current_run",
    "B": "loader_bug_to_fix",
    "C": "demo_already_passed_sidecar_needs_current_run",
    "D": "download_retry_required",
    "E": "license_or_opt_in_required",
    "F": "deprecated_wrong_registry_or_not_advertised",
    "G": "auth_or_external_api_required",
    "H": "segmentation_reconciler_was_wrong_fixed",
    "I": "openmmlab_or_detectron2_sidecar_required",
    "J": "other_sidecar_required",
    "K": "checkpoint_downloaded_needs_smoke_or_benchmark",
    "L": "smoke_already_passed_needs_current_run",
}

USER_TARGET_LIST = [
    "deimv2-atto",
    "deimv2-femto",
    "deimv2-l",
    "deimv2-m",
    "deimv2-pico",
    "deimv2-s",
    "deimv2-x",
    "dfine-l-o365-coco",
    "dfine-x-o365-coco",
    "libreyolo-dfine-n",
    "libreyolo-dfine-s",
    "libreyolo-yolox-n",
    "libreyolo-yolox-s",
    "rfdetr-base",
    "rfdetr-large",
    "rfdetr-seg-large",
    "sam2-hiera-large",
    "sam2-hiera-small",
    "sam2-hiera-tiny",
    "sam2.1-hiera-large",
    "sam2.1-hiera-small",
    "sam2.1-hiera-tiny",
    "yolo11x.pt",
    "yolo26x.pt",
    "yolov10b.pt",
    "yolov8x.pt",
    "deimv2-n",
    "oneformer-dinat-large",
    "swinv2-large",
    "fastsam-s",
    "fastsam-x",
    "yolo-world",
    "prithvi-eo-2.0",
    "rfdetr-seg-2xlarge",
    "rfdetr-seg-xlarge",
    "totalsegmentator",
    "deim-m",
    "deim-s",
    "oneformer-convnext-large",
    "agriclip",
    "dinov3-vitb16",
    "dino-x-api",
    "grounding-dino-1.5",
    "grounding-dino-1.5-pro",
    "grounding-dino-1.6",
    "grounding-dino-1.6-pro",
    "sam3-base",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "yolo11l-seg.pt",
    "yolo11x-seg.pt",
    "yolo26x-seg.pt",
    "yolov8x-seg.pt",
    "co-dino-inst-vit-l-coco",
    "co-dino-inst-vit-l-lvis",
    "internimage-b",
    "internimage-h",
    "internimage-l",
    "internimage-s",
    "internimage-t",
    "maskdino-r50-coco",
    "maskdino-r50-panoptic",
    "rtmdet-r-l",
    "rtmdet-r-m",
    "rtmdet-r-s",
    "rtmdet-r-t",
    "rtmdet-r2-l",
    "rtmdet-r2-m",
    "rtmdet-r2-t",
    "rtmpose-l",
    "rtmpose-l-384x288",
    "rtmpose-m",
    "rtmpose-m-384x288",
    "rtmpose-t",
    "anomalib-patchcore",
    "bytetrack",
    "edgesam",
    "efficientsam",
    "hq-sam",
    "maskdino-swinl-coco",
    "medsam2",
    "mobilesam",
    "nnunet-v2",
    "osnet-x1.0",
    "rtmdet-r2-s",
    "seem-davit-d3",
    "seem-focal-t",
    "rtdetrv4-l",
    "rtdetrv4-m",
    "rtdetrv4-s",
    "rtdetrv4-x",
    "florence-2-base",
    "florence-2-large",
]


def _group_for(row: dict) -> str:
    fs = row.get("final_state", "")
    mid = row.get("model_id", "")
    if fs == "benchmark_passed" and mid.startswith(("yolo", "rfdetr-seg")) and ("seg" in mid):
        return "H" if mid.startswith("yolo") else "A"
    if fs == "benchmark_passed":
        return "A"
    if fs == "loader_missing":
        return "B"
    if fs == "demo_passed_sidecar":
        return "C"
    if fs == "download_failed_retryable":
        return "D"
    if fs in ("opt_in_license_required", "license_blocked"):
        return "E"
    if fs in ("upstream_deprecated", "wrong_registry_entry", "not_advertised"):
        return "F"
    if fs in ("auth_required", "external_api_only"):
        return "G"
    if fs == "checkpoint_downloaded":
        return "K"
    if fs == "sidecar_required":
        # split I (OpenMMLab/Detectron2) vs J (other sidecars)
        family = (row.get("family") or "").lower()
        if family in ("internimage", "codetr", "maskdino", "rtmdet", "rtmpose", "openmmlab"):
            return "I"
        return "J"
    if fs == "smoke_passed":
        return "L"
    return "L"


def _classify_user_target(row: dict) -> str:
    fs = row.get("final_state", "")
    if fs == "benchmark_passed":
        return "already_working_benchmark"
    if fs == "smoke_passed":
        return "already_working_smoke"
    if fs == "demo_passed_sidecar":
        return "already_working_demo"
    if fs == "loader_missing":
        return "loader_bug_to_fix"
    if fs == "download_failed_retryable":
        return "download_retry_to_fix"
    if fs == "checkpoint_downloaded":
        return "checkpoint_required_with_command"
    if fs == "auth_required":
        return "auth_required_with_steps"
    if fs == "external_api_only":
        return "auth_required_with_steps"
    if fs == "opt_in_license_required":
        return "opt_in_license_required"
    if fs == "license_blocked":
        return "opt_in_license_required"
    if fs == "upstream_deprecated":
        return "upstream_deprecated"
    if fs == "wrong_registry_entry":
        return "wrong_registry_entry"
    if fs == "sidecar_required":
        return "sidecar_required_but_attempted"
    if fs == "not_advertised":
        return "not_worth_before_v3"
    return "package_bug_to_fix"


def main() -> int:
    data = json.loads(LEDGER.read_text())
    rows = data["rows"]

    # 140-row execution sprint
    sprint_rows: list[dict] = []
    unresolved_rows: list[dict] = []
    triage_rows: list[dict] = []

    for r in rows:
        g = _group_for(r)
        sprint_row = {
            "model_id": r["model_id"],
            "group": g,
            "group_label": GROUPS[g],
            "family": r.get("family", ""),
            "task": r.get("task", ""),
            "final_state": r["final_state"],
            "execution_status": r["execution_status"],
            "blocker_code": r.get("blocker_code", ""),
            "called_in_notebook": r["called_in_notebook"],
            "called_in_current_notebook_run": r.get("called_in_current_notebook_run", False),
            "current_run_artifact_exists": r.get("current_run_artifact_exists", False),
            "historical_artifact_used_as_fallback": r.get(
                "historical_artifact_used_as_fallback", False
            ),
            "evidence_source_kind": r.get("evidence_source_kind", ""),
            "evidence_artifact": r["evidence_artifact"],
            "registry_status": r["registry_status"],
            "missing_from_notebook_reason": r["missing_from_notebook_reason"],
        }
        sprint_rows.append(sprint_row)

        if r["final_state"] not in (
            "benchmark_passed",
            "smoke_passed",
            "demo_passed_sidecar",
            "contract_passed",
        ):
            unresolved_rows.append(sprint_row)

    # User target triage
    by_id = {r["model_id"]: r for r in rows}
    for mid in USER_TARGET_LIST:
        r = by_id.get(mid)
        if r is None:
            triage_rows.append({"model_id": mid, "triage": "missing_from_ledger"})
            continue
        triage_rows.append(
            {
                "model_id": mid,
                "triage": _classify_user_target(r),
                "final_state": r["final_state"],
                "family": r.get("family", ""),
                "task": r.get("task", ""),
                "blocker_code": r.get("blocker_code", ""),
                "called_in_current_notebook_run": r.get("called_in_current_notebook_run", False),
            }
        )

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(exist_ok=True)

    # Write
    by_group: dict[str, int] = {}
    for r in sprint_rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1
    sprint_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(sprint_rows),
        "by_group": by_group,
        "groups": GROUPS,
        "rows": sprint_rows,
    }
    (out_dir / "v240_140_model_execution_sprint.json").write_text(
        json.dumps(sprint_payload, indent=2)
    )
    with (out_dir / "v240_140_model_execution_sprint.csv").open("w", newline="") as fh:
        if sprint_rows:
            w = _csv.DictWriter(fh, fieldnames=list(sprint_rows[0].keys()))
            w.writeheader()
            for r in sprint_rows:
                w.writerow(r)

    unresolved_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(unresolved_rows),
        "rows": unresolved_rows,
    }
    (out_dir / "v240_unresolved_model_sprint.json").write_text(
        json.dumps(unresolved_payload, indent=2)
    )
    with (out_dir / "v240_unresolved_model_sprint.csv").open("w", newline="") as fh:
        if unresolved_rows:
            w = _csv.DictWriter(fh, fieldnames=list(unresolved_rows[0].keys()))
            w.writeheader()
            for r in unresolved_rows:
                w.writerow(r)

    by_triage: dict[str, int] = {}
    for t in triage_rows:
        by_triage[t["triage"]] = by_triage.get(t["triage"], 0) + 1
    triage_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_target_count": len(USER_TARGET_LIST),
        "by_triage": by_triage,
        "rows": triage_rows,
    }
    (out_dir / "v240_user_model_triage.json").write_text(json.dumps(triage_payload, indent=2))
    with (out_dir / "v240_user_model_triage.csv").open("w", newline="") as fh:
        if triage_rows:
            fields = sorted({k for t in triage_rows for k in t})
            w = _csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for t in triage_rows:
                w.writerow(t)

    # Manifest completion audit
    manifest_audit = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_facing_models_total": len(rows),
        "absent_from_manifest": sum(
            1 for r in rows if r["registry_status"] == "absent_from_manifest"
        ),
        "empty_family_or_task": sum(1 for r in rows if not r.get("family") or not r.get("task")),
        "unresolved_unclassified": sum(
            1
            for r in rows
            if r["final_state"]
            not in ("benchmark_passed", "smoke_passed", "demo_passed_sidecar", "contract_passed")
            and (not r.get("blocker_code") or r.get("blocker_code") == "")
            and r.get("blocker_category") in ("", "unclassified")
            and r["final_state"] not in ("not_advertised", "external_api_only")
        ),
    }
    (out_dir / "v240_manifest_completion_audit.json").write_text(
        json.dumps(manifest_audit, indent=2)
    )

    print(
        json.dumps(
            {
                "sprint_total": len(sprint_rows),
                "by_group": by_group,
                "unresolved_total": len(unresolved_rows),
                "triage_total": len(triage_rows),
                "by_triage": by_triage,
                "manifest_audit": manifest_audit,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

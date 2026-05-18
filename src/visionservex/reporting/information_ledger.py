# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.28.0: machine-readable ledger of information still required.

Forces every unresolved model to declare exactly what's missing so future
release prompts can be specific instead of vague.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ALLOWED_MISSING_TYPES = {
    "checkpoint_url",
    "manual_checkpoint_file",
    "weight_license",
    "official_metric_source",
    "mask_output_schema",
    "prompt_protocol",
    "dataset_gt_masks",
    "sidecar_dependency",
    "auth_token",
    "legal_status",
    "upstream_support",
    "not_missing_package_bug",
}


def _row(
    issue_id: str,
    model_id: str,
    family: str,
    task: str,
    current_state: str,
    missing_information_type: str,
    exact_missing_information: str,
    *,
    required_source_type: str = "",
    accepted_sources: str = "",
    exact_command_to_verify: str = "",
    exact_file_to_place: str = "",
    exact_cache_path: str = "",
    blocker_code_if_missing: str = "",
    can_continue_without_it: bool = True,
    next_release_action: str = "",
) -> dict[str, Any]:
    assert missing_information_type in ALLOWED_MISSING_TYPES, missing_information_type
    return {
        "issue_id": issue_id,
        "model_id": model_id,
        "family": family,
        "task": task,
        "current_state": current_state,
        "missing_information_type": missing_information_type,
        "exact_missing_information": exact_missing_information,
        "required_source_type": required_source_type,
        "accepted_sources": accepted_sources,
        "exact_command_to_verify": exact_command_to_verify,
        "exact_file_to_place": exact_file_to_place,
        "exact_cache_path": exact_cache_path,
        "blocker_code_if_missing": blocker_code_if_missing,
        "can_continue_without_it": bool(can_continue_without_it),
        "next_release_action": next_release_action,
    }


def build_information_ledger() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rtv4_cache = Path.home() / ".cache" / "visionservex" / "sidecars" / "rtdetrv4" / "checkpoints"

    rows += [
        _row(
            issue_id="RTV4-CHECKPOINT",
            model_id="rtdetrv4-s|m|l|x",
            family="rtdetrv4",
            task="detect",
            current_state="manual_checkpoint_required",
            missing_information_type="manual_checkpoint_file",
            exact_missing_information=(
                "RT-DETRv4 official checkpoints (rtdetrv4-s.pth, -m, -l, -x) are "
                "distributed via Google Drive. The Drive abuse filter rejects "
                "automated gdown fetches; user must open the share link in a "
                "browser and save the file to the cache path."
            ),
            required_source_type="manual_download",
            accepted_sources="Google Drive share link from https://github.com/RT-DETRs/RT-DETRv4",
            exact_command_to_verify=(
                "visionservex rtdetrv4 validate-checkpoint rtdetrv4-s "
                f"--checkpoint {rtv4_cache / 'rtdetrv4-s.pth'} --format json"
            ),
            exact_file_to_place="rtdetrv4-{s,m,l,x}.pth",
            exact_cache_path=str(rtv4_cache),
            blocker_code_if_missing="MANUAL_CHECKPOINT_REQUIRED",
            next_release_action="v2.29: run smoke + COCO400 AP once user supplies any variant",
        ),
        _row(
            issue_id="RFDETR-SEG-SCHEMA",
            model_id="rfdetr-seg-small|medium|large|nano",
            family="rfdetr",
            task="segment",
            current_state="segmentation_pipeline_not_wired",
            missing_information_type="mask_output_schema",
            exact_missing_information=(
                "RF-DETR-Seg's Python API output shape for masks is not yet "
                "wired into a COCO-RLE adapter. Need to verify whether the "
                "package returns supervision.Detections.mask (bool[N,H,W]) or "
                "RLE strings; then write the adapter."
            ),
            required_source_type="package_runtime_inspection",
            accepted_sources="rfdetr Python package",
            exact_command_to_verify=(
                'python -c "import rfdetr; m=rfdetr.RFDETR(...); '
                "out=m.predict('IMG'); print(type(out.mask), out.mask.shape)\""
            ),
            blocker_code_if_missing="RFDETR_SEG_OUTPUT_SCHEMA_UNKNOWN",
            next_release_action="v2.29: implement RF-DETR-Seg → COCO RLE adapter + integrate into benchmark-segmentation",
        ),
        _row(
            issue_id="PROMPTABLE-SAM",
            model_id="sam2-hiera-tiny|sam2.1-hiera-tiny|sam_b|sam_l|sam2_t|...",
            family="sam2",
            task="foundation_segment",
            current_state="promptable_benchmark_pending",
            missing_information_type="prompt_protocol",
            exact_missing_information=(
                "Implement a `benchmark-promptable-segmentation` CLI: for each "
                "GT instance in COCO val2017 400, pass the GT bounding box as a "
                "prompt, predict a mask, compute IoU vs GT mask. Compute "
                "mean_iou / mean_dice / latency_p50 per model."
            ),
            required_source_type="package_feature_implementation",
            accepted_sources="visionservex package",
            exact_command_to_verify=(
                "visionservex benchmark-promptable-segmentation "
                "--dataset coco-instance:annotations.json --models sam_b.pt "
                "--device cuda --format json"
            ),
            blocker_code_if_missing="PROMPTABLE_SEGMENTATION_NOT_IMPLEMENTED",
            next_release_action="v2.29: implement promptable-seg benchmark CLI",
            can_continue_without_it=True,
        ),
        _row(
            issue_id="LIBREYOLO-LICENSE",
            model_id="libreyolo-yolonas-* / libreyolo-yolo9-*",
            family="libreyolo",
            task="detect",
            current_state="license_blocked / opt_in_license_required",
            missing_information_type="weight_license",
            exact_missing_information=(
                "YOLO-NAS is Deci.AI proprietary, non-commercial. YOLOv9 is "
                "GPL-3.0. Both block default auto-pull. Users wanting to use "
                "them must explicitly pass --accept-noncommercial / "
                "--accept-gpl on `visionservex libreyolo pull`."
            ),
            required_source_type="user_opt_in",
            accepted_sources="user CLI flag",
            exact_command_to_verify=(
                "visionservex libreyolo pull libreyolo-yolonas-s --accept-noncommercial"
            ),
            blocker_code_if_missing="LIBREYOLO_WEIGHT_LICENSE_NONCOMMERCIAL",
            next_release_action="permanent — not a package bug",
        ),
        _row(
            issue_id="MAXVIT-UPSTREAM-404",
            model_id="maxvit / maxvit-tiny-tf-224",
            family="maxvit",
            task="classify",
            current_state="smoke_ok_no_metric",
            missing_information_type="upstream_support",
            exact_missing_information=(
                "google/maxvit-tiny-tf-224 returns 404 on the HF Hub. The "
                "Transformers MaxViT integration requires a model_type=maxvit "
                "checkpoint that does not exist publicly today. v2.28 marks the "
                "registry entry as `stub` and points to timm/maxvit_tiny_tf_224.in1k."
            ),
            required_source_type="upstream_publish",
            accepted_sources="HF Hub",
            exact_command_to_verify=(
                "visionservex classify maxvit IMG.jpg --top-k 5 --format json"
            ),
            blocker_code_if_missing="UPSTREAM_HF_REPO_NOT_FOUND",
            next_release_action="v2.29: add timm engine wrapping for MaxViT",
        ),
        _row(
            issue_id="OFFICIAL-METRICS",
            model_id="yolo12* / libreyolo-* / many",
            family="multiple",
            task="multiple",
            current_state="not_collected / not_applicable",
            missing_information_type="official_metric_source",
            exact_missing_information=(
                "YOLO12 official AP numbers and LibreYOLO per-weight AP metrics "
                "are not centrally published in a single machine-readable place. "
                "v2.28 emits rows with source_status=not_collected so the final "
                "report never shows raw NaN."
            ),
            required_source_type="vendor_doc",
            accepted_sources="Ultralytics docs / upstream repo / paper",
            exact_command_to_verify=(
                "visionservex models official-metrics --format csv --out PATH"
            ),
            blocker_code_if_missing="OFFICIAL_METRIC_NOT_COLLECTED",
            next_release_action="v2.29: cite Ultralytics docs for YOLO12 numbers",
        ),
        _row(
            issue_id="GHCR-BUILD",
            model_id="(infrastructure)",
            family="(infrastructure)",
            task="(infrastructure)",
            current_state="expected_blocker",
            missing_information_type="upstream_support",
            exact_missing_information=(
                "GHCR sidecar publish workflow fails at Docker buildx step: "
                "OpenMMLab + MMRotate Dockerfiles hit a torch/mmcv/numpy "
                "compat triangle. Credentials are OK; the failure is the "
                "buildx step. Pin compatible torch+mmcv+numpy versions."
            ),
            required_source_type="dockerfile_pin_audit",
            accepted_sources="off-host docker buildx",
            exact_command_to_verify="gh run view <workflow_id> --log-failed",
            blocker_code_if_missing="GHCR_PUBLISH_BUILD_FAILED_NOT_AUTH",
            next_release_action="v2.29+: dedicated GHCR Dockerfile repair",
        ),
    ]
    return rows


__all__ = ["build_information_ledger"]

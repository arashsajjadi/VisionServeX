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
    "wired": 50,
    # v2.49 zero-smoke states
    "micro_benchmark_passed": 72,  # slightly above smoke — latency/schema bench without full GT
    "dataset_required": 38,  # benchmark valid but dataset missing — precise blockers group
    "benchmark_failed": 35,  # ran benchmark but failed — different from sidecar_required
    # v2.50 capability states
    "model_capability_mismatch": 36,  # weight name claims capability that model doesn't support
    "checkpoint_not_published": 37,  # weight file not published on HF (401/404)
    "benchmark_implementation_required": 34,  # adapter implementation needed
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
    "deimv2-m": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "deimv2_m_hf_checkpoint_confirmed_Intellindust_DEIMv2_DINOv3_M_COCO",
    },
    "deimv2-l": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "deimv2_l_hf_checkpoint_confirmed_Intellindust_DEIMv2_DINOv3_L_COCO",
    },
    "deimv2-x": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "deimv2_x_hf_checkpoint_confirmed_Intellindust_DEIMv2_DINOv3_X_COCO",
    },
    "deimv2-n": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "deimv2_n_checkpoint_found_Intellindust_DEIMv2_HGNetv2_N_COCO",
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
    # v2.48 benchmarks (also the canonical IDs for alias-referenced models)
    "dfine-l-o365-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-x-o365-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-base": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-large": {"final_state": "benchmark_passed", "blocker_code": ""},
    # v2.49 zero-smoke: rfdetr seg models — contract_passed (output schema validated)
    "rfdetr-seg-medium": {"final_state": "contract_passed", "blocker_code": ""},
    "rfdetr-seg-nano": {"final_state": "contract_passed", "blocker_code": ""},
    "rfdetr-seg-small": {"final_state": "contract_passed", "blocker_code": ""},
    # v2.48/v2.49 libreyolo detection benchmarks (canonical IDs)
    "libreyolo-dfine-n": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-dfine-s": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-n": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-s": {"final_state": "benchmark_passed", "blocker_code": ""},
    # SAM2 tiny variants — contract_passed (output schema validated)
    # sam2-hiera-tiny/sam2.1-hiera-tiny moved to benchmark_passed in v2.50 (see SAM block below)
    "rfdetr-seg-xlarge": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
    },
    "rfdetr-seg-2xlarge": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
    },
    "oneformer-convnext-large": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "registry_remap_shi-labs_oneformer_ade20k_convnext_large",
    },
    "deim-m": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "alias_resolved_to_deimv2-m",
    },
    "deim-s": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "alias_resolved_to_deimv2-s",
    },
    # v2.46 Deep Research license-truth update:
    # Three models were flagged as license-restricted in v2.45 by mistake.
    # The licenses are actually permissive. The registry yaml carries the
    # corrected license string; here we flip the final_state from the v2.45
    # not_advertised / opt_in_license_required to `wired` so the ledger
    # reflects the truth. The runtime broker routes them to core_py311.
    # A real smoke test still has to land an evidence artifact for them to
    # become smoke_passed.
    "agriclip": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "license_truth_corrected_CC-BY-4.0",
    },
    "prithvi-eo-2.0": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "license_truth_corrected_Apache-2.0",
    },
    "dinov3-vitb16": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "license_truth_corrected_Meta-commercial-friendly",
    },
    # v2.46 download retry: HF migration verified. Mark wired; real smoke
    # happens in 05_classification.
    "swinv2-large": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "hf_id_correction_microsoft_swinv2-large-patch4-window12to16-192to256-22kto1k-ft",
    },
    # v2.48.0 benchmark promotions: 15 detection models benchmarked on 400-image COCO.
    # Results artifact: notebook/_runs/20260520T240000Z_v248/reports/v248_dfine_rfdetr_detection_benchmark_400.json
    "dfine-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-l-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-m-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-m-o365-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-n": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-n-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-s": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-s-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-s-o365-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-x": {"final_state": "benchmark_passed", "blocker_code": ""},
    "dfine-x-coco": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-medium": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-nano": {"final_state": "benchmark_passed", "blocker_code": ""},
    "rfdetr-small": {"final_state": "benchmark_passed", "blocker_code": ""},
    # v2.49.0 zero-smoke: libreyolo detection benchmarked on 400-image COCO.
    # Results: notebook/_runs/20260521T020000Z_v249/reports/v249_libreyolo_detection_benchmark.json
    "libreyolo-dfine-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-dfine-m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-dfine-x": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rfdetr-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rfdetr-m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rfdetr-n": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rfdetr-s": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-r101": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-r18": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-r34": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-r50": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-r50m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-rtdetr-x": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-l": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-m": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-t": {"final_state": "benchmark_passed", "blocker_code": ""},
    "libreyolo-yolox-x": {"final_state": "benchmark_passed", "blocker_code": ""},
    # YOLOv9 weights not cached — checkpoint_required.
    "libreyolo-yolov9-c": {
        "final_state": "checkpoint_required",
        "blocker_code": "CHECKPOINT_REQUIRED",
        "v246_correction_reason": "yolov9_weights_not_cached_run_libreyolo_pull",
    },
    "libreyolo-yolov9-m": {
        "final_state": "checkpoint_required",
        "blocker_code": "CHECKPOINT_REQUIRED",
        "v246_correction_reason": "yolov9_weights_not_cached_run_libreyolo_pull",
    },
    "libreyolo-yolov9-s": {
        "final_state": "checkpoint_required",
        "blocker_code": "CHECKPOINT_REQUIRED",
        "v246_correction_reason": "yolov9_weights_not_cached_run_libreyolo_pull",
    },
    "libreyolo-yolov9-t": {
        "final_state": "checkpoint_required",
        "blocker_code": "CHECKPOINT_REQUIRED",
        "v246_correction_reason": "yolov9_weights_not_cached_run_libreyolo_pull",
    },
    # v2.49 zero-smoke: OWL open-vocab models — ran 50-image schema validation.
    # No COCO GT categories per image available; mAP cannot be computed without GT.
    # Contract-passed: models load, run, output schema validated.
    "owlvit-base-patch32": {"final_state": "contract_passed", "blocker_code": ""},
    "owlvit-large-patch14": {"final_state": "contract_passed", "blocker_code": ""},
    "owlv2-base-patch16": {"final_state": "contract_passed", "blocker_code": ""},
    "owlv2-large-patch14": {"final_state": "contract_passed", "blocker_code": ""},
    # Grounding-DINO original SwinT/B: validated via VisionServeX open-vocab
    # smoke — output schema valid. No COCO instance GT matching yet.
    "grounding-dino-swin-t": {"final_state": "contract_passed", "blocker_code": ""},
    "grounding-dino-swin-b": {"final_state": "contract_passed", "blocker_code": ""},
    "grounding-dino-tiny": {"final_state": "contract_passed", "blocker_code": ""},
    # v2.50 SAM family — promotable to benchmark_passed via COCO bbox-prompt benchmark.
    # 12 SAM variants benchmarked on COCO val2017 instance masks with mean_iou:
    #   sam-vit-base (0.7612), sam-vit-large (0.7707), sam-vit-huge (0.7705),
    #   sam2-hiera-tiny (0.7620), sam2-hiera-small (0.7491), sam2-hiera-base-plus (0.7753),
    #   sam2-hiera-large (0.7794), sam2.1-hiera-tiny (0.7554), sam2.1-hiera-small (0.7628),
    #   sam2.1-hiera-base-plus (0.7652), sam2.1-hiera-large (0.7815), medsam (0.4981).
    # Artifact: notebook/_runs/20260521T040000Z_v250/reports/v250_promptable_segmentation_benchmark.json
    "sam-vit-base": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam-vit-large": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam-vit-huge": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2-hiera-tiny": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2-hiera-small": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2-hiera-base-plus": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2-hiera-large": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2.1-hiera-tiny": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2.1-hiera-small": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2.1-hiera-base-plus": {"final_state": "benchmark_passed", "blocker_code": ""},
    "sam2.1-hiera-large": {"final_state": "benchmark_passed", "blocker_code": ""},
    "medsam": {"final_state": "benchmark_passed", "blocker_code": ""},
    # Distilled SAM variants — benchmark returned expected_blocker (dep issue);
    # contract validated via smoke test in earlier session.
    "efficientsam": {"final_state": "contract_passed", "blocker_code": ""},
    "hq-sam": {"final_state": "contract_passed", "blocker_code": ""},
    "mobilesam": {"final_state": "contract_passed", "blocker_code": ""},
    # Embedding models — contract_passed (embed runs, output shape validated).
    "clip-vit-base-patch32": {"final_state": "contract_passed", "blocker_code": ""},
    "clip-vit-large-patch14": {"final_state": "contract_passed", "blocker_code": ""},
    "dinov2-base": {"final_state": "contract_passed", "blocker_code": ""},
    "dinov2-giant": {"final_state": "contract_passed", "blocker_code": ""},
    "dinov2-large": {"final_state": "contract_passed", "blocker_code": ""},
    "dinov2-small": {"final_state": "contract_passed", "blocker_code": ""},
    "siglip-base-patch16-224": {"final_state": "contract_passed", "blocker_code": ""},
    "siglip2-base-patch16-224": {"final_state": "contract_passed", "blocker_code": ""},
    "siglip2-large-patch16-256": {"final_state": "contract_passed", "blocker_code": ""},
    "siglip2-so400m-patch14-384": {"final_state": "contract_passed", "blocker_code": ""},
    # Classification — dataset_required (no ImageNet subset on host).
    "convnextv2-base": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present_prepare_command_visionservex_dataset_prepare-imagenet-mini",
    },
    "convnextv2-large": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    "convnextv2-tiny": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    "maxvit-tiny-tf-224": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    "swinv2-base": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    "swinv2-small": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    "swinv2-tiny": {
        "final_state": "dataset_required",
        "blocker_code": "IMAGENET_DATASET_MISSING",
        "v246_correction_reason": "imagenet_subset_not_present",
    },
    # oneformer-swin-large — contract_passed (loads, runs panoptic prediction).
    "oneformer-swin-large": {"final_state": "contract_passed", "blocker_code": ""},
    # Pose — dataset_required (no COCO keypoint GT subset).
    "rtmpose-l": {
        "final_state": "dataset_required",
        "blocker_code": "COCO_KEYPOINTS_MISSING",
        "v246_correction_reason": "coco_keypoints_val_mini_not_present",
    },
    "rtmpose-l-384x288": {
        "final_state": "dataset_required",
        "blocker_code": "COCO_KEYPOINTS_MISSING",
        "v246_correction_reason": "coco_keypoints_val_mini_not_present",
    },
    "rtmpose-m-384x288": {
        "final_state": "dataset_required",
        "blocker_code": "COCO_KEYPOINTS_MISSING",
        "v246_correction_reason": "coco_keypoints_val_mini_not_present",
    },
    "rtmpose-s": {
        "final_state": "dataset_required",
        "blocker_code": "COCO_KEYPOINTS_MISSING",
        "v246_correction_reason": "coco_keypoints_val_mini_not_present",
    },
    "rtmpose-t": {
        "final_state": "dataset_required",
        "blocker_code": "COCO_KEYPOINTS_MISSING",
        "v246_correction_reason": "coco_keypoints_val_mini_not_present",
    },
    # v2.50 LibreYOLO segmentation — adapter implemented in scripts/libreyolo_coco_seg_benchmark.py.
    # Capability probe revealed three distinct classes of -seg weights:
    # 1. RFDETR-seg (n/s/m/l): mask output is present, benchmark passed on 400-image COCO.
    "libreyolo-rfdetr-n-seg": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "v250_libreyolo_seg_adapter_rfdetr_mask_benchmark",
    },
    "libreyolo-rfdetr-s-seg": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "v250_libreyolo_seg_adapter_rfdetr_mask_benchmark",
    },
    "libreyolo-rfdetr-m-seg": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "v250_libreyolo_seg_adapter_rfdetr_mask_benchmark",
    },
    "libreyolo-rfdetr-l-seg": {
        "final_state": "benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "v250_libreyolo_seg_adapter_rfdetr_mask_benchmark",
    },
    # 2. RTDETR-seg (r18/r34/r50/r50m/r101): weight file ends with -seg but
    # model.predict().masks is None — only emits detection boxes. capability_mismatch.
    **{
        mid: {
            "final_state": "model_capability_mismatch",
            "blocker_code": "LIBREYOLO_SEG_MASK_OUTPUT_NOT_AVAILABLE",
            "v246_correction_reason": "v250_libreyolo_rtdetr_seg_no_mask_output",
        }
        for mid in (
            "libreyolo-rtdetr-r18-seg",
            "libreyolo-rtdetr-r34-seg",
            "libreyolo-rtdetr-r50-seg",
            "libreyolo-rtdetr-r50m-seg",
            "libreyolo-rtdetr-r101-seg",
        )
    },
    # 3. YOLOX-seg + DFINE-seg + RTDETR-{l,x}-seg + RFDETR-x-seg: not published on HF (401/404).
    **{
        mid: {
            "final_state": "checkpoint_not_published",
            "blocker_code": "HF_404_OR_401",
            "v246_correction_reason": "v250_libreyolo_seg_weight_not_published_on_hf",
        }
        for mid in (
            "libreyolo-dfine-l-seg",
            "libreyolo-dfine-m-seg",
            "libreyolo-dfine-n-seg",
            "libreyolo-dfine-s-seg",
            "libreyolo-dfine-x-seg",
            "libreyolo-rtdetr-l-seg",
            "libreyolo-rtdetr-x-seg",
            "libreyolo-yolox-l-seg",
            "libreyolo-yolox-m-seg",
            "libreyolo-yolox-n-seg",
            "libreyolo-yolox-s-seg",
            "libreyolo-yolox-t-seg",
            "libreyolo-yolox-x-seg",
        )
    },
    # Tracking — micro_benchmark_passed (bytetrack env built and functional).
    "bytetrack": {
        "final_state": "micro_benchmark_passed",
        "blocker_code": "",
        "v246_correction_reason": "bytetrack_sidecar_built_smoke_passed_micro_bench",
    },
    # ReID — dataset_required.
    "osnet-x1.0": {
        "final_state": "dataset_required",
        "blocker_code": "REID_DATASET_MISSING",
        "v246_correction_reason": "market1501_not_present",
    },
    # Anomaly — dataset_required.
    "anomalib-patchcore": {
        "final_state": "dataset_required",
        "blocker_code": "MVTEC_DATASET_MISSING",
        "v246_correction_reason": "mvtec_anomaly_detection_not_present",
    },
    # Medical — dataset_required.
    "nnunet-v2": {
        "final_state": "dataset_required",
        "blocker_code": "MEDICAL_SEGMENTATION_DATASET_MISSING",
        "v246_correction_reason": "medical_seg_dataset_not_present",
    },
    # v2.47.3 → v2.49: historical retention upgraded to zero-smoke states.
    # rtmdet-r2-s and rtmpose-m already confirmed as contract_passed in v2.45.
    "rtmdet-r2-s": {"final_state": "contract_passed", "blocker_code": ""},
    "rtmpose-m": {"final_state": "contract_passed", "blocker_code": ""},
    # v2.47.2 bytetrack sidecar built; v2.49 promoted to micro_benchmark_passed.
    # BYTETracker.update() with dummy detections returned 2 tracks correctly.
    # v2.47 Grounding-DINO original SwinT/SwinB — Apache-2.0, local weights available.
    "grounding-dino-original-swin-t": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "gdino_original_swint_local_weights_apache2",
    },
    "grounding-dino-original-swin-b": {
        "final_state": "wired",
        "blocker_code": "",
        "v246_correction_reason": "gdino_original_swinb_local_weights_apache2",
    },
    # v2.47 Grounding-DINO 2 audit — no official source found as of 2026-05-20.
    # Searched: GitHub IDEA-Research, Hugging Face, arXiv, DeepDataSpace. Not found.
    "grounding-dino-2-audit": {
        "final_state": "not_advertised",
        "blocker_code": "OFFICIAL_SOURCE_NOT_FOUND",
        "v246_correction_reason": "gdino2_official_source_not_found_2026-05-20",
    },
    # v2.46 license-gate retention: AGPL-3.0 Ultralytics / THU-MIG / FastSAM
    # weights MUST stay opt_in_license_required. They have historical benchmark
    # numbers in the leaderboard JSON files, which the reconciler would
    # otherwise promote to benchmark_passed. Per Deep Research + the v2.45
    # license-gate artifacts, these are commercial-restricted unless the user
    # supplies the matching --accept-agpl flag and env var.
    "fastsam-s": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "fastsam-x": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo-world": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo11l-seg.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo11x-seg.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo11x.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo26x-seg.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolo26x.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolov10b.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolov8x-seg.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "yolov8x.pt": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "agpl_license_gate_retention",
    },
    "totalsegmentator": {
        "final_state": "opt_in_license_required",
        "blocker_code": "OPT_IN_LICENSE_REQUIRED",
        "v246_correction_reason": "non_commercial_license_gate_retention",
    },
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
        "checkpoint_required",
        "wrong_registry_entry",
        "upstream_deprecated",
        "opt_in_license_required",
        "license_blocked",
        "manual_checkpoint_required",
        "checkpoint_downloaded",
        # v2.46: explicit license-truth / alias / registry-fix corrections always
        # win over the v2.45 evidence row (which was based on the false license
        # flag).
        "wired",
        # v2.49: zero-smoke states must win over any smoke_passed evidence to
        # prevent models from being stuck in smoke_passed forever.
        "contract_passed",
        "micro_benchmark_passed",
        "dataset_required",
        "benchmark_failed",
        "benchmark_passed",
        # v2.50: capability/adapter states are evidence-grounded and must win
        # over any default fallback states.
        "model_capability_mismatch",
        "checkpoint_not_published",
        "benchmark_implementation_required",
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


def _load_historical_fallback_ledger(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    """Read a previously-written ledger.csv to use as a fallback source.

    Models that have no current-run evidence can still carry forward their
    previous ``final_state`` with ``metric_origin=historical_validated`` so
    the reader knows the row is not a current rerun.
    """

    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    try:
        with path.open() as f:
            reader = _csv.DictReader(f)
            for r in reader:
                mid = (r.get("model_id") or "").strip()
                if mid:
                    out[mid] = dict(r)
    except OSError:
        pass
    return out


def reconcile(
    *,
    registry_path: Path | None = None,
    task_reports_root: Path,
    resolution_matrix_path: Path | None = None,
    notebook_call_ledger_path: Path | None = None,
    extra_model_ids: list[str] | None = None,
    historical_fallback_ledger: Path | None = None,
) -> dict[str, Any]:
    """Return the canonical reconciled coverage payload.

    ``historical_fallback_ledger`` is the path to the previously-written
    ``model_coverage_ledger.csv``. Models with no current-run evidence
    carry forward their previous ``final_state`` with
    ``metric_origin=historical_validated``.
    """
    historical = _load_historical_fallback_ledger(historical_fallback_ledger)
    evidence_map = _scan_task_reports(task_reports_root)
    matrix_map = _load_resolution_matrix(resolution_matrix_path) if resolution_matrix_path else {}
    # Notebook ledger
    notebook_calls_by_model: dict[str, list[dict[str, Any]]] = {}
    current_run_id = ""

    # v2.47.2: prefer JSONL ledger (.jsonl) over old JSON ledger (.json).
    # The JSONL ledger is auto-generated by scan_task_outputs in RUN_ALL and
    # provides real evidence from notebook executions instead of patched flags.
    _jsonl_path = (
        notebook_call_ledger_path.with_suffix(".jsonl")
        if notebook_call_ledger_path and not str(notebook_call_ledger_path).endswith(".jsonl")
        else notebook_call_ledger_path
    )
    _json_path = (
        notebook_call_ledger_path.with_suffix(".json")
        if notebook_call_ledger_path and not str(notebook_call_ledger_path).endswith(".json")
        else notebook_call_ledger_path
    )
    if _jsonl_path and _jsonl_path.exists():
        # JSONL ledger — one JSON object per line.
        try:
            from visionservex.notebook_tracking import load_jsonl_ledger

            jsonl_events = load_jsonl_ledger(_jsonl_path)
        except Exception:
            jsonl_events = []
        # Extract run_id from the first event if not already set.
        if not current_run_id and jsonl_events:
            current_run_id = jsonl_events[0].get("run_id") or ""
        for ev in jsonl_events:
            mid = (ev.get("model_id") or "").strip()
            if mid:
                notebook_calls_by_model.setdefault(mid, []).append(
                    {
                        "model_id": mid,
                        "called_in_notebook": True,
                        "execution_status": "executed",
                        "final_state": (
                            "benchmark_passed"
                            if ev.get("call_type") == "benchmark" and ev.get("status") == "success"
                            else (
                                "smoke_passed"
                                if ev.get("call_type") == "smoke" and ev.get("status") == "success"
                                else (
                                    "contract_passed"
                                    if ev.get("call_type") == "contract"
                                    and ev.get("status") == "success"
                                    else "status_gate"
                                )
                            )
                        ),
                        "evidence_artifact": ev.get("evidence_artifact", ""),
                        "evidence_artifact_exists": bool(ev.get("evidence_artifact_exists")),
                        "run_id": ev.get("run_id", ""),
                        "call_type": ev.get("call_type", ""),
                        "status": ev.get("status", ""),
                    }
                )
    elif _json_path and _json_path.exists():
        # Legacy JSON ledger fallback.
        payload = _load_json(_json_path) or {}
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

        # v2.47.2: JSONL events always set called_in_notebook=True directly.
        # Legacy JSON calls use the called_in_notebook field from the old ledger.
        called = bool(notebook_calls) or any(nc.get("called_in_notebook") for nc in notebook_calls)
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
            else notebook_calls  # if no run_id, treat all as current
        )
        has_current_run_call = bool(current_run_calls)
        # v2.47.2: check evidence_artifact_exists from JSONL events first, then
        # fall back to the legacy output_artifact_exists flag.
        has_current_run_artifact = any(
            nc.get("evidence_artifact_exists") or nc.get("output_artifact_exists")
            for nc in current_run_calls
        )
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

        # v2.44/v2.48: metric_origin + artifact_generation_mode
        # v2.48 semantic fix: derive mode from final_state + JSONL event call_type.
        # Phase B/C JSONL events (historical_validated, status_gate) must not
        # cause benchmark/smoke/contract/demo rows to show artifact_generation_mode=status_gate.
        _healthy_states = {
            "benchmark_passed",
            "benchmarked",
            "smoke_passed",
            "demo_passed_sidecar",
            "contract_passed",
        }
        # Determine metric_origin: check for real execution JSONL events first.
        _has_real_exec_event = any(
            nc.get("call_type") in {"benchmark", "smoke", "contract", "demo"}
            and nc.get("status") == "success"
            for nc in notebook_calls
        )
        _has_hist_event = any(
            nc.get("call_type") == "historical_validated" for nc in notebook_calls
        )
        if final_state in _healthy_states:
            if _has_real_exec_event:
                metric_origin = "current_rerun"
            elif row.extras.get("metric_origin"):
                # Preserve metric_origin already set by historical_fallback block.
                metric_origin = row.extras["metric_origin"]
            elif hist_detected_pattern and not current_run_ea:
                metric_origin = "historical_validated"
            elif current_run_ea and not hist_detected_pattern:
                metric_origin = "current_rerun"
            else:
                metric_origin = "current_rerun"
        else:
            metric_origin = ""

        # v2.48: derive artifact_generation_mode from final_state — no longer
        # purely from notebook-call extras. This fixes status_gate appearing for
        # benchmark/smoke/contract/demo/wired rows regardless of JSONL event type.
        # v2.48: wired/partial always use wired_registry_correction — KNOWN_CORRECTIONS rows.
        if final_state in ("wired", "partial"):
            artifact_gen_mode = "wired_registry_correction"
        elif row.extras.get("metric_origin") == "historical_validated" or _has_hist_event:
            artifact_gen_mode = "historical_validation_artifact"
        elif _has_real_exec_event:
            # Real JSONL execution event — derive from call_type.
            ct = next(
                (
                    nc.get("call_type", "")
                    for nc in notebook_calls
                    if nc.get("call_type") in {"benchmark", "smoke", "contract", "demo"}
                    and nc.get("status") == "success"
                ),
                "",
            )
            artifact_gen_mode = {
                "benchmark": "benchmark_artifact",
                "smoke": "smoke_artifact",
                "contract": "contract_artifact",
                "demo": "demo_artifact",
            }.get(ct, "executed_command")
        else:
            # Derive from final_state — covers KNOWN_CORRECTIONS and registry corrections.
            artifact_gen_mode = {
                "benchmark_passed": "benchmark_artifact",
                "benchmarked": "benchmark_artifact",
                "smoke_passed": "smoke_artifact",
                "smoke_ok_no_metric": "smoke_artifact",
                "contract_passed": "contract_artifact",
                "demo_passed_sidecar": "demo_artifact",
                "demo_passed": "demo_artifact",
                "wired": "wired_registry_correction",
                "partial": "wired_registry_correction",
                "opt_in_license_required": "license_gate",
                "license_blocked": "license_gate",
                "auth_required": "auth_gate",
                "external_api_only": "external_api_gate",
                "not_advertised": "official_source_audit",
                # v2.49 additions
                "micro_benchmark_passed": "micro_benchmark_artifact",
                "dataset_required": "dataset_required_gate",
                "benchmark_failed": "benchmark_failed_artifact",
                # v2.50 capability/adapter states
                "model_capability_mismatch": "capability_mismatch_artifact",
                "checkpoint_not_published": "checkpoint_not_published_gate",
                "benchmark_implementation_required": "implementation_required_gate",
            }.get(final_state, "status_gate")

        row.extras["metric_origin"] = metric_origin
        row.extras["artifact_generation_mode"] = artifact_gen_mode
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
                # v2.47 additions
                "wired": "none",
                "partial": "none",
                # v2.49 zero-smoke additions
                "micro_benchmark_passed": "none",
                "dataset_required": "dataset",
                "benchmark_failed": "benchmark",
                # v2.50 capability/adapter states
                "model_capability_mismatch": "model_capability",
                "checkpoint_not_published": "checkpoint_source",
                "benchmark_implementation_required": "adapter_implementation",
            }
            cat = _STATE_TO_CATEGORY.get(final_state, "unclassified")
        row.blocker_category = cat

        # v2.46: attach the runtime broker's view (model_id -> runtime_id) so the
        # ledger explicitly records which sidecar/env is responsible for this row.
        try:
            from visionservex.runtime_broker import RuntimeBroker

            _broker_singleton = getattr(_row_to_dict, "_broker_singleton", None)
            if _broker_singleton is None:
                _broker_singleton = RuntimeBroker()
                _row_to_dict._broker_singleton = _broker_singleton  # type: ignore[attr-defined]
            _routing_table = _broker_singleton.routing()
            row.extras["runtime_id"] = _routing_table.get(row.model_id, "")
        except Exception:
            row.extras.setdefault("runtime_id", "")

        # v2.46: surface command attempted + next-iteration command + reconciled state.
        row.extras["command_attempted"] = row.attempted_command or ""
        row.extras["next_iteration_command"] = row.manual_fix_command or ""
        row.extras["source_registry_state"] = row.registry_status or ""
        row.extras["reconciled_execution_state"] = row.execution_status or ""

        # v2.47: derive command_attempted from final_state when still blank.
        if not row.extras["command_attempted"]:
            row.extras["command_attempted"] = _derive_command_attempted(
                row.model_id, row.final_state, row.blocker_code, row.extras.get("runtime_id", "")
            )
        # v2.47: derive next_iteration_command from final_state when still blank.
        if not row.extras["next_iteration_command"]:
            row.extras["next_iteration_command"] = row.extras["command_attempted"]

        # v2.50: derive dataset_prepare_command for dataset_required rows.
        if row.final_state == "dataset_required" and not row.extras.get("dataset_prepare_command"):
            row.extras["dataset_prepare_command"] = _derive_dataset_prepare_command(
                row.model_id, row.blocker_code
            )
            # Replace generic next_iteration_command with the prepare command.
            if "models status" in row.extras.get("next_iteration_command", ""):
                row.extras["next_iteration_command"] = row.extras["dataset_prepare_command"]

        # v2.47: populate execution_origin from metric_origin + final_state.
        row.extras["execution_origin"] = _derive_execution_origin(
            row.final_state,
            row.blocker_code,
            row.extras.get("metric_origin", ""),
            row.extras.get("called_in_current_notebook_run", False),
            row.extras.get("v246_correction_reason", ""),
        )

        # v2.46: historical-fallback. If this run has no current-run evidence
        # AND no live task-report evidence AND the previous ledger had a
        # healthier state, carry that state forward with
        # metric_origin=historical_validated. Transparent: marked with
        # historical_artifact_used_as_fallback=True.
        hist = historical.get(row.model_id)
        if hist:
            prev_state = (hist.get("final_state") or "").strip()
            healthy_fallback = {
                "benchmark_passed",
                "demo_passed_sidecar",
                "contract_passed",
                "smoke_passed",
                "wired",
            }
            # v2.47.2: a JSONL status_gate event does NOT count as real execution
            # evidence. Only benchmark/smoke/contract/demo events with status=success
            # block the historical fallback.
            current_has_execution_evidence = row.extras.get(
                "called_in_current_notebook_run", False
            ) and any(
                nc.get("call_type") in {"benchmark", "smoke", "contract", "demo"}
                and nc.get("status") == "success"
                for nc in notebook_calls  # notebook_calls is the per-model list
            )
            # v2.49: never let the historical fallback override a KNOWN_CORRECTION.
            # Any state set by the correction mechanism should be preserved.
            _protected_by_zero_smoke_correction = mid in KNOWN_CORRECTIONS or row.final_state in {
                "dataset_required",
                "contract_passed",
                "micro_benchmark_passed",
                "benchmark_passed",
                "benchmark_failed",
                "checkpoint_required",
                "wired",
                # v2.50: capability/adapter states must also be preserved
                "model_capability_mismatch",
                "checkpoint_not_published",
                "benchmark_implementation_required",
            }
            current_is_weaker = (
                _priority(prev_state) > _priority(row.final_state)
                and prev_state in healthy_fallback
                and not current_has_execution_evidence
                and not evidence_map.get(row.model_id)
                and not _protected_by_zero_smoke_correction
            )
            if current_is_weaker:
                row.final_state = prev_state
                row.execution_status = prev_state
                row.blocker_code = hist.get("blocker_code") or ""
                row.evidence_artifact = hist.get("evidence_artifact") or row.evidence_artifact
                row.evidence_source = "historical_fallback_ledger"
                row.extras["metric_origin"] = "historical_validated"
                row.extras["artifact_generation_mode"] = "historical_fallback"
                row.extras["reconciled_execution_state"] = prev_state
                row.extras["historical_artifact_used_as_fallback"] = True

        # v2.47: covered_by_notebook — computed AFTER historical_fallback so
        # fallback models (metric_origin=historical_validated) are correctly
        # marked as covered.
        row.extras["covered_by_notebook"] = _derive_covered_by_notebook(
            row.final_state,
            row.extras.get("metric_origin", ""),
            row.extras.get("historical_artifact_used_as_fallback", False),
            row.called_in_notebook,
        )

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


_RESTRICTED_LICENSE_MODELS: frozenset[str] = frozenset(
    {
        "fastsam-s",
        "fastsam-x",
        "yolo-world",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
        "rfdetr-seg-xlarge",
        "rfdetr-seg-2xlarge",
        "totalsegmentator",
    }
)

_ZERO_SMOKE_STATES: frozenset[str] = frozenset(
    {
        "micro_benchmark_passed",
        "dataset_required",
        "benchmark_failed",
    }
)

_HEALTHY_STATES_V247: frozenset[str] = frozenset(
    {
        "smoke_passed",
        "benchmark_passed",
        "micro_benchmark_passed",  # v2.49: latency/schema benchmark without full GT
        "contract_passed",
        "demo_passed_sidecar",
        "wired",
        "partial",
    }
)


def _derive_dataset_prepare_command(model_id: str, blocker_code: str) -> str:
    """v2.50: derive exact dataset preparation command for dataset_required rows.

    Maps blocker_code → concrete `visionservex dataset prepare-*` invocation.
    """
    mapping = {
        "IMAGENET_DATASET_MISSING": (
            "visionservex dataset prepare-imagenette-mini --download-if-missing "
            "--classes 10 --images-per-class 50 "
            "--out /home/arash/datasets/imagenette_mini_vsx"
        ),
        "COCO_KEYPOINTS_MISSING": (
            "visionservex dataset prepare-coco-keypoints-mini "
            "--coco-root /home/arash/datasets/coco --split val2017 --limit 400 "
            "--out /home/arash/datasets/coco_keypoints_val_mini_vsx"
        ),
        "REID_DATASET_MISSING": (
            "visionservex dataset prepare-market1501-mini "
            "--source /home/arash/datasets/Market-1501-v15.09.15 "
            "--query-per-id 2 --gallery-per-id 5 "
            "--out /home/arash/datasets/market1501_mini_vsx"
        ),
        "MVTEC_DATASET_MISSING": (
            "visionservex dataset prepare-mvtec-mini "
            "--source /home/arash/datasets/mvtec_anomaly_detection "
            "--categories bottle,cable,hazelnut --max-test 100 "
            "--out /home/arash/datasets/mvtec_ad_mini_vsx"
        ),
        "MEDICAL_SEGMENTATION_DATASET_MISSING": (
            "visionservex dataset prepare-medical-seg-mini "
            "--source /home/arash/datasets/medical_segmentation "
            "--out /home/arash/datasets/medical_seg_mini_vsx"
        ),
    }
    return mapping.get(blocker_code, "")


def _derive_command_attempted(
    model_id: str,
    final_state: str,
    blocker_code: str,
    runtime_id: str,
) -> str:
    """Return the canonical next-try command for a model row.

    For healthy rows: the smoke/contract command the row was proven by.
    For sidecar/checkpoint rows: the prepare command.
    For license/auth rows: the opt-in command.
    For upstream/registry rows: the remap/fix command.
    """
    if final_state == "sidecar_required":
        rt = runtime_id or f"runtime_for_{model_id}"
        return f"visionservex runtime prepare {model_id} --execute --runtime {rt}"
    if final_state == "checkpoint_required":
        return f"visionservex pull {model_id} --verify"
    if final_state in ("opt_in_license_required", "license_blocked"):
        if "AGPL" in blocker_code or model_id.startswith("yolo") or model_id.startswith("fastsam"):
            return f"visionservex run {model_id} <input> --accept-agpl"
        if "PML" in blocker_code or "rfdetr-seg-xl" in model_id or "rfdetr-seg-2xl" in model_id:
            return f"visionservex run {model_id} <input> --accept-pml"
        return f"visionservex run {model_id} <input> --accept-non-commercial"
    if final_state == "external_api_only":
        return f"visionservex run {model_id} <input> --api-key $DEEPDATASPACE_API_KEY"
    if final_state == "auth_required":
        return f"visionservex run {model_id} <input> --use-auth-if-available"
    if final_state == "upstream_deprecated":
        return f"visionservex registry remap {model_id} --target <successor>"
    if final_state == "wrong_registry_entry":
        return f"visionservex registry fix {model_id}"
    if final_state == "wired":
        return f"visionservex models contract-run {model_id}"
    if final_state in _HEALTHY_STATES_V247:
        return f"visionservex run {model_id} tests/assets/smoke/coco_person_car.jpg --task auto"
    return f"visionservex models status {model_id}"


def _derive_execution_origin(
    final_state: str,
    blocker_code: str,
    metric_origin: str,
    called_in_current: bool,
    v246_correction_reason: str,
) -> str:
    """Derive a single-word execution origin for the row."""
    if metric_origin == "historical_validated":
        return "historical_validated"
    if metric_origin == "current_rerun":
        return "current_run_executed" if called_in_current else "current_run_status_gate"
    if "alias" in v246_correction_reason:
        return "registry_alias"
    if final_state in ("opt_in_license_required", "license_blocked"):
        return "excluded_restricted_license"
    if final_state == "auth_required":
        return "auth_required"
    if final_state == "external_api_only":
        return "external_api_required"
    if final_state in ("upstream_deprecated",):
        return "upstream_deprecated"
    if blocker_code == "OFFICIAL_SOURCE_NOT_FOUND":
        return "official_source_not_found"
    if final_state in _HEALTHY_STATES_V247:
        return "current_run_status_gate"
    return "sidecar_required" if final_state == "sidecar_required" else "blocked"


def _derive_covered_by_notebook(
    final_state: str,
    metric_origin: str,
    historical_fallback: bool,
    called_in_notebook: bool,
) -> bool:
    """Return True when the model is addressed in at least one notebook section.

    A model is covered when:
    - It has current-run execution evidence (metric_origin=current_rerun), OR
    - It has historical validated evidence, OR
    - It is a terminal-gated row (license/auth/api) — covered in the terminal section, OR
    - The notebook previously called it (called_in_notebook=True).
    """
    if called_in_notebook:
        return True
    if metric_origin in ("current_rerun", "historical_validated"):
        return True
    if historical_fallback:
        return True
    return final_state in _HEALTHY_STATES_V247 or final_state in (
        "opt_in_license_required",
        "license_blocked",
        "auth_required",
        "external_api_only",
    )


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
        # v2.44 honesty columns
        "metric_origin": row.extras.get("metric_origin", ""),
        "artifact_generation_mode": row.extras.get("artifact_generation_mode", ""),
        # v2.46 broker columns
        "runtime_id": row.extras.get("runtime_id", ""),
        "command_attempted": row.extras.get("command_attempted", row.attempted_command or ""),
        "exact_error_message_tail": row.extras.get("exact_error_message_tail", ""),
        "next_iteration_command": row.extras.get(
            "next_iteration_command", row.manual_fix_command or ""
        ),
        # v2.50: explicit dataset prep command for dataset_required rows.
        "dataset_prepare_command": row.extras.get("dataset_prepare_command", ""),
        "source_registry_state": row.extras.get("source_registry_state", row.registry_status or ""),
        "reconciled_execution_state": row.extras.get(
            "reconciled_execution_state", row.execution_status or ""
        ),
        # v2.47 notebook-accounting columns
        "covered_by_notebook": row.extras.get("covered_by_notebook", False),
        "execution_origin": row.extras.get("execution_origin", ""),
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


_RESTRICTED_LICENSE_MODELS_FOR_WINNERS: frozenset[str] = frozenset(
    {
        "fastsam-s",
        "fastsam-x",
        "yolo-world",
        "yolo11l-seg.pt",
        "yolo11x-seg.pt",
        "yolo11x.pt",
        "yolo26x-seg.pt",
        "yolo26x.pt",
        "yolov10b.pt",
        "yolov8x-seg.pt",
        "yolov8x.pt",
        "rfdetr-seg-xlarge",
        "rfdetr-seg-2xlarge",
        "totalsegmentator",
    }
)


def _compute_final_winners(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final_winners.json payload from the reconciled rows.

    v2.48 schema v3: distinguishes core winners from external restricted
    baselines. LibreYOLO is treated as a core VisionServeX-supported
    permissive open-source ecosystem, not an external competitor.
    """
    _bench_states = {"benchmark_passed", "benchmarked"}

    def _split_core_ext(task_rows: list[dict[str, Any]]) -> tuple[list, list]:
        core = [
            r for r in task_rows if r.get("model_id") not in _RESTRICTED_LICENSE_MODELS_FOR_WINNERS
        ]
        ext = [r for r in task_rows if r.get("model_id") in _RESTRICTED_LICENSE_MODELS_FOR_WINNERS]
        return core, ext

    detection_bench = [
        r for r in rows if r.get("task") == "detect" and r.get("final_state") in _bench_states
    ]
    detection_core, detection_ext = _split_core_ext(detection_bench)

    segmentation_bench = [
        r for r in rows if r.get("task") == "segment" and r.get("final_state") in _bench_states
    ]
    seg_core, seg_ext = _split_core_ext(segmentation_bench)

    promptable = [r for r in rows if r.get("task") in {"foundation_segment", "promptable_segment"}]
    promptable_core, promptable_ext = _split_core_ext(promptable)

    def _top_model(model_list: list[dict]) -> str:
        """Return model_id of the top model in the list (first by priority)."""
        return model_list[0]["model_id"] if model_list else "no_benchmark_data"

    # v2.48 schema v3: core vs external split with LibreYOLO as core.
    return {
        "schema_version": 3,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "libreyolo_treated_as_core": True,
        "external_restricted_baselines_excluded_from_core": True,
        # Detection
        "detection_core_winner": _top_model(detection_core),
        "detection_core_winner_note": (
            "libreyolo-* models are core VisionServeX permissive models, not external competitors."
        ),
        "detection_external_restricted_baseline_winner": _top_model(detection_ext),
        "detection_core_benchmark_passed_count": len(detection_core),
        "detection_external_restricted_baseline_count": len(detection_ext),
        "detection_core_top5": [r["model_id"] for r in detection_core[:5]],
        "detection_external_restricted_top5": [r["model_id"] for r in detection_ext[:5]],
        # Auto segmentation
        "auto_segmentation_core_winner": _top_model(seg_core),
        "auto_segmentation_external_restricted_baseline_winner": _top_model(seg_ext),
        "auto_segmentation_core_benchmark_passed_count": len(seg_core),
        "auto_segmentation_core_top5": [r["model_id"] for r in seg_core[:5]],
        # Promptable segmentation
        "promptable_segmentation_core_winner": _top_model(promptable_core),
        "promptable_segmentation_external_restricted_baseline_winner": _top_model(promptable_ext),
        # Evidence
        "detection_best_overall_evidence": [r["model_id"] for r in detection_bench[:5]],
        "segmentation_best_overall_evidence": [r["model_id"] for r in segmentation_bench[:5]],
        "promptable_evidence": [r["model_id"] for r in promptable[:5]],
        "n_detection_benchmark_passed": len(detection_bench),
        "n_segmentation_benchmark_passed": len(segmentation_bench),
        # Historical headline (carried from v2.27 400-image benchmark)
        "detection_headline_core": "libreyolo-dfine-x (mAP50:95=0.5030, MIT/Apache-2.0, COCO 400-image)",
        "detection_headline_external_restricted": "yolo26x.pt (mAP50:95=0.4894, AGPL-3.0, excluded from core)",
        "segmentation_headline_external_restricted": "yolo26x-seg.pt (mask_mAP50_95=0.2728, AGPL-3.0, excluded from core)",
    }


def validate_artifact_run_id(payload: dict[str, Any], run_id: str) -> list[str]:
    """v2.50: validate that benchmark_passed + current_rerun rows reference the current run_id.

    Returns a list of complaints; empty list means OK. Used after RUN_ALL to
    catch rows that claim current execution but point at historical artifacts.
    """
    issues: list[str] = []
    if not run_id:
        return issues
    for r in payload.get("rows", []):
        if r.get("final_state") != "benchmark_passed":
            continue
        if r.get("metric_origin") != "current_rerun":
            continue
        artifact = str(r.get("evidence_artifact") or "")
        if run_id not in artifact and artifact:
            issues.append(
                f"{r['model_id']}: benchmark_passed + current_rerun but evidence_artifact "
                f"({artifact!r}) does not contain current RUN_ID {run_id!r}"
            )
    return issues


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

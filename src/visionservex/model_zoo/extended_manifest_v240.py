# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v2.40.0: extended manifest covering the 81 model_ids that v2.39's
canonical ledger reported as ``absent_from_manifest``.

These entries are derived from the same upstream sources that the
existing :data:`visionservex.model_zoo.manifest.SOURCE_MANIFEST` rows
cite — they are size/preset variants of already-tracked families. The
extended block is merged into ``SOURCE_MANIFEST`` at import time by
:func:`apply_v240_extension`.

This module is additive: it never removes or overwrites a manifest row.
"""

from __future__ import annotations

from visionservex.model_zoo.manifest import ModelSource

_EXTENDED_ROWS: list[ModelSource] = [
    # ---------------- D-FINE family (size variants of dfine-x-o365-coco) -------------
    *[
        ModelSource(
            model_id=mid,
            family="dfine",
            task="detect",
            official_repo="https://github.com/Peterande/D-FINE",
            paper_url="https://arxiv.org/abs/2410.13842",
            hf_repo=f"PekingU/{mid}",
            license="Apache-2.0",
            license_risk="none",
            install_command="pip install visionservex",
            hf_class="AutoModelForObjectDetection",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid in (
            "dfine-n",
            "dfine-s",
            "dfine-m",
            "dfine-l",
            "dfine-x",
            "dfine-n-coco",
            "dfine-s-coco",
            "dfine-m-coco",
            "dfine-l-coco",
            "dfine-x-coco",
            "dfine-m-o365-coco",
        )
    ],
    # ---------------- DEIMv2 extra sizes (atto/femto/pico/n) -----------------
    *[
        ModelSource(
            model_id=mid,
            family="deimv2",
            task="detect",
            official_repo="https://github.com/Intellindust-AI-Lab/DEIMv2",
            hf_repo=f"Intellindust/DEIMv2_DINOv3_{size.upper()}_COCO",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="expert_sidecar",
        )
        for mid, size in (
            ("deimv2-atto", "atto"),
            ("deimv2-femto", "femto"),
            ("deimv2-pico", "pico"),
            ("deimv2-n", "n"),
        )
    ],
    # ---------------- DEIM legacy (deprecated upstream) ----------------------
    *[
        ModelSource(
            model_id=mid,
            family="deim",
            task="detect",
            official_repo="https://github.com/ShihuaHuang95/DEIM",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="unavailable",
            recommended_action="do_not_add",
            known_blockers=["upstream_deprecated"],
            notes="DEIM (v1) deprecated in favour of DEIMv2.",
        )
        for mid in ("deim-m", "deim-s")
    ],
    # ---------------- RF-DETR-Seg variants (nano/small/medium core; XL/2XL plus) -----
    *[
        ModelSource(
            model_id=mid,
            family="rfdetr_seg",
            task="segment",
            official_repo="https://github.com/roboflow/rf-detr",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid in ("rfdetr-seg-nano", "rfdetr-seg-small", "rfdetr-seg-medium")
    ],
    # ---------------- RF-DETR detect size variants ---------------------------
    *[
        ModelSource(
            model_id=mid,
            family="rfdetr",
            task="detect",
            official_repo="https://github.com/roboflow/rf-detr",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid in ("rfdetr-base", "rfdetr-medium", "rfdetr-nano")
    ],
    # ---------------- LibreYOLO permissive core bundles ----------------------
    # v2.48: LibreYOLO is a core VisionServeX-supported permissive open-source
    # ecosystem (MIT code license; Apache-2.0 weights verified per HF model card).
    # NOT an external competitor — should appear as core winner in leaderboards.
    *[
        ModelSource(
            model_id=mid,
            family="libreyolo",
            task="detect",
            official_repo="https://github.com/LibreYOLO/libreyolo",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
            notes="LibreYOLO detection model (Apache-2.0 weights, MIT code)",
        )
        for mid in (
            # D-FINE family (Apache-2.0)
            "libreyolo-dfine-n",
            "libreyolo-dfine-s",
            "libreyolo-dfine-m",
            "libreyolo-dfine-l",
            "libreyolo-dfine-x",
            # YOLOX family (Apache-2.0)
            "libreyolo-yolox-n",
            "libreyolo-yolox-t",
            "libreyolo-yolox-s",
            "libreyolo-yolox-m",
            "libreyolo-yolox-l",
            "libreyolo-yolox-x",
            # RT-DETR family (Apache-2.0)
            "libreyolo-rtdetr-r18",
            "libreyolo-rtdetr-r34",
            "libreyolo-rtdetr-r50",
            "libreyolo-rtdetr-r50m",
            "libreyolo-rtdetr-r101",
            "libreyolo-rtdetr-l",
            "libreyolo-rtdetr-x",
            # RF-DETR family (Apache-2.0)
            "libreyolo-rfdetr-n",
            "libreyolo-rfdetr-s",
            "libreyolo-rfdetr-m",
            "libreyolo-rfdetr-l",
        )
    ],
    # v2.58 source-truth cleanup: Only RF-DETR segmentation is officially published
    # by LibreYOLO. D-FINE/YOLOX/RT-DETR seg weights were never published (HF 404).
    # RT-DETR r18-r101-seg weights exist but produce no masks (capability_mismatch).
    # Their detection counterparts (without -seg) already exist and are benchmark_passed.
    *[
        ModelSource(
            model_id=mid,
            family="libreyolo",
            task="segment",
            official_repo="https://github.com/LibreYOLO/libreyolo",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
            notes="LibreYOLO RF-DETR seg — only officially published LibreYOLO seg model (Apache-2.0)",
        )
        for mid in (
            "libreyolo-rfdetr-n-seg",
            "libreyolo-rfdetr-s-seg",
            "libreyolo-rfdetr-m-seg",
            "libreyolo-rfdetr-l-seg",
        )
    ],
    # YOLOv9 family — v2.48 audit: LibreYOLO/LibreYOLO9s HF model card shows MIT
    # (weights from MultimediaTechLab/YOLO which relicensed under MIT). Auto-pull.
    *[
        ModelSource(
            model_id=mid,
            family="libreyolo",
            task="detect",
            official_repo="https://github.com/LibreYOLO/libreyolo",
            license="MIT",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
            notes="LibreYOLO YOLOv9 (MIT per HF model card via MultimediaTechLab/YOLO)",
        )
        for mid in (
            "libreyolo-yolov9-t",
            "libreyolo-yolov9-s",
            "libreyolo-yolov9-m",
            "libreyolo-yolov9-c",
        )
    ],
    # YOLO-NAS — still non-commercial (Deci.AI). Excluded from default-safe core.
    *[
        ModelSource(
            model_id=mid,
            family="libreyolo",
            task="detect",
            official_repo="https://github.com/LibreYOLO/libreyolo",
            license="Deci-AI-non-commercial",
            license_risk="non_commercial",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="do_not_add",
            notes="LibreYOLO YOLO-NAS — Deci.AI non-commercial license. Blocked by default.",
        )
        for mid in (
            "libreyolo-yolonas-s",
            "libreyolo-yolonas-m",
            "libreyolo-yolonas-l",
        )
    ],
    # ---------------- Ultralytics baselines (AGPL-3.0) -----------------------
    *[
        ModelSource(
            model_id=mid,
            family="ultralytics",
            task=task,
            official_repo="https://github.com/ultralytics/ultralytics",
            license="AGPL-3.0",
            license_risk="gpl",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="do_not_add",
            notes="Ultralytics baseline weight — license-gated. Use for baseline metrics only.",
        )
        for mid, task in (
            ("yolo11x.pt", "detect"),
            ("yolo26x.pt", "detect"),
            ("yolov10b.pt", "detect"),
            ("yolov8x.pt", "detect"),
            ("yolo11l-seg.pt", "segment"),
            ("yolo11x-seg.pt", "segment"),
            ("yolo26x-seg.pt", "segment"),
            ("yolov8x-seg.pt", "segment"),
        )
    ],
    # ---------------- Grounding-DINO open-source variants --------------------
    *[
        ModelSource(
            model_id=mid,
            family="grounding_dino",
            task="open_vocab",
            official_repo="https://github.com/IDEA-Research/GroundingDINO",
            hf_repo=hf,
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid, hf in (
            ("grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny"),
            ("grounding-dino-swin-t", "IDEA-Research/grounding-dino-tiny"),
            ("grounding-dino-swin-b", "IDEA-Research/grounding-dino-base"),
        )
    ],
    # ---------------- Grounding-DINO 1.5 / 1.6 (cloud or auth) ---------------
    *[
        ModelSource(
            model_id=mid,
            family="grounding_dino",
            task="open_vocab",
            official_repo="https://github.com/IDEA-Research/Grounded-SAM-2",
            license="Apache-2.0",
            license_risk="api_only",
            runnable_in_visionservex=False,
            access_status="api_token",
            recommended_action="external_api",
            notes="Grounding-DINO 1.5/1.6 require official API key for inference.",
        )
        for mid in (
            "grounding-dino-1.5",
            "grounding-dino-1.6",
        )
    ],
    # ---------------- SAM family extras -------------------------------------
    *[
        ModelSource(
            model_id=mid,
            family="sam",
            task="foundation_segment",
            official_repo="https://github.com/facebookresearch/segment-anything",
            hf_repo=hf,
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid, hf in (
            ("sam-vit-large", "facebook/sam-vit-large"),
            ("sam-vit-huge", "facebook/sam-vit-huge"),
        )
    ],
    # ---------------- SAM2 (Facebook) ---------------------------------------
    *[
        ModelSource(
            model_id=mid,
            family="sam2",
            task="foundation_segment",
            official_repo="https://github.com/facebookresearch/sam2",
            hf_repo=hf,
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid, hf in (
            ("sam2-hiera-base-plus", "facebook/sam2-hiera-base-plus"),
            ("sam2-hiera-large", "facebook/sam2-hiera-large"),
            ("sam2-hiera-small", "facebook/sam2-hiera-small"),
            ("sam2-hiera-tiny", "facebook/sam2-hiera-tiny"),
        )
    ],
    # ---------------- SigLIP2 extra variants --------------------------------
    *[
        ModelSource(
            model_id=mid,
            family="siglip2",
            task="embed",
            official_repo="https://huggingface.co/google",
            hf_repo=f"google/{mid}",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid in ("siglip2-large-patch16-256", "siglip2-so400m-patch14-384")
    ],
    # ---------------- SwinV2 size variants ----------------------------------
    *[
        ModelSource(
            model_id=mid,
            family="swinv2",
            task="classify",
            official_repo="https://github.com/microsoft/Swin-Transformer",
            hf_repo=f"microsoft/{mid}",
            license="MIT",
            license_risk="none",
            runnable_in_visionservex=True,
            access_status="open",
            recommended_action="add_now",
        )
        for mid in ("swinv2-tiny", "swinv2-small", "swinv2-base")
    ],
    # ---------------- OneFormer / MaxViT variants ---------------------------
    ModelSource(
        model_id="oneformer-swin-large",
        family="oneformer",
        task="segment",
        official_repo="https://github.com/SHI-Labs/OneFormer",
        hf_repo="shi-labs/oneformer_coco_swin_large",
        license="MIT",
        license_risk="none",
        runnable_in_visionservex=True,
        access_status="open",
        recommended_action="add_now",
    ),
    ModelSource(
        model_id="maxvit-tiny-tf-224",
        family="maxvit",
        task="classify",
        official_repo="https://github.com/google-research/maxvit",
        hf_repo="timm/maxvit_tiny_tf_224.in1k",
        license="Apache-2.0",
        license_risk="none",
        runnable_in_visionservex=True,
        access_status="open",
        recommended_action="add_now",
    ),
    # ---------------- OpenMMLab — InternImage / CO-DETR / RTMDet / RTMPose --
    *[
        ModelSource(
            model_id=mid,
            family="internimage",
            task="classify",
            official_repo="https://github.com/OpenGVLab/InternImage",
            license="MIT",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes="OpenMMLab sidecar required.",
        )
        for mid in (
            "internimage-t",
            "internimage-s",
            "internimage-b",
            "internimage-l",
            "internimage-h",
        )
    ],
    *[
        ModelSource(
            model_id=mid,
            family="codetr",
            task="segment",
            official_repo="https://github.com/Sense-X/Co-DETR",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes="OpenMMLab detseg sidecar required.",
        )
        for mid in ("co-dino-inst-vit-l-coco", "co-dino-inst-vit-l-lvis")
    ],
    *[
        ModelSource(
            model_id=mid,
            family="rtmdet",
            task="obb",
            official_repo="https://github.com/open-mmlab/mmrotate",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes="OpenMMLab OBB sidecar required.",
        )
        for mid in (
            "rtmdet-r-t",
            "rtmdet-r-s",
            "rtmdet-r-m",
            "rtmdet-r-l",
            "rtmdet-r2-t",
            "rtmdet-r2-s",
            "rtmdet-r2-m",
            "rtmdet-r2-l",
        )
    ],
    *[
        ModelSource(
            model_id=mid,
            family="rtmpose",
            task="pose",
            official_repo="https://github.com/open-mmlab/mmpose",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes="OpenMMLab pose sidecar required.",
        )
        for mid in (
            "rtmpose-t",
            "rtmpose-s",
            "rtmpose-m",
            "rtmpose-m-384x288",
            "rtmpose-l",
            "rtmpose-l-384x288",
        )
    ],
    # ---------------- MaskDINO (legacy Detectron2 sidecar) -------------------
    *[
        ModelSource(
            model_id=mid,
            family="maskdino",
            task="segment",
            official_repo="https://github.com/IDEA-Research/MaskDINO",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            known_blockers=["MASKDINO_LEGACY_CUDA_BLACKWELL_UNSUPPORTED"],
            notes="Detectron2 sidecar (py3.8 + torch 1.9 + CUDA 11.1); incompatible with Blackwell sm_120.",
        )
        for mid in ("maskdino-r50-coco", "maskdino-r50-panoptic", "maskdino-swinl-coco")
    ],
    # ---------------- SAM/Promptable extras (smaller community variants) -----
    *[
        ModelSource(
            model_id=mid,
            family=fam,
            task="foundation_segment",
            official_repo=repo,
            license=lic,
            license_risk=risk,
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action=action,
            notes=note,
        )
        for mid, fam, repo, lic, risk, action, note in (
            (
                "edgesam",
                "edgesam",
                "https://github.com/chongzhou96/EdgeSAM",
                "Apache-2.0",
                "none",
                "expert_sidecar",
                "Edge SAM sidecar (lightweight promptable).",
            ),
            (
                "efficientsam",
                "efficientsam",
                "https://github.com/yformer/EfficientSAM",
                "Apache-2.0",
                "none",
                "expert_sidecar",
                "EfficientSAM sidecar.",
            ),
            (
                "hq-sam",
                "hq_sam",
                "https://github.com/SysCV/sam-hq",
                "Apache-2.0",
                "none",
                "expert_sidecar",
                "HQ-SAM sidecar.",
            ),
            (
                "mobilesam",
                "mobilesam",
                "https://github.com/ChaoningZhang/MobileSAM",
                "Apache-2.0",
                "none",
                "expert_sidecar",
                "MobileSAM sidecar.",
            ),
            (
                "medsam2",
                "medsam2",
                "https://github.com/bowang-lab/MedSAM",
                "Apache-2.0",
                "none",
                "expert_sidecar",
                "MedSAM2 sidecar.",
            ),
            (
                "fastsam-s",
                "fastsam",
                "https://github.com/CASIA-IVA-Lab/FastSAM",
                "AGPL-3.0",
                "agpl",
                "do_not_add",
                "FastSAM AGPL — opt-in only.",
            ),
            (
                "fastsam-x",
                "fastsam",
                "https://github.com/CASIA-IVA-Lab/FastSAM",
                "AGPL-3.0",
                "agpl",
                "do_not_add",
                "FastSAM AGPL — opt-in only.",
            ),
        )
    ],
    # ---------------- Surveillance / tracking sidecars ----------------------
    *[
        ModelSource(
            model_id=mid,
            family=fam,
            task="surveillance",
            official_repo=repo,
            license=lic,
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes=note,
        )
        for mid, fam, repo, lic, note in (
            (
                "bytetrack",
                "bytetrack",
                "https://github.com/ifzhang/ByteTrack",
                "MIT",
                "ByteTrack sidecar (tracker).",
            ),
            (
                "osnet-x1.0",
                "osnet",
                "https://github.com/KaiyangZhou/deep-person-reid",
                "MIT",
                "OSNet reid sidecar.",
            ),
        )
    ],
    # ---------------- Anomaly / SEEM / others -------------------------------
    ModelSource(
        model_id="anomalib-patchcore",
        family="anomalib",
        task="anomaly",
        official_repo="https://github.com/open-edge-platform/anomalib",
        license="Apache-2.0",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="expert_sidecar",
        notes="Anomalib sidecar.",
    ),
    *[
        ModelSource(
            model_id=mid,
            family="seem",
            task="segment",
            official_repo="https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="expert_sidecar",
            notes="SEEM sidecar (py3.9 + openmpi).",
        )
        for mid in ("seem-davit-d3", "seem-focal-t")
    ],
    # ---------------- Medical / Earth observation (opt-in) ------------------
    ModelSource(
        model_id="nnunet-v2",
        family="nnunet",
        task="medical",
        official_repo="https://github.com/MIC-DKFZ/nnUNet",
        license="Apache-2.0",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="expert_sidecar",
        notes="nnU-Net v2 medical segmentation sidecar.",
    ),
    ModelSource(
        model_id="totalsegmentator",
        family="totalsegmentator",
        task="medical",
        official_repo="https://github.com/wasserth/TotalSegmentator",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="non_core_license_optional",
        notes="TotalSegmentator non-core optional medical model.",
    ),
    ModelSource(
        model_id="prithvi-eo-2.0",
        family="prithvi",
        task="medical",
        official_repo="https://github.com/IBM/Prithvi-WxC",
        license="Apache-2.0",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="non_core_license_optional",
        notes="Prithvi Earth observation model — opt-in.",
    ),
    # ---------------- YOLO-World ---------------------------------------------
    ModelSource(
        model_id="yolo-world",
        family="yolo_world",
        task="open_vocab",
        official_repo="https://github.com/AILab-CVC/YOLO-World",
        license="AGPL-3.0",
        license_risk="gpl",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="do_not_add",
        notes="YOLO-World AGPL — license-gated.",
    ),
    # ---------------- RT-DETRv4 size variants (l/m/x missing in v2.x manifest) ----
    *[
        ModelSource(
            model_id=mid,
            family="rtdetrv4",
            task="detect",
            official_repo="https://github.com/RT-DETRs/RT-DETRv4",
            license="Apache-2.0",
            license_risk="none",
            runnable_in_visionservex=False,
            access_status="manual_download",
            recommended_action="expert_sidecar",
            known_blockers=["MANUAL_CHECKPOINT_REQUIRED"],
            notes="Google Drive checkpoint required.",
        )
        for mid in ("rtdetrv4-l", "rtdetrv4-m", "rtdetrv4-x")
    ],
    # ---------------- SwinV2-Large (download retry) -------------------------
    ModelSource(
        model_id="swinv2-large",
        family="swinv2",
        task="classify",
        official_repo="https://github.com/microsoft/Swin-Transformer",
        hf_repo="microsoft/swinv2-large-patch4-window12-192-22k",
        license="MIT",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="add_now",
        known_blockers=["DOWNLOAD_FAILED_RETRYABLE"],
        notes="HF CDN brotli/network issue intermittent.",
    ),
    # ---------------- OneFormer-DiNAT-Large (NATTEN sidecar) ----------------
    ModelSource(
        model_id="oneformer-dinat-large",
        family="oneformer",
        task="segment",
        official_repo="https://github.com/SHI-Labs/OneFormer",
        hf_repo="shi-labs/oneformer_coco_dinat_large",
        license="MIT",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="expert_sidecar",
        known_blockers=["NATTEN_REQUIRED"],
        notes="DiNAT backbone requires NATTEN, Python 3.11 sidecar.",
    ),
    # ---------------- OneFormer wrong-registry entry ------------------------
    ModelSource(
        model_id="oneformer-convnext-large",
        family="oneformer",
        task="segment",
        official_repo="https://github.com/SHI-Labs/OneFormer",
        license="MIT",
        license_risk="none",
        runnable_in_visionservex=False,
        access_status="unavailable",
        recommended_action="do_not_add",
        known_blockers=["WRONG_REGISTRY_ENTRY"],
        notes="SHI-Labs has no oneformer_coco_convnext_large; use swin/dinat instead.",
    ),
    # ---------------- RF-DETR-Seg PML opt-in variants -----------------------
    *[
        ModelSource(
            model_id=mid,
            family="rfdetr_seg",
            task="segment",
            official_repo="https://github.com/roboflow/rf-detr",
            license="PML-1.0",
            license_risk="check",
            runnable_in_visionservex=False,
            access_status="open",
            recommended_action="non_core_license_optional",
            known_blockers=["OPT_IN_LICENSE_REQUIRED"],
            notes="RF-DETR-Seg XLarge/2XLarge are Plus/PML-1.0 — opt-in only.",
        )
        for mid in ("rfdetr-seg-xlarge", "rfdetr-seg-2xlarge")
    ],
    # ---------------- AgriCLIP / DinoV3 (audit-only) -------------------------
    ModelSource(
        model_id="agriclip",
        family="agriclip",
        task="agriculture",
        official_repo="",
        license="check",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="audit_only",
        notes="AgriCLIP — audited only; not advertised in default suite.",
    ),
    ModelSource(
        model_id="dinov3-vitb16",
        family="dinov3",
        task="embed",
        official_repo="https://github.com/facebookresearch/dinov3",
        license="check",
        license_risk="check",
        runnable_in_visionservex=False,
        access_status="open",
        recommended_action="audit_only",
        notes="DINOv3 vit-b/16 — audited only; awaits official open weights.",
    ),
]


def apply_v240_extension() -> None:
    """Merge the v2.40 extended rows into :data:`SOURCE_MANIFEST` in place.

    Idempotent: existing rows are left untouched.
    """
    from visionservex.model_zoo import manifest as _m

    for src in _EXTENDED_ROWS:
        _m.SOURCE_MANIFEST.setdefault(src.model_id, src)


__all__ = ["_EXTENDED_ROWS", "apply_v240_extension"]

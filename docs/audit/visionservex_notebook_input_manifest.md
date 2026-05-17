# VisionServeX Notebook Input Manifest

Version: 2.13.1  
Audit date: 2026-05-16  
Models: 113  
Families: 27

## Notebook Sections

- **0_package_audit** — Package audit (✅ default)
- **1_load_matrix** — Model load matrix (✅ default)
- **2_detection** — Detection vs Ultralytics (✅ default)
- **3_open_vocab** — Open-vocabulary & VLM (✅ default)
- **4_classification** — Classification (✅ default)
- **5_embedding** — Feature / Embedding / Retrieval (✅ default)
- **6_sam** — SAM / Promptable Segmentation (✅ default)
- **7_medical** — Medical imaging (⚙️ optional)
- **8_anomaly** — Industrial anomaly detection (⚙️ optional)
- **9_surveillance** — Surveillance / Video search (⚙️ optional)
- **10_openmmlab** — OpenMMLab sidecar (RTMPose/RTMDet) (⚙️ optional)
- **11_aerial** — Aerial / Remote sensing / OBB (⚙️ optional)
- **12_non_core** — Non-core / gated / sidecar (⚙️ optional)

## Ultralytics Comparison Eligible Models

- `deim-m`
- `deim-s`
- `deimv2-m`
- `deimv2-s`
- `dfine-l`
- `dfine-l-coco`
- `dfine-l-o365-coco`
- `dfine-m`
- `dfine-m-coco`
- `dfine-m-o365-coco`
- `dfine-n`
- `dfine-n-coco`
- `dfine-s`
- `dfine-s-coco`
- `dfine-s-o365-coco`
- `dfine-x`
- `dfine-x-coco`
- `dfine-x-o365-coco`
- `rfdetr-base`
- `rfdetr-large`
- `rfdetr-medium`
- `rfdetr-nano`
- `rfdetr-small`
- `rtdetrv4-l`
- `rtdetrv4-m`
- `rtdetrv4-s`
- `rtdetrv4-x`

## Expected Blockers

- `GATED_HF_AUTH_REQUIRED`: visionservex sam-family login-help sam3.1
- `DETECTRON2_REQUIRED`: bash scripts/run_maskdino_smoke.sh
- `OPENMMLAB_REQUIRED`: bash scripts/run_openmmlab_rtmpose_smoke.sh
- `OBB_INFERENCER_UNAVAILABLE`: bash scripts/run_mmrotate_oriented_rcnn_smoke.sh
- `ANOMALIB_REQUIRED`: bash scripts/run_anomaly_smoke.sh
- `DO_NOT_ADD`: FastSAM (AGPL-3.0), DeepSORT (GPL-3.0) — excluded from permissive core
- `NON_CORE_LICENSE_OPTIONAL`: RF-DETR Plus/XL/2XL (PML 1.0), TotalSegmentator tissue subtasks (proprietary)

# VisionServeX v3.2 — Real Model Activation Continuation

## Real new-mode model activations: 5 (target was 12)

Honest outcome: 5 REAL new-runtime-mode activations + a fully-proven blocker table for every remaining model (the prompt's stated alternative). No fake benchmarks; no token; no mirroring.

| model_id | new_mode | metric | evidence |
|---|---|---|---|
| mobilesam-onnx | onnx_cpu_runtime | decoder 17.58ms, iou 0.455 | reports(_runs)/v32_sam_onnx_benchmark.json |
| sam-vit-b-onnx | onnx_cpu_runtime | decoder 17.86ms, iou 0.867 | reports(_runs)/v32_sam_onnx_benchmark.json |
| sam2.1-hiera-tiny (transformers-image) | transformers_image_backend | mask_area 21399, 313.7ms | v32_sam2_transformers_image.json |
| sam2.1-video-tiny | video_object_tracking | 6 frames, areas [5596, 5597, 5596, 5587, | v32_sam2_video*.json |
| sam2.1-video-small | video_object_tracking | 6 frames, areas [5600, 5599, 5598, 5599, | v32_sam2_video*.json |

## Sidecar attempts (logged blockers)

- **medsam2**: FAILED — no image_processor config (raw SAM2 .pt, not transformers format) → next: `conda create -n vsx-medsam2 python=3.10 && pip install SAM-2 + MedSAM2`
- **rtmdet-r2-s (+20 OpenMMLab)**: mmcv build fails on torch 2.11+cu130: ModuleNotFoundError: No module named 'pkg_resources' (setuptools incompatibility;  → next: `conda create -n vsx-mmlab python=3.10 && pip install torch==2.1.0 && p`

## BYOT (no token present → auth_required; paths implemented, weights never mirrored)

- sam3-base (HF_TOKEN): auth_required
- grounding-dino-1.5 (DEEPDATASPACE_API_KEY): auth_required
- grounding-dino-1.6 (DEEPDATASPACE_API_KEY): auth_required
- grounding-dino-1.5-pro (DEEPDATASPACE_API_KEY): auth_required
- grounding-dino-1.6-pro (DEEPDATASPACE_API_KEY): auth_required
- dino-x-api (DEEPDATASPACE_API_KEY): auth_required
- dinov3-vitb16 (HF_TOKEN): auth_required

## Proven blocker table (every remaining family)
| model | blocker | state | next_command |
|---|---|---|---|
| internimage-t/s/b/l/h | mmcv build fails (torch 2.11+cu130, pkg_resources) | sidecar_required | conda env + pip install mmcv==2.1.0 (cu121/torch2. |
| maskdino-r50-coco/panoptic/swinl | Detectron2 + MaskDINO build chain | sidecar_required | conda env + pip install detectron2 (torch2.1 wheel |
| seem-davit-d3 / seem-focal-t | X-Decoder/SEEM custom ops + mpi4py | sidecar_required | conda env + SEEM repo install + checkpoint |
| co-dino-inst-vit-l-coco/lvis | OpenMMLab Co-DETR projects (mmcv) | sidecar_required | conda env + mmcv2.1 + mmdet + Co-DETR project conf |
| oneformer-dinat-large | NATTEN compile (no wheel for torch 2.11) | sidecar_required | pip install natten -f https://shi-labs.com/natten/ |
| rtdetrv4-l/m/s/x | checkpoint gated on Google Drive (gdown abuse filt | checkpoint_required | gdown <drive-id> -O ckpt && visionservex rtdetrv4  |
| sam3 image/video/text/visual/exemp | no separately published checkpoint (sam3-base gate | not_released | watch github.com/facebookresearch/sam3 releases |
| sam3.1-* | SAM 3.1 not publicly released as of 2026-06 | not_released | watch Meta SAM 3.1 release |
| tinysam / q-tinysam | Apache-2.0 tag but SA-1B research-only distillatio | legal_review_required | visionservex legal review tinysam |
| hq-sam2 / light-hq-sam / focalclic | non-commercial training data (HQSeg-44K / MAE CC-B | legal_review_required | visionservex legal review <model> |
| dino-x detection/segmentation/phra | API-only, no downloadable weights | external_api_only | export DEEPDATASPACE_API_KEY=... && visionservex d |
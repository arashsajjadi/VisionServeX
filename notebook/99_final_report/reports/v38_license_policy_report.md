# VisionServeX v3.8 — License Policy Matrix Report

Generated from `visionservex.licensing.policy` (single source of truth). The CLI, Python API, tests, notebooks, and docs all read this same table.

**Total models classified: 94**

## Summary by final policy

| final_policy | count |
|---|---|
| `commercial_safe_core` | 39 |
| `byot_license_required` | 23 |
| `auth_required_license_pending` | 0 |
| `external_api_only_terms_required` | 9 |
| `noncommercial_restricted` | 7 |
| `enterprise_license_required` | 4 |
| `legal_review_required` | 11 |
| `excluded_from_core` | 0 |
| `not_released_or_unverifiable` | 1 |
| **total** | **94** |

## Hard rules (enforced in code + tests)

- A Hugging Face token does NOT grant redistribution rights.
- Gated models are never packaged into PyPI / GitHub / Docker (can_ship_weights=False for every row).
- Non-commercial models never run in production mode and are never default_safe.
- AGPL / enterprise models never enter the default_safe core.
- API-only models are never counted as local models (is_local=False).
- legal_review models are never commercial_safe until the review is resolved.
- Code license, weights license, and dataset/pretraining risk are tracked separately.

## Warning texts

- **byot**: This model is gated or uses a custom upstream license. You must use your own Hugging Face token and accept the upstream license yourself. VisionServeX does not redistribute the weights. Commercial use depends on the upstream license terms you accepted.
- **noncommercial**: WARNING: This model is non-commercial/restricted. Do not use it for paid SaaS, client work, production annotation, or commercial products unless you have written permission from the model owner.
- **enterprise**: WARNING: This model requires an enterprise/commercial license or has AGPL/copyleft obligations. It is disabled in VisionServeX commercial-safe core.
- **api**: External API model. Your data may leave the local environment. You must provide your own provider API key and comply with provider terms.
- **legal_review**: License/provenance is unclear. Legal review required before commercial use.
- **commercial_safe**: Commercial-safe core model (permissive license). Weights are pulled from the official source on demand; VisionServeX does not bundle them.
- **not_released**: This model is not released, not found at an official source, or its provenance could not be verified. It cannot be run or shipped.

## Models by policy

### `commercial_safe_core` (39)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `sam-vit-base` | sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam-vit-base` |
| `sam-vit-large` | sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam-vit-large` |
| `sam-vit-huge` | sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam-vit-huge` |
| `mobilesam` | sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull mobilesam` |
| `efficientsam` | sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull efficientsam` |
| `sam2-hiera-tiny` | sam2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2-hiera-tiny` |
| `sam2-hiera-small` | sam2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2-hiera-small` |
| `sam2-hiera-base-plus` | sam2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2-hiera-base-plus` |
| `sam2-hiera-large` | sam2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2-hiera-large` |
| `sam2.1-hiera-tiny` | sam2.1 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2.1-hiera-tiny` |
| `sam2.1-hiera-small` | sam2.1 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2.1-hiera-small` |
| `sam2.1-hiera-base-plus` | sam2.1 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2.1-hiera-base-plus` |
| `sam2.1-hiera-large` | sam2.1 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull sam2.1-hiera-large` |
| `dinov2-small` | dinov2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull dinov2-small` |
| `dinov2-base` | dinov2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull dinov2-base` |
| `dinov2-large` | dinov2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull dinov2-large` |
| `dinov2-giant` | dinov2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull dinov2-giant` |
| `grounding-dino-tiny` | grounding-dino | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull grounding-dino-tiny` |
| `grounding-dino-base` | grounding-dino | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull grounding-dino-base` |
| `grounding-dino-swin-t` | grounding-dino | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull grounding-dino-swin-t` |
| `grounding-dino-swin-b` | grounding-dino | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull grounding-dino-swin-b` |
| `florence-2-base` | florence2 | MIT | MIT | False | True | True | `visionservex model pull florence-2-base` |
| `florence-2-large` | florence2 | MIT | MIT | False | True | True | `visionservex model pull florence-2-large` |
| `clip-vit-base-patch32` | clip | MIT | MIT | False | True | True | `visionservex model pull clip-vit-base-patch32` |
| `owlvit-base-patch32` | owlvit | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull owlvit-base-patch32` |
| `owlv2-base-patch16-ensemble` | owlv2 | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull owlv2-base-patch16-ensemble` |
| `depth-anything-small` | depth-anything | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull depth-anything-small` |
| `rfdetr-seg-nano` | rf-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rfdetr-seg-nano` |
| `rfdetr-seg-small` | rf-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rfdetr-seg-small` |
| `rfdetr-seg-medium` | rf-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rfdetr-seg-medium` |
| `rfdetr-seg-large` | rf-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rfdetr-seg-large` |
| `efficientvit-sam-l0` | efficient-sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull efficientvit-sam-l0` |
| `efficientvit-sam-l1` | efficient-sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull efficientvit-sam-l1` |
| `efficientvit-sam-l2` | efficient-sam | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull efficientvit-sam-l2` |
| `ritm` | interactive-seg | MIT | MIT | False | True | True | `visionservex model pull ritm  # MIT; user-supplied checkpoint path ok` |
| `maskdino` | maskdino | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull maskdino` |
| `co-dino` | co-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull co-dino` |
| `rt-detrv4` | rt-detr | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rt-detrv4` |
| `rtmdet` | rtmdet | Apache-2.0 | Apache-2.0 | False | True | True | `visionservex model pull rtmdet` |

### `byot_license_required` (23)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `sam3-base` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-base --accept-upstream-license` |
| `sam3-image` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-image --accept-upstream-license` |
| `sam3-video` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-video --accept-upstream-license` |
| `sam3-text-prompt` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-text-prompt --accept-upstream-license` |
| `sam3-visual-prompt` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-visual-prompt --accept-upstream-license` |
| `sam3-exemplar-prompt` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-exemplar-prompt --accept-upstream-license` |
| `sam3-open-vocabulary` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-open-vocabulary --accept-upstream-license` |
| `sam3-tracking` | sam3 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3-tracking --accept-upstream-license` |
| `sam3.1-base` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-base --accept-upstream-license` |
| `sam3.1-image` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-image --accept-upstream-license` |
| `sam3.1-video` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-video --accept-upstream-license` |
| `sam3.1-open-vocabulary` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-open-vocabulary --accept-upstream-license` |
| `sam3.1-text-prompt` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-text-prompt --accept-upstream-license` |
| `sam3.1-visual-prompt` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-visual-prompt --accept-upstream-license` |
| `sam3.1-real-time-tracking` | sam3.1 | SAM License (Meta custom, gated) | SAM License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull sam3.1-real-time-tracking --accept-upstream-license` |
| `dinov3-vits16` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-vits16 --accept-upstream-license` |
| `dinov3-vitb16` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-vitb16 --accept-upstream-license` |
| `dinov3-vitl16` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-vitl16 --accept-upstream-license` |
| `dinov3-vit7b16` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-vit7b16 --accept-upstream-license` |
| `dinov3-convnext-tiny` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-convnext-tiny --accept-upstream-license` |
| `dinov3-convnext-small` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-convnext-small --accept-upstream-license` |
| `dinov3-convnext-base` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-convnext-base --accept-upstream-license` |
| `dinov3-convnext-large` | dinov3 | DINOv3 License (Meta custom, gated) | DINOv3 License (Meta custom, gated) | True | False | False | `visionservex hf connect && visionservex model pull dinov3-convnext-large --accept-upstream-license` |

### `external_api_only_terms_required` (9)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `grounding-dino-1.5` | grounding-dino | Apache-2.0 (client SDK only) | not released | False | False | False | `export DDS_API_KEY=... && visionservex model license grounding-dino-1.5` |
| `grounding-dino-1.5-pro` | grounding-dino | Apache-2.0 (client SDK only) | not released | False | False | False | `export DDS_API_KEY=... && visionservex model license grounding-dino-1.5-pro` |
| `grounding-dino-1.6-pro` | grounding-dino | Apache-2.0 (client SDK only) | not released | False | False | False | `export DDS_API_KEY=... && visionservex model license grounding-dino-1.6-pro` |
| `dino-x-api` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-api` |
| `dino-x-detection` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-detection` |
| `dino-x-segmentation` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-segmentation` |
| `dino-x-phrase-grounding` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-phrase-grounding` |
| `dino-x-counting` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-counting` |
| `dino-x-region-captioning` | dino-x | proprietary (hosted API) | not released | False | False | False | `export DINOX_API_KEY=... && visionservex model license dino-x-region-captioning` |

### `noncommercial_restricted` (7)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `edge-sam` | efficient-sam | NTU S-Lab License 1.0 | NTU S-Lab License 1.0 (non-commercial) | False | False | False | `visionservex model license edge-sam  # non-commercial; research-only` |
| `locate-anything-3b` | vlm-grounding | NVIDIA License | NVIDIA License (non-commercial) | True | False | False | `visionservex model license locate-anything-3b  # non-commercial; research-only` |
| `describe-anything-3b` | vlm-grounding | NVIDIA License | NVIDIA License (non-commercial) | True | False | False | `visionservex model license describe-anything-3b  # non-commercial; research-only` |
| `medsam2` | sam | Apache-2.0 (code) | non-commercial (medical dataset provenance) | False | False | False | `visionservex model license medsam2  # non-commercial; research-only` |
| `depth-anything-v2-large` | depth-anything | Apache-2.0 (code) | CC-BY-NC-4.0 | False | False | False | `visionservex model license depth-anything-v2-large  # non-commercial; research-only` |
| `simpleclick` | interactive-seg | MIT (code) | encumbered (MAE CC-BY-NC backbone) | False | False | False | `visionservex model license simpleclick  # non-commercial; research-only` |
| `focalclick` | interactive-seg | MIT (code) | encumbered (NVIDIA SegFormer backbone, non-commercial) | False | False | False | `visionservex model license focalclick  # non-commercial; research-only` |

### `enterprise_license_required` (4)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `fastsam-s` | fastsam | Apache-2.0 (CASIA repo) | AGPL-3.0 coupling (ultralytics YOLOv8-seg dependency) | False | False | False | `visionservex model license fastsam-s  # enterprise/AGPL license required` |
| `fastsam-x` | fastsam | Apache-2.0 (CASIA repo) | AGPL-3.0 coupling (ultralytics YOLOv8x-seg dependency) | False | False | False | `visionservex model license fastsam-x  # enterprise/AGPL license required` |
| `yolov8-seg` | ultralytics | AGPL-3.0 | AGPL-3.0 | False | False | False | `visionservex model license yolov8-seg  # enterprise/AGPL license required` |
| `yolo11-seg` | ultralytics | AGPL-3.0 | AGPL-3.0 | False | False | False | `visionservex model license yolo11-seg  # enterprise/AGPL license required` |

### `legal_review_required` (11)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `hq-sam` | sam-hq | Apache-2.0 (code) | Apache-2.0 weights / HQSeg-44K dataset partly NC | False | False | False | `visionservex legal review hq-sam` |
| `light-hq-sam` | sam-hq | Apache-2.0 (code) | Apache-2.0 weights / HQSeg-44K dataset partly NC | False | False | False | `visionservex legal review light-hq-sam` |
| `sam-hq2` | sam-hq | Apache-2.0 (code) | Apache-2.0 weights / HQSeg-44K dataset partly NC | False | False | False | `visionservex legal review sam-hq2` |
| `tinysam` | efficient-sam | Apache-2.0 | Apache-2.0 | False | False | False | `visionservex legal review tinysam` |
| `q-tinysam` | efficient-sam | Apache-2.0 | Apache-2.0 | False | False | False | `visionservex legal review q-tinysam` |
| `clickseg` | interactive-seg | MIT (code) | mixed (CDNet/HRNet permissive; FocalClick/SegFormer NC) | False | False | False | `visionservex legal review clickseg` |
| `oneformer` | oneformer | MIT (code) | MIT weights / training-data review | False | False | False | `visionservex legal review oneformer` |
| `internimage` | internimage | MIT (code) | MIT weights / DCNv3 + dataset review | False | False | False | `visionservex legal review internimage` |
| `medsam` | sam | Apache-2.0 (code) | Apache-2.0 weights / medical dataset provenance | False | False | False | `visionservex legal review medsam` |
| `rfdetr-seg-xl` | rf-detr | Apache-2.0 (per v3.7 research) | Apache-2.0 (seg) — verify not PML-1.0 detection-XL | False | False | False | `visionservex legal review rfdetr-seg-xl` |
| `rfdetr-seg-2xl` | rf-detr | Apache-2.0 (per v3.7 research) | Apache-2.0 (seg) — verify not PML-1.0 detection-2XL | False | False | False | `visionservex legal review rfdetr-seg-2xl` |

### `not_released_or_unverifiable` (1)

| model_id | family | code license | weights license | gated | commercial_safe | production | next command |
|---|---|---|---|---|---|---|---|
| `grounding-dino-2` | grounding-dino | unknown | unknown | False | False | False | `visionservex model status grounding-dino-2 --explain` |


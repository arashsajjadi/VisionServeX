# VisionServeX v3.7 — Post-v2.59 Inventory

Total tracked items (models/tools/pipelines/variants added or materially changed since v2.59): **105**

## Current-state distribution

- `benchmark_passed`: 44
- `auth_required`: 23
- `external_api_only`: 10
- `checkpoint_required`: 9
- `legal_review_required`: 8
- `excluded_restricted`: 5
- `blocked_documented`: 4
- `sidecar_required`: 1
- `tool_available`: 1

## Product-grade-status distribution

- `product_grade_pass`: 45
- `auth_required`: 23
- `external_api_only`: 10
- `checkpoint_required`: 8
- `legal_review_required`: 8
- `excluded_restricted`: 5
- `blocked_documented`: 4
- `sidecar_required`: 1
- `product_grade_candidate`: 1

## Family distribution

- sam: 51
- dino: 17
- grounding-dino: 10
- dino-x: 6
- rf-detr: 6
- interactive-seg: 5
- fastsam: 2
- ultralytics: 2
- clip: 1
- owlvit: 1
- owlv2: 1
- LiheYoung/depth-anything-small-hf: 1
- efficient-sam: 1
- vlm-grounding: 1

## Runtime-verified (benchmark_passed) — 44 items, each with on-disk evidence artifact

| item_id | family | task | latency-evidence | artifact_exists |
|---|---|---|---|---|
| sam-vit-b | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/sam-vit-b | True |
| sam-vit-l | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/sam-vit-l | True |
| sam-vit-h | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/sam-vit-h | True |
| sam-vit-b-onnx | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/sam-vit-b | True |
| mobilesam | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/mobilesam | True |
| mobilesam-onnx | sam | promptable_segmentation | notebook/99_final_report/artifacts/v37/mobilesam | True |
| efficientsam-l0 | sam | promptable_segmentation | notebook/99_final_report/artifacts/v35/efficient | True |
| efficientsam-onnx | sam | promptable_segmentation | notebook/99_final_report/artifacts/v35/efficient | True |
| sam2-hiera-tiny | sam | promptable_segmentation |  | False |
| sam2-hiera-small | sam | promptable_segmentation |  | False |
| sam2-hiera-base-plus | sam | promptable_segmentation |  | False |
| sam2-hiera-large | sam | promptable_segmentation |  | False |
| sam2.1-hiera-tiny | sam | promptable_segmentation |  | False |
| sam2.1-hiera-small | sam | promptable_segmentation |  | False |
| sam2.1-hiera-base-plus | sam | promptable_segmentation |  | False |
| sam2.1-hiera-large | sam | promptable_segmentation |  | False |
| sam2.1-video-tiny | sam | promptable_segmentation |  | False |
| sam2.1-video-small | sam | promptable_segmentation |  | False |
| sam2.1-video-base-plus | sam | promptable_segmentation |  | False |
| sam2.1-video-large | sam | promptable_segmentation |  | False |
| medsam | sam | promptable_segmentation |  | False |
| dinov2-vits14 | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-sm | True |
| dinov2-vitb14 | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-ba | True |
| dinov2-vitl14 | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-la | True |
| dinov2-vitg14 | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-gi | True |
| dinov2-small | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-sm | True |
| dinov2-base | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-ba | True |
| dinov2-large | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-la | True |
| dinov2-giant | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dinov2-gi | True |
| dino-vits8 | dino | embed_or_detect | notebook/99_final_report/artifacts/v37/dino_vits | True |
| grounding-dino-tiny | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| grounding-dino-swin-t | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| grounding-dino-swin-b | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| grounding-dino-base | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| grounding-dino-original-swin-t | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| grounding-dino-original-swin-b | grounding-dino | embed_or_detect | notebook/99_final_report/artifacts/v37/gdino_gro | True |
| clip-vit-base-patch32 | clip | image_embedding | notebook/99_final_report/artifacts/v37/clip_vit_ | True |
| owlvit-base-patch32 | owlvit | open_vocab_detect | notebook/99_final_report/artifacts/v37/owlvit_b3 | True |
| owlv2-base-patch16 | owlv2 | open_vocab_detect | notebook/99_final_report/artifacts/v37/owlv2_det | True |
| depth-anything-small-hf | LiheYoung/depth-anything-small-hf | depth | notebook/99_final_report/artifacts/v37/depth_any | True |
| rfdetr-seg-nano | rf-detr | instance_segmentation | notebook/99_final_report/artifacts/v37/rfdetr_se | True |
| rfdetr-seg-small | rf-detr | instance_segmentation | notebook/99_final_report/artifacts/v37/rfdetr_se | True |
| rfdetr-seg-medium | rf-detr | instance_segmentation | notebook/99_final_report/artifacts/v37/rfdetr_se | True |
| rfdetr-seg-large | rf-detr | instance_segmentation | notebook/99_final_report/artifacts/v37/rfdetr_se | True |

## Blocked/restricted (honest states with exact next command)

| item_id | state | license | exact_next_command |
|---|---|---|---|
| sam-vit-l-onnx | checkpoint_required | Apache-2.0 | visionservex pull sam-vit-l-onnx  # BYOT Apache-2.0 checkpoint |
| sam-vit-h-onnx | checkpoint_required | Apache-2.0 | visionservex pull sam-vit-h-onnx  # BYOT Apache-2.0 checkpoint |
| efficientsam-l1 | checkpoint_required | Apache-2.0 | visionservex pull efficientsam-l1  # BYOT Apache-2.0 checkpoint |
| efficientsam-l2 | checkpoint_required | Apache-2.0 | visionservex pull efficientsam-l2  # BYOT Apache-2.0 checkpoint |
| sam2.1-onnx-tiny | blocked_documented | Apache-2.0 | pip install sam2 (isolated env) && python tools/export_image_predictor |
| sam2.1-onnx-small | blocked_documented | Apache-2.0 | pip install sam2 (isolated env) && python tools/export_image_predictor |
| sam2.1-onnx-base-plus | blocked_documented | Apache-2.0 | pip install sam2 (isolated env) && python tools/export_image_predictor |
| sam2.1-onnx-large | blocked_documented | Apache-2.0 | pip install sam2 (isolated env) && python tools/export_image_predictor |
| medsam2 | sidecar_required | Apache-2.0 | visionservex sidecar create medsam2 --execute |
| hq-sam | legal_review_required | Apache-2.0 code / HQSeg-44K NC data | visionservex legal review hq-sam  # HQSeg-44K dataset NC review |
| hq-sam2 | legal_review_required | Apache-2.0 code / HQSeg-44K NC data | visionservex legal review hq-sam2  # HQSeg-44K dataset NC review |
| light-hq-sam | legal_review_required | Apache-2.0 code / HQSeg-44K NC data | visionservex legal review light-hq-sam  # HQSeg-44K dataset NC review |
| tinysam | checkpoint_required | Apache-2.0 | visionservex pull tinysam  # BYOT Apache-2.0 checkpoint |
| q-tinysam | checkpoint_required | Apache-2.0 | visionservex pull q-tinysam  # BYOT Apache-2.0 checkpoint |
| edgesam | excluded_restricted | NTU S-Lab License 1.0 (NON-COMMERCIAL) | # edgesam is non-commercial — excluded from commercial core; negotiate |
| sam3-base | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-base  # request ga |
| sam3-image | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-image  # request g |
| sam3-video | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-video  # request g |
| sam3-text-prompt | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-text-prompt  # req |
| sam3-visual-prompt | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-visual-prompt  # r |
| sam3-exemplar-prompt | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-exemplar-prompt  # |
| sam3-open-vocabulary | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-open-vocabulary  # |
| sam3-tracking | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3-tracking  # reques |
| sam3.1-base | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-base  # request  |
| sam3.1-image | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-image  # request |
| sam3.1-video | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-video  # request |
| sam3.1-real-time-tracking | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-real-time-tracki |
| sam3.1-text-prompt | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-text-prompt  # r |
| sam3.1-visual-prompt | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-visual-prompt  # |
| sam3.1-open-vocabulary | auth_required | SAM License (Meta custom, gated) | export HF_TOKEN=... && visionservex sam status sam3.1-open-vocabulary  |
| dinov3-vits16 | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-vits16  # gated |
| dinov3-vitb16 | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-vitb16  # gated |
| dinov3-vitl16 | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-vitl16  # gated |
| dinov3-vit7b16 | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-vit7b16  # gate |
| dinov3-convnext-tiny | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-convnext-tiny   |
| dinov3-convnext-small | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-convnext-small  |
| dinov3-convnext-base | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-convnext-base   |
| dinov3-convnext-large | auth_required | DINOv3 License (Meta custom, gated) | export HF_TOKEN=... && visionservex dino status dinov3-convnext-large  |
| grounding-dino-1.5 | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api grounding-dino-1.5 i |
| grounding-dino-1.6 | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api grounding-dino-1.6 i |
| grounding-dino-1.5-pro | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api grounding-dino-1.5-p |
| grounding-dino-1.6-pro | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api grounding-dino-1.6-p |
| dino-x-api | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-api image.jpg |
| dino-x-detection | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-detection ima |
| dino-x-segmentation | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-segmentation  |
| dino-x-phrase-grounding | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-phrase-ground |
| dino-x-counting | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-counting imag |
| dino-x-region-captioning | external_api_only | proprietary API (Apache SDK only) | export DINOX_API_KEY=... && visionservex dino api dino-x-region-captio |
| rfdetr-seg-xl | checkpoint_required | Apache-2.0 | visionservex segment-instances image.jpg --model rfdetr-seg-xl --out o |
| rfdetr-seg-2xl | checkpoint_required | Apache-2.0 | visionservex segment-instances image.jpg --model rfdetr-seg-2xl --out  |
| ritm | checkpoint_required | MIT | git clone https://github.com/SamsungLabs/ritm_interactive_segmentation |
| clickseg | legal_review_required | MIT | git clone https://github.com/alibaba/ClickSEG && cd ClickSEG && pip in |
| simpleclick | legal_review_required | MIT | git clone https://github.com/uncbiag/SimpleClick && cd SimpleClick &&  |
| focalclick | legal_review_required | MIT | git clone https://github.com/yzluka/FocalClick && cd FocalClick && pip |
| edgesam | excluded_restricted | NTU S-Lab License 1.0 (NC) | # edgesam: NTU S-Lab License 1.0 (NC) — enterprise/negotiated license  |
| fastsam-s | legal_review_required | ultralytics AGPL-3.0 coupling | # fastsam-s: ultralytics AGPL-3.0 coupling — enterprise/negotiated lic |
| fastsam-x | legal_review_required | ultralytics AGPL-3.0 coupling | # fastsam-x: ultralytics AGPL-3.0 coupling — enterprise/negotiated lic |
| yolov8-seg | excluded_restricted | AGPL-3.0 | # yolov8-seg: AGPL-3.0 — enterprise/negotiated license required |
| yolo11-seg | excluded_restricted | AGPL-3.0 | # yolo11-seg: AGPL-3.0 — enterprise/negotiated license required |
| locateanything-3b | excluded_restricted | NVIDIA License (non-commercial) | # locateanything-3b: NVIDIA License (non-commercial) — enterprise/nego |

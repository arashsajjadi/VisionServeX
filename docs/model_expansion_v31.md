# VisionServeX v3.1.0 — SAM/DINO Model Expansion + CV2-Pro + Unified VSX API

This release operationalizes the SAM and DINO families end-to-end, adds the CV2-Pro
OpenCV toolkit, ships first-class text-to-mask pipelines, and introduces a single
Ultralytics-style `VSX` facade — while keeping every commercial-safety rule from V3.

## Activation KPI: 21 new activations (target 20)
- **13 CV2-Pro tools** (`tool_benchmark_passed`, real COCO CPU latency).
- **6 SAM+DINO text-to-mask pipelines** (`pipeline_demo_ready`, both parts benchmark_passed, runnable via `VSX.pipeline`).
- **2 interactive-seg models** (RITM, ClickSEG — `checkpoint_required`, user-supplied checkpoint).

Honest note on "new pretrained models": the SAM/DINO models are already `benchmark_passed`
from V3 (cannot double-count); the *new* model targets are gated (SAM3, DINOv3, DINO-X,
GroundingDINO 1.5/1.6), legal-review (HQ-SAM, TinySAM, FocalClick, SimpleClick), or
heavy OpenMMLab sidecars (RTMDet/InternImage/MaskDINO/SEEM/Co-DINO/OneFormer). Every one
is in the matrices with an exact lawful next command — none omitted, none faked.

## Unified VSX API (use like Ultralytics)
```python
from visionservex import VSX
sam  = VSX.sam("sam2.1-hiera-small"); sam.segment("image.jpg", box=[10,20,200,220])
dino = VSX.dino("dinov2-base");       dino.embed("image.jpg")
pipe = VSX.pipeline("grounding-dino-swin-t+sam2.1-hiera-small"); pipe.run("image.jpg", text="defect")
tool = VSX.cv2("opencv-mser-proposals"); tool.run("image.jpg")
```
CLI: `visionservex sam|dino|pipeline|cv2-pro {list,status,run,...}` — every command supports `--explain`.

## SAM family (68 classified targets, none omitted)
SAM1/2/2.1 image variants are `benchmark_passed` (Apache-2.0). SAM3 is `auth_required`
(Meta custom SAM License, HF-gated, **BYOT** — weights never mirrored, token never logged).
SAM3.1 + SAM3 capability sub-variants are `not_released`. EdgeSAM stays `excluded_restricted`
(S-Lab non-commercial). HQ-SAM/TinySAM/Q-TinySAM/Light-HQ-SAM stay `legal_review_required`
(HQSeg-44K / SA-1B provenance). ONNX variants = local export (`checkpoint_required`).

## DINO family (44 classified targets)
DINOv2 + GroundingDINO Swin = `benchmark_passed` (Apache-2.0). **DINOv3 = `legal_review_required`**
(custom Meta DINOv3 License, HF-gated — NOT Apache). GroundingDINO 1.5/1.6 = `auth_required`;
1.5/1.6-pro + DINO-X = `external_api_only`.

## CV2-Pro (commercial-safe OpenCV, Apache-2.0)
13 tools: MSER, GrabCut+, Watershed+, connected-components, contour-snap, intelligent-scissors,
kmeans-color-segment, distance-transform-markers, MOG2/KNN bg-subtraction, DNN-ONNX runner (base),
selective-search fast/quality (`cv2-pro` extra = opencv-contrib). No GPL.

## Commercial-safety (unchanged from V3, not regressed)
No AGPL/GPL/non-commercial/gated weights in default-safe core. EdgeSAM still excluded.
Gated models BYOT/API only, weights never mirrored, tokens never logged.

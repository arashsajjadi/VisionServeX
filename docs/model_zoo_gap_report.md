# VisionServeX Model Zoo Gap Report

## Runnable (wired, usable now) (30)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `dfine-s-o365-coco` | dfine | detect | Apache-2.0 | - |
| `dfine-x-o365-coco` | dfine | detect | Apache-2.0 | - |
| `dfine-l-o365-coco` | dfine | detect | Apache-2.0 | - |
| `rfdetr-small` | rfdetr | detect | Apache-2.0 | - |
| `rfdetr-large` | rfdetr | detect | Apache-2.0 | - |
| `rfdetr-seg-medium` | rfdetr | segment | Apache-2.0 | - |
| `sam-vit-base` | sam | foundation_segment | Apache-2.0 | - |
| `sam2-hiera-tiny` | sam2 | foundation_segment | Apache-2.0 | - |
| `dinov2-small` | dinov2 | embed | Apache-2.0 | - |
| `dinov2-base` | dinov2 | embed | Apache-2.0 | - |
| `dinov2-large` | dinov2 | embed | Apache-2.0 | - |
| `dinov2-giant` | dinov2 | embed | Apache-2.0 | - |
| `florence-2-base` | florence-2 | vlm | MIT | - |
| `florence-2-large` | florence-2 | vlm | MIT | - |
| `owlv2-base-patch16` | owlv2 | open_vocab_detect | Apache-2.0 | - |
| `owlv2-large-patch14` | owlv2 | open_vocab_detect | Apache-2.0 | - |
| `siglip2-base-patch16-224` | siglip2 | embed | Apache-2.0 | - |
| `medsam` | medsam | foundation_segment | Apache-2.0 | - |
| `owlvit-base-patch32` | owlvit | open_vocab_detect | Apache-2.0 | - |
| `owlvit-large-patch14` | owlvit | open_vocab_detect | Apache-2.0 | - |
| `clip-vit-base-patch32` | clip | embed | MIT | - |
| `clip-vit-large-patch14` | clip | embed | MIT | - |
| `siglip-base-patch16-224` | siglip | embed | Apache-2.0 | - |
| `convnextv2-tiny` | convnextv2 | classify | Apache-2.0 | - |
| `convnextv2-base` | convnextv2 | classify | Apache-2.0 | - |
| `convnextv2-large` | convnextv2 | classify | Apache-2.0 | - |
| `sam2.1-hiera-tiny` | sam2.1 | foundation_segment | Apache-2.0 | - |
| `sam2.1-hiera-small` | sam2.1 | foundation_segment | Apache-2.0 | - |
| `sam2.1-hiera-base-plus` | sam2.1 | foundation_segment | Apache-2.0 | - |
| `sam2.1-hiera-large` | sam2.1 | foundation_segment | Apache-2.0 | - |

## Optional extras (require extra install) (0)

_None_

## Expert sidecars (OpenMMLab, Detectron2, etc.) (12)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `co-dino-inst-vit-l-coco` | co-dino | segment | Apache-2.0 | Requires OpenMMLab MMDetection sidecar.; OpenMMLab not installed by default in core. |
| `maskdino-swinl-coco` | maskdino | segment | Apache-2.0 | Requires Detectron2 environment.; No pip-installable inference path. |
| `anomalib-patchcore` | anomalib | anomaly | Apache-2.0 | anomalib is heavy; kept outside core dependencies.; Pipeline integration not yet wired. |
| `osnet-x1.0` | osnet | reid | MIT | torchreid not in core dependencies. |
| `bytetrack` | bytetrack | track | MIT | ByteTrack not packaged; tracker pipeline pending. |
| `medsam2` | medsam2 | foundation_segment | Apache-2.0 | - |
| `nnunet-v2` | nnunet | segment | Apache-2.0 | Training/inference framework, not a single model. |
| `rtmdet-r2-s` | rtmdet | obb | Apache-2.0 | Requires OpenMMLab MMRotate sidecar. |
| `mobilesam` | mobilesam | foundation_segment | Apache-2.0 | No HF Hub model card for pip-installable weights. Checkpoint distributed via GitHub release.; Not yet wired in VisionServeX engine; manual download + sam_hf adapter path needed. |
| `efficientsam` | efficientsam | foundation_segment | Apache-2.0 | No HF Hub model; GitHub-only install.; Not yet wired in VisionServeX engine. |
| `hq-sam` | hq-sam | foundation_segment | Apache-2.0 | No HF Hub model with standard SAM API compatibility confirmed.; Not yet wired in VisionServeX engine. |
| `edgesam` | edgesam | foundation_segment | Apache-2.0 | GitHub-only install; no HF Hub model.; Targets edge/mobile devices; not prioritized for server deployment. |

## External / gated APIs (4)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `sam3-base` | sam3 | foundation_segment | Apache-2.0 | Access may be gated at the HF namespace.; VisionServeX does not auto-pull gated weights. |
| `grounding-dino-1.5-pro` | grounding-dino | open_vocab_detect | Custom | API-gated. Requires IDEA-Research token. |
| `grounding-dino-1.6-pro` | grounding-dino | open_vocab_detect | Custom | API-gated. Requires IDEA-Research token. |
| `dino-x-api` | dino-x | open_vocab_detect | Custom | - |

## Non-core license (optional) (2)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `totalsegmentator` | totalsegmentator | segment | Apache-2.0 | Heavy dependency on nibabel, SimpleITK, etc.; License has commercial restrictions on some submodels. |
| `prithvi-eo-2.0` | prithvi | embed | Apache-2.0 | Multispectral input — not standard RGB.; Geospatial extra; not in core. |

## Excluded (do_not_add with reason) (3)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `yolo-world` | yolo-world | open_vocab_detect | check | License likely GPL/AGPL — excluded from permissive Apache/MIT core.; Use Grounding DINO or OWLv2 instead. |
| `fastsam-s` | fastsam | foundation_segment | AGPL-3.0 | License is AGPL-3.0. VisionServeX core requires Apache-2.0/MIT-compatible licenses.; AGPL-3.0 requires downstream users to open-source their applications if served over a network. |
| `fastsam-x` | fastsam | foundation_segment | AGPL-3.0 | License AGPL-3.0 — excluded from core. |

## Audit only (no blockers yet) (0)

_None_

## Unresolved blockers (8)

| Model ID | Family | Task | License | Blockers |
|----------|--------|------|---------|---------|
| `rfdetr-seg-large` | rfdetr | segment | Apache-2.0 | HF checkpoint roboflow/rf-detr-seg-large not yet published at last audit.; Verify via rfdetr.utils or roboflow docs before enabling. |
| `deimv2-s` | deimv2 | detect | Apache-2.0 | Not yet supported by HF Transformers (issue #41211).; Requires native PyTorch loader from official repo. |
| `deimv2-m` | deimv2 | detect | Apache-2.0 | Same as deimv2-s — repo native loader required. |
| `deimv2-l` | deimv2 | detect | Apache-2.0 | Heavy model. Native loader required. |
| `deimv2-x` | deimv2 | detect | Apache-2.0 | Heaviest DEIMv2. Native loader required. |
| `rtdetrv4-s` | rtdetrv4 | detect | Apache-2.0 | RT-DETRv4 release/checkpoint URLs not officially numbered.; Known TensorRT/RTX 5080 issues reported. |
| `dinov3-vitb16` | dinov3 | embed | Apache-2.0 | DINOv3 HF model card name not verified live.; Use DINOv2 until DINOv3 access is confirmed. |
| `agriclip` | agriclip | embed | check | License and HF availability not verified live. |

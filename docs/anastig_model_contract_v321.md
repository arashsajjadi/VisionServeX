# Anastig Model Contract (v3.21)

Source of truth: `visionservex.model_capabilities(model_id)` (v3.23.0).
The **primary buckets** are a disjoint partition of all models; the **view buckets**
are overlapping live sub-views. v3.21 adds the **sidecar dimension**: models live only
through an isolated Docker sidecar get `*_ready_live_sidecar` primary buckets and the
`sidecar_required_live` view. Anastig drives UI from `anastig_visibility` /
`anastig_train_visibility` / `anastig_finetune_visibility` / `anastig_sidecar_visibility`.

## v3.21 capability fields
- `sidecar_supported` / `sidecar_required` / `sidecar_name` — isolated-sidecar routing.
- `sidecar_live` / `sidecar_cpu_verified` / `sidecar_gpu_verified` — sidecar live-proof.
- `anastig_sidecar_visibility` — `show_*_sidecar` / `host_preferred_sidecar_available` / `none`.
- `fine_tune_kind` — `full_supervised` / `frozen_backbone_head` / `frozen_encoder_decoder` / `none`.

## Buckets + UI copy

| Bucket | Count | UI copy |
|---|---:|---|
| `train_ready_live` | 24 | Train ready |
| `inference_ready_live` | 41 | Inference only |
| `embedding_fine_tune_ready_live` | 10 | Fine-tune ready (embedding head) |
| `foundation_segment_inference_only_live` | 15 | Inference only (promptable segmentation) |
| `open_vocab_ready_live` | 10 | Inference only (open-vocabulary) |
| `vlm_ready_live` | 0 | Inference only (VLM) |
| `pose_ready_live` | 1 | Inference only (pose) |
| `obb_ready_live` | 1 | Inference only (oriented boxes) |
| `vlm_ready_live_sidecar` | 2 | Inference only (VLM, Docker sidecar) |
| `inference_ready_live_sidecar` | 1 | Inference only (Docker sidecar) |
| `segmentation_ready_live_sidecar` | 0 | Inference only (segmentation, Docker sidecar) |
| `gated_token_required` | 1 | Needs token |
| `hidden_catalog_only` | 26 | Hidden: engine not wired |
| `hidden_custom_loader_required` | 15 | Hidden: custom loader required |
| `blocked_dependency` | 1 | Admin only: dependency missing |
| `blocked_weights` | 1 | Hidden: weights unavailable |
| `blocked_license` | 0 | Hidden: legal/license |
| `blocked_partial` | 2 | Hidden: partial implementation |
| `blocked_oom` | 0 | Admin only: out of memory |
| `blocked_upstream` | 0 | Admin only: upstream crash |
| `unknown_review_required` | 0 | Admin only: review required |
| `fine_tune_ready_live` | 11 | Fine-tune ready |
| `sidecar_required_live` | 3 | Live via Docker sidecar |
| `sam_decoder_fine_tune_ready` | 4 | Fine-tune ready (SAM mask decoder) |
| `classification_train_ready_live` | 13 |  |
| `detection_train_ready_live` | 8 |  |
| `segmentation_train_ready_live` | 3 |  |

## Members

### `train_ready_live` (24)

`libreyolo-rtdetr-r50`, `libreyolo-yolov9-s`, `libreyolo-yolox-s`, `rfdetr-base`, `rfdetr-large`, `rfdetr-medium`, `rfdetr-nano`, `rfdetr-seg-medium`, `rfdetr-seg-nano`, `rfdetr-seg-small`, `rfdetr-small`, `torchvision-alexnet`, `torchvision-convnext-tiny`, `torchvision-densenet121`, `torchvision-efficientnet-b0`, `torchvision-mobilenet-v2`, `torchvision-mobilenet-v3-large`, `torchvision-resnet101`, `torchvision-resnet152`, `torchvision-resnet18`, `torchvision-resnet34`, `torchvision-resnet50`, `torchvision-resnext50-32x4d`, `torchvision-wide-resnet50-2`

### `inference_ready_live` (41)

`convnextv2-base`, `convnextv2-large`, `convnextv2-tiny`, `dfine-l`, `dfine-l-coco`, `dfine-l-o365-coco`, `dfine-m`, `dfine-m-coco`, `dfine-m-o365-coco`, `dfine-n`, `dfine-n-coco`, `dfine-s`, `dfine-s-coco`, `dfine-s-o365-coco`, `dfine-x`, `dfine-x-coco`, `dfine-x-o365-coco`, `grounded-sam`, `grounded-sam2`, `libreyolo-dfine-l`, `libreyolo-dfine-m`, `libreyolo-dfine-n`, `libreyolo-dfine-s`, `libreyolo-dfine-x`, `libreyolo-rtdetr-r101`, `libreyolo-yolov9-c`, `libreyolo-yolov9-m`, `libreyolo-yolox-l`, `libreyolo-yolox-m`, `libreyolo-yolox-x`, `maxvit-tiny-tf-224`, `mock-classify`, `mock-detect`, `mock-foundation-segment`, `mock-grounded-segment`, `mock-segment`, `oneformer-swin-large`, `swinv2-base`, `swinv2-large`, `swinv2-small`, `swinv2-tiny`

### `embedding_fine_tune_ready_live` (10)

`clip-vit-base-patch32`, `clip-vit-large-patch14`, `dinov2-base`, `dinov2-giant`, `dinov2-large`, `dinov2-small`, `siglip-base-patch16-224`, `siglip2-base-patch16-224`, `siglip2-large-patch16-256`, `siglip2-so400m-patch14-384`

### `foundation_segment_inference_only_live` (15)

`efficientsam`, `hq-sam`, `medsam`, `mobilesam`, `sam-vit-base`, `sam-vit-huge`, `sam-vit-large`, `sam2-hiera-base-plus`, `sam2-hiera-large`, `sam2-hiera-small`, `sam2-hiera-tiny`, `sam2.1-hiera-base-plus`, `sam2.1-hiera-large`, `sam2.1-hiera-small`, `sam2.1-hiera-tiny`

### `open_vocab_ready_live` (10)

`grounding-dino-original-swin-b`, `grounding-dino-original-swin-t`, `grounding-dino-swin-b`, `grounding-dino-swin-t`, `grounding-dino-tiny`, `mock-open-vocab`, `owlv2-base-patch16`, `owlv2-large-patch14`, `owlvit-base-patch32`, `owlvit-large-patch14`

### `vlm_ready_live` (0)

_(none)_

### `pose_ready_live` (1)

`mock-pose`

### `obb_ready_live` (1)

`mock-obb`

### `vlm_ready_live_sidecar` (2)

`florence-2-base`, `florence-2-large`

### `inference_ready_live_sidecar` (1)

`rtmpose-m`

### `segmentation_ready_live_sidecar` (0)

_(none)_

### `gated_token_required` (1)

`sam3-base`

### `hidden_catalog_only` (26)

`co-dino-inst-vit-l-coco`, `co-dino-inst-vit-l-lvis`, `grounding-dino-1.5`, `grounding-dino-1.6`, `internimage-b`, `internimage-h`, `internimage-l`, `internimage-s`, `internimage-t`, `rfdetr-seg-2xlarge`, `rfdetr-seg-large`, `rfdetr-seg-xlarge`, `rtdetrv4-s`, `rtmdet-r-l`, `rtmdet-r-m`, `rtmdet-r-s`, `rtmdet-r-t`, `rtmdet-r2-l`, `rtmdet-r2-m`, `rtmdet-r2-t`, `rtmpose-l`, `rtmpose-l-384x288`, `rtmpose-m-384x288`, `rtmpose-t`, `seem-davit-d3`, `seem-focal-t`

### `hidden_custom_loader_required` (15)

`deim-m`, `deim-s`, `deimv2-atto`, `deimv2-femto`, `deimv2-l`, `deimv2-m`, `deimv2-n`, `deimv2-pico`, `deimv2-s`, `deimv2-x`, `maskdino-r50-coco`, `maskdino-r50-panoptic`, `rtdetrv4-l`, `rtdetrv4-m`, `rtdetrv4-x`

### `blocked_dependency` (1)

`oneformer-dinat-large`

### `blocked_weights` (1)

`oneformer-convnext-large`

### `blocked_license` (0)

_(none)_

### `blocked_partial` (2)

`rtmdet-r2-s`, `rtmpose-s`

### `blocked_oom` (0)

_(none)_

### `blocked_upstream` (0)

_(none)_

### `unknown_review_required` (0)

_(none)_

### `fine_tune_ready_live` (11)

`clip-vit-base-patch32`, `clip-vit-large-patch14`, `dinov2-base`, `dinov2-giant`, `dinov2-large`, `dinov2-small`, `sam-vit-base`, `siglip-base-patch16-224`, `siglip2-base-patch16-224`, `siglip2-large-patch16-256`, `siglip2-so400m-patch14-384`

### `sidecar_required_live` (3)

`florence-2-base`, `florence-2-large`, `rtmpose-m`

### `sam_decoder_fine_tune_ready` (4)

`medsam`, `sam-vit-base`, `sam-vit-huge`, `sam-vit-large`

### `classification_train_ready_live` (13)

`torchvision-alexnet`, `torchvision-convnext-tiny`, `torchvision-densenet121`, `torchvision-efficientnet-b0`, `torchvision-mobilenet-v2`, `torchvision-mobilenet-v3-large`, `torchvision-resnet101`, `torchvision-resnet152`, `torchvision-resnet18`, `torchvision-resnet34`, `torchvision-resnet50`, `torchvision-resnext50-32x4d`, `torchvision-wide-resnet50-2`

### `detection_train_ready_live` (8)

`libreyolo-rtdetr-r50`, `libreyolo-yolov9-s`, `libreyolo-yolox-s`, `rfdetr-base`, `rfdetr-large`, `rfdetr-medium`, `rfdetr-nano`, `rfdetr-small`

### `segmentation_train_ready_live` (3)

`rfdetr-seg-medium`, `rfdetr-seg-nano`, `rfdetr-seg-small`


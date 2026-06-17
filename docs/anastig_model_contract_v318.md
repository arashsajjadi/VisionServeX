# Anastig Model Contract (v3.18)

Source of truth: `visionservex.model_capabilities(model_id)` (v3.19.0).
Anastig drives its entire model UI from this contract — there is **no hardcoded
model allowlist** required beyond optional product ranking. The machine-readable
buckets live in `docs/anastig_model_allowlist_v318.json`.

## How Anastig must consume each field

Every model exposes `readiness_state` and `anastig_visibility`. Anastig switches
on `anastig_visibility`:

| anastig_visibility | Anastig behaviour |
|---|---|
| `show_train` | Show train **and** inference UI (state `TRAIN_READY_LIVE`). |
| `show_inference` | Show inference UI (`INFERENCE_READY_LIVE` / `OPEN_VOCAB_READY_LIVE` / `VLM_READY_LIVE`). |
| `show_embedding` | Show embedding UI (`EMBEDDING_READY_LIVE`). |
| `show_segmentation` | Show segmentation UI (`SEGMENTATION_READY_LIVE`). |
| `show_token_required` | Show BYOT/token UI (`GATED_TOKEN_REQUIRED`) — never run without the user's token. |
| `hide` | Hide entirely (catalog-only, custom-loader, derived-not-live, partial, dependency/weights/crash/oom). |
| `blocked_admin_only` | Hide from end users; visible to admins (legal-review, license-blocked, unknown). |

**Hard rule:** only `*_READY_LIVE` states are usable-by-default. A
`*_DERIVED_NEEDS_LIVE_CONFIRMATION` model is capability-derived and **not yet
live-verified** — Anastig must keep it hidden until it is promoted to `*_LIVE`.

## Bucket counts

| Bucket | Count |
|---|---:|
| `train_ready_live` | 24 |
| `inference_ready_live` | 37 |
| `embedding_ready_live` | 10 |
| `segmentation_ready_live` | 21 |
| `open_vocab_ready_live` | 10 |
| `gated_token_required` | 1 |
| `hidden_catalog_only` | 43 |
| `blocked` | 5 |
| `license_blocked` | 0 |

## Buckets

### `train_ready_live` (24)

`libreyolo-rtdetr-r50`, `libreyolo-yolov9-s`, `libreyolo-yolox-s`, `rfdetr-base`, `rfdetr-large`, `rfdetr-medium`, `rfdetr-nano`, `rfdetr-seg-medium`, `rfdetr-seg-nano`, `rfdetr-seg-small`, `rfdetr-small`, `torchvision-alexnet`, `torchvision-convnext-tiny`, `torchvision-densenet121`, `torchvision-efficientnet-b0`, `torchvision-mobilenet-v2`, `torchvision-mobilenet-v3-large`, `torchvision-resnet101`, `torchvision-resnet152`, `torchvision-resnet18`, `torchvision-resnet34`, `torchvision-resnet50`, `torchvision-resnext50-32x4d`, `torchvision-wide-resnet50-2`

### `inference_ready_live` (37)

`convnextv2-base`, `convnextv2-large`, `convnextv2-tiny`, `dfine-l`, `dfine-l-coco`, `dfine-l-o365-coco`, `dfine-m`, `dfine-m-coco`, `dfine-m-o365-coco`, `dfine-n`, `dfine-n-coco`, `dfine-s`, `dfine-s-coco`, `dfine-s-o365-coco`, `dfine-x`, `dfine-x-coco`, `dfine-x-o365-coco`, `libreyolo-dfine-l`, `libreyolo-dfine-m`, `libreyolo-dfine-n`, `libreyolo-dfine-s`, `libreyolo-dfine-x`, `libreyolo-rtdetr-r101`, `libreyolo-yolov9-c`, `libreyolo-yolov9-m`, `libreyolo-yolox-l`, `libreyolo-yolox-m`, `libreyolo-yolox-x`, `maxvit-tiny-tf-224`, `mock-classify`, `mock-detect`, `mock-obb`, `mock-pose`, `swinv2-base`, `swinv2-large`, `swinv2-small`, `swinv2-tiny`

### `embedding_ready_live` (10)

`clip-vit-base-patch32`, `clip-vit-large-patch14`, `dinov2-base`, `dinov2-giant`, `dinov2-large`, `dinov2-small`, `siglip-base-patch16-224`, `siglip2-base-patch16-224`, `siglip2-large-patch16-256`, `siglip2-so400m-patch14-384`

### `segmentation_ready_live` (21)

`efficientsam`, `grounded-sam`, `grounded-sam2`, `hq-sam`, `medsam`, `mobilesam`, `mock-foundation-segment`, `mock-grounded-segment`, `mock-segment`, `oneformer-swin-large`, `sam-vit-base`, `sam-vit-huge`, `sam-vit-large`, `sam2-hiera-base-plus`, `sam2-hiera-large`, `sam2-hiera-small`, `sam2-hiera-tiny`, `sam2.1-hiera-base-plus`, `sam2.1-hiera-large`, `sam2.1-hiera-small`, `sam2.1-hiera-tiny`

### `open_vocab_ready_live` (10)

`grounding-dino-original-swin-b`, `grounding-dino-original-swin-t`, `grounding-dino-swin-b`, `grounding-dino-swin-t`, `grounding-dino-tiny`, `mock-open-vocab`, `owlv2-base-patch16`, `owlv2-large-patch14`, `owlvit-base-patch32`, `owlvit-large-patch14`

### `gated_token_required` (1)

`sam3-base`

### `hidden_catalog_only` (43)

`co-dino-inst-vit-l-coco`, `co-dino-inst-vit-l-lvis`, `deim-m`, `deim-s`, `deimv2-atto`, `deimv2-femto`, `deimv2-l`, `deimv2-m`, `deimv2-n`, `deimv2-pico`, `deimv2-s`, `deimv2-x`, `grounding-dino-1.5`, `grounding-dino-1.6`, `internimage-b`, `internimage-h`, `internimage-l`, `internimage-s`, `internimage-t`, `maskdino-r50-coco`, `maskdino-r50-panoptic`, `oneformer-convnext-large`, `rfdetr-seg-2xlarge`, `rfdetr-seg-large`, `rfdetr-seg-xlarge`, `rtdetrv4-l`, `rtdetrv4-m`, `rtdetrv4-s`, `rtdetrv4-x`, `rtmdet-r-l`, `rtmdet-r-m`, `rtmdet-r-s`, `rtmdet-r-t`, `rtmdet-r2-l`, `rtmdet-r2-m`, `rtmdet-r2-t`, `rtmpose-l`, `rtmpose-l-384x288`, `rtmpose-m`, `rtmpose-m-384x288`, `rtmpose-t`, `seem-davit-d3`, `seem-focal-t`

### `blocked` (5)

`florence-2-base`, `florence-2-large`, `oneformer-dinat-large`, `rtmdet-r2-s`, `rtmpose-s`

### `license_blocked` (0)

_(none)_


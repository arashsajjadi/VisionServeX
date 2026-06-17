# v3.19 Baseline (frozen v3.18.0 held state) — 151 models

Snapshot BEFORE the operationalize sprint. Promotions must show real evidence vs this.

| readiness_state | count |
|---|---:|
| `INFERENCE_READY_LIVE` | 36 |
| `CATALOG_ONLY_ENGINE_NOT_WIRED` | 27 |
| `SEGMENTATION_READY_LIVE` | 21 |
| `TRAIN_READY_LIVE` | 16 |
| `CUSTOM_LOADER_REQUIRED` | 15 |
| `EMBEDDING_READY_LIVE` | 10 |
| `OPEN_VOCAB_READY_LIVE` | 10 |
| `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | 8 |
| `DEPENDENCY_MISSING` | 3 |
| `PARTIAL_IMPLEMENTATION_BLOCKED` | 3 |
| `WEIGHTS_MISSING` | 1 |
| `GATED_TOKEN_REQUIRED` | 1 |

| Model | Task | Family | readiness_state | live_inf | live_train | visibility | blocker |
|---|---|---|---|:-:|:-:|---|---|
| `co-dino-inst-vit-l-coco` | segment | co-dino | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `co-dino-inst-vit-l-lvis` | segment | co-dino | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `grounding-dino-1.5` | open_vocab_detect | grounding-dino | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'grounding_dino' not wired (i |
| `grounding-dino-1.6` | open_vocab_detect | grounding-dino | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'grounding_dino' not wired (i |
| `internimage-b` | classify | internimage | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `internimage-h` | classify | internimage | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `internimage-l` | classify | internimage | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `internimage-s` | classify | internimage | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `internimage-t` | classify | internimage | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rfdetr-seg-2xlarge` | segment | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | HF checkpoint roboflow/rf-detr-seg-2xlarge not yet |
| `rfdetr-seg-large` | segment | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | HF checkpoint roboflow/rf-detr-seg-large not yet a |
| `rfdetr-seg-xlarge` | segment | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | HF checkpoint roboflow/rf-detr-seg-xlarge not yet  |
| `rtdetrv4-s` | detect | rtdetrv4 | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | RT-DETRv4 is not an officially released numbered v |
| `rtmdet-r-l` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r-m` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r-s` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r-t` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r2-l` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r2-m` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmdet-r2-t` | obb | rtmdet | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmpose-l` | pose | rtmpose | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmpose-l-384x288` | pose | rtmpose | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmpose-m` | pose | rtmpose | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmpose-m-384x288` | pose | rtmpose | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `rtmpose-t` | pose | rtmpose | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'openmmlab' not wired (implem |
| `seem-davit-d3` | grounded_segment | seem | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'huggingface' not wired (impl |
| `seem-focal-t` | grounded_segment | seem | `CATALOG_ONLY_ENGINE_NOT_WIRED` | - | - | hide | Catalog-only: engine 'huggingface' not wired (impl |
| `deim-m` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | No HF or pip-installable path verified. Custom loa |
| `deim-s` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | No HF or pip-installable path verified. Requires c |
| `deimv2-atto` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 atto custom loader required from official r |
| `deimv2-femto` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 femto custom loader required from official  |
| `deimv2-l` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 large custom loader required from official  |
| `deimv2-m` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Custom loader required from official repo. Checkpo |
| `deimv2-n` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 nano custom loader required from official r |
| `deimv2-pico` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 pico custom loader required from official r |
| `deimv2-s` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 repo and checkpoint availability not indepe |
| `deimv2-x` | detect | deim | `CUSTOM_LOADER_REQUIRED` | - | - | hide | DEIMv2 XLarge custom loader required from official |
| `maskdino-r50-coco` | segment | maskdino | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Requires detectron2 + manual checkpoint. No HF pat |
| `maskdino-r50-panoptic` | segment | maskdino | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Requires detectron2 + manual checkpoint. Custom lo |
| `rtdetrv4-l` | detect | rtdetrv4 | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Custom loader and verified checkpoint source requi |
| `rtdetrv4-m` | detect | rtdetrv4 | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Custom loader and verified checkpoint source requi |
| `rtdetrv4-x` | detect | rtdetrv4 | `CUSTOM_LOADER_REQUIRED` | - | - | hide | Custom loader and verified checkpoint source requi |
| `florence-2-base` | vlm | florence-2 | `DEPENDENCY_MISSING` | - | - | hide | A required optional dependency is not installed. |
| `florence-2-large` | vlm | florence-2 | `DEPENDENCY_MISSING` | - | - | hide | A required optional dependency is not installed. |
| `oneformer-dinat-large` | segment | oneformer | `DEPENDENCY_MISSING` | - | - | hide | A required optional dependency is not installed. |
| `clip-vit-base-patch32` | embed | clip | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `clip-vit-large-patch14` | embed | clip | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `dinov2-base` | embed | dinov2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `dinov2-giant` | embed | dinov2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `dinov2-large` | embed | dinov2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `dinov2-small` | embed | dinov2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `siglip-base-patch16-224` | embed | siglip | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `siglip2-base-patch16-224` | embed | siglip2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `siglip2-large-patch16-256` | embed | siglip2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `siglip2-so400m-patch14-384` | embed | siglip2 | `EMBEDDING_READY_LIVE` | Y | - | show_embedding |  |
| `sam3-base` | foundation_segment | sam3 | `GATED_TOKEN_REQUIRED` | - | - | show_token_required | Gated: requires your own Hugging Face token and ac |
| `convnextv2-base` | classify | convnextv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `convnextv2-large` | classify | convnextv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `convnextv2-tiny` | classify | convnextv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-l` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-l-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-l-o365-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-m` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-m-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-m-o365-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-n` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-n-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-s` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-s-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-s-o365-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-x` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-x-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `dfine-x-o365-coco` | detect | dfine | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-dfine-l` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-dfine-m` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-dfine-n` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-dfine-s` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-dfine-x` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-rtdetr-r101` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-yolov9-c` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-yolov9-m` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-yolox-l` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-yolox-m` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `libreyolo-yolox-x` | detect | libreyolo | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `mock-classify` | classify | mock | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `mock-detect` | detect | mock | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `mock-obb` | obb | mock | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `mock-pose` | pose | mock | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `swinv2-base` | classify | swinv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `swinv2-large` | classify | swinv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `swinv2-small` | classify | swinv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `swinv2-tiny` | classify | swinv2 | `INFERENCE_READY_LIVE` | Y | - | show_inference |  |
| `grounding-dino-original-swin-b` | open_vocab_detect | grounding-dino | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `grounding-dino-original-swin-t` | open_vocab_detect | grounding-dino | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `grounding-dino-swin-b` | open_vocab_detect | grounding-dino | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `grounding-dino-swin-t` | open_vocab_detect | grounding-dino | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `grounding-dino-tiny` | open_vocab_detect | grounding-dino | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `mock-open-vocab` | open_vocab_detect | mock | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `owlv2-base-patch16` | open_vocab_detect | owlv2 | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `owlv2-large-patch14` | open_vocab_detect | owlv2 | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `owlvit-base-patch32` | open_vocab_detect | owlvit | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `owlvit-large-patch14` | open_vocab_detect | owlvit | `OPEN_VOCAB_READY_LIVE` | Y | - | show_inference |  |
| `maxvit-tiny-tf-224` | classify | maxvit | `PARTIAL_IMPLEMENTATION_BLOCKED` | - | - | hide | Partial implementation (implementation_status=part |
| `rtmdet-r2-s` | obb | rtmdet | `PARTIAL_IMPLEMENTATION_BLOCKED` | - | - | hide | Partial implementation (implementation_status=part |
| `rtmpose-s` | pose | rtmpose | `PARTIAL_IMPLEMENTATION_BLOCKED` | - | - | hide | Partial implementation (implementation_status=part |
| `efficientsam` | foundation_segment | efficientsam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `grounded-sam` | grounded_segment | grounded-sam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `grounded-sam2` | grounded_segment | grounded-sam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `hq-sam` | foundation_segment | hq-sam | `SEGMENTATION_READY_LIVE` | Y | - | blocked_admin_only |  |
| `medsam` | foundation_segment | sam | `SEGMENTATION_READY_LIVE` | Y | - | blocked_admin_only |  |
| `mobilesam` | foundation_segment | mobilesam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `mock-foundation-segment` | foundation_segment | mock | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `mock-grounded-segment` | grounded_segment | mock | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `mock-segment` | segment | mock | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `oneformer-swin-large` | segment | oneformer | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam-vit-base` | foundation_segment | sam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam-vit-huge` | foundation_segment | sam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam-vit-large` | foundation_segment | sam | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2-hiera-base-plus` | foundation_segment | sam2 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2-hiera-large` | foundation_segment | sam2 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2-hiera-small` | foundation_segment | sam2 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2-hiera-tiny` | foundation_segment | sam2 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2.1-hiera-base-plus` | foundation_segment | sam2.1 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2.1-hiera-large` | foundation_segment | sam2.1 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2.1-hiera-small` | foundation_segment | sam2.1 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `sam2.1-hiera-tiny` | foundation_segment | sam2.1 | `SEGMENTATION_READY_LIVE` | Y | - | show_segmentation |  |
| `rfdetr-base` | detect | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_inference | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-large` | detect | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_inference | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-medium` | detect | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_inference | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-nano` | detect | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_inference | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-seg-medium` | segment | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_segmentation | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-seg-nano` | segment | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_segmentation | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-seg-small` | segment | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_segmentation | Train lifecycle is capability-derived; not live-ve |
| `rfdetr-small` | detect | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | Y | - | show_inference | Train lifecycle is capability-derived; not live-ve |
| `libreyolo-rtdetr-r50` | detect | libreyolo | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `libreyolo-yolov9-s` | detect | libreyolo | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `libreyolo-yolox-s` | detect | libreyolo | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-alexnet` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-convnext-tiny` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-densenet121` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-efficientnet-b0` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-mobilenet-v2` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-mobilenet-v3-large` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnet101` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnet152` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnet18` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnet34` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnet50` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-resnext50-32x4d` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `torchvision-wide-resnet50-2` | classify | torchvision-classify | `TRAIN_READY_LIVE` | Y | Y | show_train |  |
| `oneformer-convnext-large` | segment | oneformer | `WEIGHTS_MISSING` | - | - | hide | Weights not released or unverifiable. |

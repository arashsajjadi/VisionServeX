# v3.18 Discovered Models — 151 total (v3.19.0)

Programmatically discovered from `list_models()` + the registry + the license
policy. Every row is a pure projection of `model_capabilities(model_id)`.

## Readiness state distribution

| readiness_state | count |
|---|---:|
| `INFERENCE_READY_LIVE` | 37 |
| `CATALOG_ONLY_ENGINE_NOT_WIRED` | 27 |
| `TRAIN_READY_LIVE` | 24 |
| `SEGMENTATION_READY_LIVE` | 21 |
| `CUSTOM_LOADER_REQUIRED` | 15 |
| `EMBEDDING_READY_LIVE` | 10 |
| `OPEN_VOCAB_READY_LIVE` | 10 |
| `DEPENDENCY_MISSING` | 3 |
| `PARTIAL_IMPLEMENTATION_BLOCKED` | 2 |
| `WEIGHTS_MISSING` | 1 |
| `GATED_TOKEN_REQUIRED` | 1 |

## Full inventory

| Model | Family | Task | Engine | License | Commercial | Gated | readiness_state | Visible |
|---|---|---|---|---|:-:|:-:|---|:-:|
| `convnextv2-base` | convnextv2 | classify | convnextv2 | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `convnextv2-large` | convnextv2 | classify | convnextv2 | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `convnextv2-tiny` | convnextv2 | classify | convnextv2 | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `internimage-b` | internimage | classify | openmmlab | MIT | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `internimage-h` | internimage | classify | openmmlab | MIT | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `internimage-l` | internimage | classify | openmmlab | MIT | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `internimage-s` | internimage | classify | openmmlab | MIT | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `internimage-t` | internimage | classify | openmmlab | MIT | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `maxvit-tiny-tf-224` | maxvit | classify | maxvit | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `mock-classify` | mock | classify | mock | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `swinv2-base` | swinv2 | classify | swinv2 | MIT | — | — | `INFERENCE_READY_LIVE` | yes |
| `swinv2-large` | swinv2 | classify | swinv2 | MIT | — | — | `INFERENCE_READY_LIVE` | yes |
| `swinv2-small` | swinv2 | classify | swinv2 | MIT | — | — | `INFERENCE_READY_LIVE` | yes |
| `swinv2-tiny` | swinv2 | classify | swinv2 | MIT | — | — | `INFERENCE_READY_LIVE` | yes |
| `torchvision-alexnet` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-convnext-tiny` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-densenet121` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-efficientnet-b0` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-mobilenet-v2` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-mobilenet-v3-large` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnet101` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnet152` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnet18` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnet34` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnet50` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-resnext50-32x4d` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `torchvision-wide-resnet50-2` | torchvision-classify | classify | torchvision_classify | BSD-3-Clause | yes | — | `TRAIN_READY_LIVE` | yes |
| `deim-m` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deim-s` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-atto` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-femto` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-l` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-m` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-n` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-pico` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-s` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `deimv2-x` | deim | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `dfine-l` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-l-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-l-o365-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-m` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-m-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-m-o365-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-n` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-n-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-s` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-s-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-s-o365-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-x` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-x-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `dfine-x-o365-coco` | dfine | detect | dfine | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-dfine-l` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-dfine-m` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-dfine-n` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-dfine-s` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-dfine-x` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-rtdetr-r101` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-rtdetr-r50` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `TRAIN_READY_LIVE` | yes |
| `libreyolo-yolov9-c` | libreyolo | detect | libreyolo | MIT | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-yolov9-m` | libreyolo | detect | libreyolo | MIT | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-yolov9-s` | libreyolo | detect | libreyolo | MIT | yes | — | `TRAIN_READY_LIVE` | yes |
| `libreyolo-yolox-l` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-yolox-m` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `libreyolo-yolox-s` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `TRAIN_READY_LIVE` | yes |
| `libreyolo-yolox-x` | libreyolo | detect | libreyolo | Apache-2.0 | yes | — | `INFERENCE_READY_LIVE` | yes |
| `mock-detect` | mock | detect | mock | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `rfdetr-base` | rfdetr | detect | rfdetr | Apache-2.0 | — | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-large` | rfdetr | detect | rfdetr | Apache-2.0 | — | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-medium` | rfdetr | detect | rfdetr | Apache-2.0 | — | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-nano` | rfdetr | detect | rfdetr | Apache-2.0 | — | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-small` | rfdetr | detect | rfdetr | Apache-2.0 | — | — | `TRAIN_READY_LIVE` | yes |
| `rtdetrv4-l` | rtdetrv4 | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `rtdetrv4-m` | rtdetrv4 | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `rtdetrv4-s` | rtdetrv4 | detect | _stub | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtdetrv4-x` | rtdetrv4 | detect | _stub | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `clip-vit-base-patch32` | clip | embed | clip | MIT | yes | — | `EMBEDDING_READY_LIVE` | yes |
| `clip-vit-large-patch14` | clip | embed | clip | MIT | — | — | `EMBEDDING_READY_LIVE` | yes |
| `dinov2-base` | dinov2 | embed | dinov2 | Apache-2.0 | yes | — | `EMBEDDING_READY_LIVE` | yes |
| `dinov2-giant` | dinov2 | embed | dinov2 | Apache-2.0 | yes | — | `EMBEDDING_READY_LIVE` | yes |
| `dinov2-large` | dinov2 | embed | dinov2 | Apache-2.0 | yes | — | `EMBEDDING_READY_LIVE` | yes |
| `dinov2-small` | dinov2 | embed | dinov2 | Apache-2.0 | yes | — | `EMBEDDING_READY_LIVE` | yes |
| `siglip-base-patch16-224` | siglip | embed | siglip | Apache-2.0 | — | — | `EMBEDDING_READY_LIVE` | yes |
| `siglip2-base-patch16-224` | siglip2 | embed | siglip2 | Apache-2.0 | — | — | `EMBEDDING_READY_LIVE` | yes |
| `siglip2-large-patch16-256` | siglip2 | embed | siglip2 | Apache-2.0 | — | — | `EMBEDDING_READY_LIVE` | yes |
| `siglip2-so400m-patch14-384` | siglip2 | embed | siglip2 | Apache-2.0 | — | — | `EMBEDDING_READY_LIVE` | yes |
| `efficientsam` | efficientsam | foundation_segment | efficientsam | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `hq-sam` | hq-sam | foundation_segment | hq-sam | Apache-2.0 weights / HQSeg-44K dataset partly NC | — | — | `SEGMENTATION_READY_LIVE` | — |
| `medsam` | sam | foundation_segment | sam_hf | Apache-2.0 weights / medical dataset provenance | — | — | `SEGMENTATION_READY_LIVE` | — |
| `mobilesam` | mobilesam | foundation_segment | mobilesam | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `mock-foundation-segment` | mock | foundation_segment | mock | Apache-2.0 | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam-vit-base` | sam | foundation_segment | sam_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam-vit-huge` | sam | foundation_segment | sam_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam-vit-large` | sam | foundation_segment | sam_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2-hiera-base-plus` | sam2 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2-hiera-large` | sam2 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2-hiera-small` | sam2 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2-hiera-tiny` | sam2 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2.1-hiera-base-plus` | sam2.1 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2.1-hiera-large` | sam2.1 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2.1-hiera-small` | sam2.1 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam2.1-hiera-tiny` | sam2.1 | foundation_segment | sam2_hf | Apache-2.0 | yes | — | `SEGMENTATION_READY_LIVE` | yes |
| `sam3-base` | sam3 | foundation_segment | huggingface | SAM License (Meta custom, gated) | — | yes | `GATED_TOKEN_REQUIRED` | — |
| `grounded-sam` | grounded-sam | grounded_segment | grounded_sam | Apache-2.0 | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `grounded-sam2` | grounded-sam | grounded_segment | grounded_sam2 | Apache-2.0 | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `mock-grounded-segment` | mock | grounded_segment | mock | Apache-2.0 | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `seem-davit-d3` | seem | grounded_segment | huggingface | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `seem-focal-t` | seem | grounded_segment | huggingface | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `mock-obb` | mock | obb | mock | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `rtmdet-r-l` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r-m` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r-s` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r-t` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r2-l` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r2-m` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmdet-r2-s` | rtmdet | obb | openmmlab_sidecar | Apache-2.0 | — | — | `PARTIAL_IMPLEMENTATION_BLOCKED` | — |
| `rtmdet-r2-t` | rtmdet | obb | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `grounding-dino-1.5` | grounding-dino | open_vocab_detect | grounding_dino | not released | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `grounding-dino-1.6` | grounding-dino | open_vocab_detect | grounding_dino | Custom | — | yes | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `grounding-dino-original-swin-b` | grounding-dino | open_vocab_detect | grounding_dino | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `grounding-dino-original-swin-t` | grounding-dino | open_vocab_detect | grounding_dino | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `grounding-dino-swin-b` | grounding-dino | open_vocab_detect | grounding_dino | Apache-2.0 | yes | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `grounding-dino-swin-t` | grounding-dino | open_vocab_detect | grounding_dino | Apache-2.0 | yes | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `grounding-dino-tiny` | grounding-dino | open_vocab_detect | grounding_dino | Apache-2.0 | yes | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `mock-open-vocab` | mock | open_vocab_detect | mock | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `owlv2-base-patch16` | owlv2 | open_vocab_detect | owlv2 | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `owlv2-large-patch14` | owlv2 | open_vocab_detect | owlv2 | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `owlvit-base-patch32` | owlvit | open_vocab_detect | owlvit | Apache-2.0 | yes | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `owlvit-large-patch14` | owlvit | open_vocab_detect | owlvit | Apache-2.0 | — | — | `OPEN_VOCAB_READY_LIVE` | yes |
| `mock-pose` | mock | pose | mock | Apache-2.0 | — | — | `INFERENCE_READY_LIVE` | yes |
| `rtmpose-l` | rtmpose | pose | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmpose-l-384x288` | rtmpose | pose | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmpose-m` | rtmpose | pose | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmpose-m-384x288` | rtmpose | pose | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rtmpose-s` | rtmpose | pose | openmmlab_sidecar | Apache-2.0 | — | — | `PARTIAL_IMPLEMENTATION_BLOCKED` | — |
| `rtmpose-t` | rtmpose | pose | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `co-dino-inst-vit-l-coco` | co-dino | segment | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `co-dino-inst-vit-l-lvis` | co-dino | segment | openmmlab | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `maskdino-r50-coco` | maskdino | segment | openmmlab | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `maskdino-r50-panoptic` | maskdino | segment | openmmlab | Apache-2.0 | — | — | `CUSTOM_LOADER_REQUIRED` | — |
| `mock-segment` | mock | segment | mock | Apache-2.0 | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `oneformer-convnext-large` | oneformer | segment | oneformer | MIT weights / training-data review | — | — | `WEIGHTS_MISSING` | — |
| `oneformer-dinat-large` | oneformer | segment | oneformer | MIT weights / training-data review | — | — | `DEPENDENCY_MISSING` | — |
| `oneformer-swin-large` | oneformer | segment | oneformer | MIT | — | — | `SEGMENTATION_READY_LIVE` | yes |
| `rfdetr-seg-2xlarge` | rfdetr | segment | rfdetr | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rfdetr-seg-large` | rfdetr | segment | rfdetr | Apache-2.0 | yes | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `rfdetr-seg-medium` | rfdetr | segment | rfdetr | Apache-2.0 | yes | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-seg-nano` | rfdetr | segment | rfdetr | Apache-2.0 | yes | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-seg-small` | rfdetr | segment | rfdetr | Apache-2.0 | yes | — | `TRAIN_READY_LIVE` | yes |
| `rfdetr-seg-xlarge` | rfdetr | segment | rfdetr | Apache-2.0 | — | — | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — |
| `florence-2-base` | florence-2 | vlm | florence2 | MIT | yes | — | `DEPENDENCY_MISSING` | — |
| `florence-2-large` | florence-2 | vlm | florence2 | MIT | yes | — | `DEPENDENCY_MISSING` | — |

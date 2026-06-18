# v3.20 Train / Fine-tune Matrix

Embedding head-finetune live this sprint; full-train families referenced from committed v3.18/v3.19 matrices; foundation segmenters honestly inference-only.
**fine_tune_ready_live (embedding head-probe): 34**

| Model | Task | Method | Train status | Fine-tune status | Live | Reload | Export |
|---|---|---|---|---|:-:|:-:|:-:|
| `torchvision-alexnet` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-convnext-tiny` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-densenet121` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-efficientnet-b0` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-mobilenet-v2` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-mobilenet-v3-large` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnet101` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnet152` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnet18` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnet34` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnet50` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-resnext50-32x4d` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `torchvision-wide-resnet50-2` | classify | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `libreyolo-rtdetr-r50` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `libreyolo-yolov9-s` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `libreyolo-yolox-s` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-base` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-large` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-medium` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-nano` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-small` | detect | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `clip-vit-base-patch32` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `clip-vit-large-patch14` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `dinov2-base` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `dinov2-giant` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `dinov2-large` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `dinov2-small` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `siglip-base-patch16-224` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `siglip2-base-patch16-224` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `siglip2-large-patch16-256` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `siglip2-so400m-patch14-384` | embed | head_train | NOT_TRAINABLE_BY_DESIGN | FINE_TUNE_READY_LIVE | yes | yes | — |
| `efficientsam` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `hq-sam` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `medsam` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `mobilesam` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam-vit-base` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam-vit-huge` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam-vit-large` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2-hiera-base-plus` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2-hiera-large` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2-hiera-small` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2-hiera-tiny` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2.1-hiera-base-plus` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2.1-hiera-large` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2.1-hiera-small` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `sam2.1-hiera-tiny` | foundation_segment | not_supported | NOT_TRAINABLE_BY_DESIGN | FOUNDATION_SEGMENT_INFERENCE_ONLY_LIVE | yes | — | — |
| `rfdetr-seg-medium` | segment | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-seg-nano` | segment | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |
| `rfdetr-seg-small` | segment | train | TRAIN_READY_LIVE | FINE_TUNE_READY_LIVE | yes | yes | yes |

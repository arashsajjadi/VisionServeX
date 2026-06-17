# VisionServeX Training Matrix (v3.18)

Live train lifecycle: **train → checkpoint → reload → predict-after-reload →
schema → export**. A model is `TRAIN_READY_LIVE` only when every stage passed
this sprint; otherwise it is honestly `TRAIN_READY_DERIVED` (e.g. RF-DETR,
whose native COCO trainer is too heavy for a CPU smoke and is not faked).

| Model | Family | readiness_state | Train | Ckpt | Reload | Predict | Export | live_train |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `libreyolo-rtdetr-r50` | libreyolo | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `libreyolo-yolov9-s` | libreyolo | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `libreyolo-yolox-s` | libreyolo | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `rfdetr-base` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-large` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-medium` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-nano` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-seg-2xlarge` | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — | — | — | — | — | — |
| `rfdetr-seg-large` | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — | — | — | — | — | — |
| `rfdetr-seg-medium` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-seg-nano` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-seg-small` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `rfdetr-seg-xlarge` | rfdetr | `CATALOG_ONLY_ENGINE_NOT_WIRED` | — | — | — | — | — | — |
| `rfdetr-small` | rfdetr | `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION` | — | — | — | — | — | — |
| `torchvision-alexnet` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-convnext-tiny` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-densenet121` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-efficientnet-b0` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-mobilenet-v2` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-mobilenet-v3-large` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnet101` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnet152` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnet18` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnet34` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnet50` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-resnext50-32x4d` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |
| `torchvision-wide-resnet50-2` | torchvision-classify | `TRAIN_READY_LIVE` | yes | yes | yes | yes | yes | yes |

**16 `TRAIN_READY_LIVE`** of 27 train-supported models. Full evidence: `docs/qa/v318_full_model_truth/live_train_lifecycle_matrix.json`.

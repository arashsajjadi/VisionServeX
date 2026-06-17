# v3.18 Live Train-Lifecycle Matrix

Full CPU smoke lifecycle for every train-ready candidate: train → checkpoint → reload → predict-after-reload → schema → export.
A model earns `TRAIN_READY_LIVE` only when every applicable stage passes. Native-trainer families (RF-DETR) are honestly recorded `DERIVED`, not faked.

**Totals:** 16 PASS · 0 FAIL · 8 SKIP_BLOCKED

| Model | Family | Train | Ckpt | Reload | Predict | Schema | Export | Final | Status |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| `libreyolo-rtdetr-r50` | libreyolo | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `libreyolo-yolov9-s` | libreyolo | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `libreyolo-yolox-s` | libreyolo | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `rfdetr-base` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-large` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-medium` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-nano` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-seg-medium` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-seg-nano` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-seg-small` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `rfdetr-small` | rfdetr | — | — | — | — | — | — | DERIVED | SKIP_BLOCKED |
| `torchvision-alexnet` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-convnext-tiny` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-densenet121` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-efficientnet-b0` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-mobilenet-v2` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-mobilenet-v3-large` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnet101` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnet152` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnet18` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnet34` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnet50` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-resnext50-32x4d` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |
| `torchvision-wide-resnet50-2` | torchvision-classify | yes | yes | yes | yes | yes | yes | TRAIN_READY_LIVE | PASS |

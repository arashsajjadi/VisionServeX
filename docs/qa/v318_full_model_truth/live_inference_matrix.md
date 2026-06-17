# v3.18 Live Inference Matrix

Real CPU smoke inference for every wired, legal, non-gated model. Each row
ran the model's own public API on a tiny fixture and validated the result
schema. `PASS` ⇒ the model is eligible for an `*_READY_LIVE` readiness state.

**Totals:** 101 PASS · 2 FAIL · 2 SKIP_BLOCKED · device=cpu

| Model | Task | Status | Method | Schema | Latency ms | Error / Blocker |
|---|---|---|---|---|---:|---|
| `convnextv2-base` | classify | PASS | classify | ok | 4066.3 |  |
| `convnextv2-large` | classify | PASS | classify | ok | 4802.7 |  |
| `convnextv2-tiny` | classify | PASS | classify | ok | 3316.4 |  |
| `mock-classify` | classify | PASS | classify | ok | 847.6 |  |
| `swinv2-base` | classify | PASS | classify | ok | 4449.8 |  |
| `swinv2-large` | classify | PASS | classify | ok | 5279.1 |  |
| `swinv2-small` | classify | PASS | classify | ok | 4184.7 |  |
| `swinv2-tiny` | classify | PASS | classify | ok | 3562.4 |  |
| `torchvision-alexnet` | classify | PASS | classify | ok | 4048.0 |  |
| `torchvision-convnext-tiny` | classify | PASS | classify | ok | 2772.4 |  |
| `torchvision-densenet121` | classify | PASS | classify | ok | 1931.6 |  |
| `torchvision-efficientnet-b0` | classify | PASS | classify | ok | 1800.3 |  |
| `torchvision-mobilenet-v2` | classify | PASS | classify | ok | 1698.8 |  |
| `torchvision-mobilenet-v3-large` | classify | PASS | classify | ok | 1790.3 |  |
| `torchvision-resnet101` | classify | PASS | classify | ok | 3420.3 |  |
| `torchvision-resnet152` | classify | PASS | classify | ok | 4072.0 |  |
| `torchvision-resnet18` | classify | PASS | classify | ok | 1403.0 |  |
| `torchvision-resnet34` | classify | PASS | classify | ok | 2500.1 |  |
| `torchvision-resnet50` | classify | PASS | classify | ok | 1571.6 |  |
| `torchvision-resnext50-32x4d` | classify | PASS | classify | ok | 2595.7 |  |
| `torchvision-wide-resnet50-2` | classify | PASS | classify | ok | 4410.9 |  |
| `dfine-l` | detect | PASS | detect | ok | 3841.8 |  |
| `dfine-l-coco` | detect | PASS | detect | ok | 4151.7 |  |
| `dfine-l-o365-coco` | detect | PASS | detect | ok | 3853.2 |  |
| `dfine-m` | detect | PASS | detect | ok | 3795.1 |  |
| `dfine-m-coco` | detect | PASS | detect | ok | 3911.7 |  |
| `dfine-m-o365-coco` | detect | PASS | detect | ok | 3570.9 |  |
| `dfine-n` | detect | PASS | detect | ok | 3473.2 |  |
| `dfine-n-coco` | detect | PASS | detect | ok | 3395.7 |  |
| `dfine-s` | detect | PASS | detect | ok | 3508.0 |  |
| `dfine-s-coco` | detect | PASS | detect | ok | 3741.5 |  |
| `dfine-s-o365-coco` | detect | PASS | detect | ok | 3394.4 |  |
| `dfine-x` | detect | PASS | detect | ok | 5047.1 |  |
| `dfine-x-coco` | detect | PASS | detect | ok | 4952.5 |  |
| `dfine-x-o365-coco` | detect | PASS | detect | ok | 4262.6 |  |
| `libreyolo-dfine-l` | detect | PASS | detect | ok | 7098.3 |  |
| `libreyolo-dfine-m` | detect | PASS | detect | ok | 5446.0 |  |
| `libreyolo-dfine-n` | detect | PASS | detect | ok | 1486.8 |  |
| `libreyolo-dfine-s` | detect | PASS | detect | ok | 5463.8 |  |
| `libreyolo-dfine-x` | detect | PASS | detect | ok | 9634.6 |  |
| `libreyolo-rtdetr-r101` | detect | PASS | detect | ok | 12321.5 |  |
| `libreyolo-rtdetr-r50` | detect | PASS | detect | ok | 2037.2 |  |
| `libreyolo-yolov9-c` | detect | PASS | detect | ok | 5365.8 |  |
| `libreyolo-yolov9-m` | detect | PASS | detect | ok | 6279.3 |  |
| `libreyolo-yolov9-s` | detect | PASS | detect | ok | 1556.6 |  |
| `libreyolo-yolox-l` | detect | PASS | detect | ok | 16119.5 |  |
| `libreyolo-yolox-m` | detect | PASS | detect | ok | 9514.4 |  |
| `libreyolo-yolox-s` | detect | PASS | detect | ok | 1490.6 |  |
| `libreyolo-yolox-x` | detect | PASS | detect | ok | 22264.9 |  |
| `mock-detect` | detect | PASS | detect | ok | 843.4 |  |
| `rfdetr-base` | detect | PASS | detect | ok | 4909.1 |  |
| `rfdetr-large` | detect | PASS | detect | ok | 3988.2 |  |
| `rfdetr-medium` | detect | PASS | detect | ok | 5033.1 |  |
| `rfdetr-nano` | detect | PASS | detect | ok | 4778.8 |  |
| `rfdetr-small` | detect | PASS | detect | ok | 4890.5 |  |
| `clip-vit-base-patch32` | embed | PASS | embed | ok | 3918.6 |  |
| `clip-vit-large-patch14` | embed | PASS | embed | ok | 5561.5 |  |
| `dinov2-base` | embed | PASS | embed | ok | 3333.9 |  |
| `dinov2-giant` | embed | PASS | embed | ok | 10545.3 |  |
| `dinov2-large` | embed | PASS | embed | ok | 3712.9 |  |
| `dinov2-small` | embed | PASS | embed | ok | 3347.9 |  |
| `siglip-base-patch16-224` | embed | PASS | embed | ok | 4324.6 |  |
| `siglip2-base-patch16-224` | embed | PASS | embed | ok | 3921.1 |  |
| `siglip2-large-patch16-256` | embed | PASS | embed | ok | 5539.5 |  |
| `siglip2-so400m-patch14-384` | embed | PASS | embed | ok | 7456.1 |  |
| `efficientsam` | foundation_segment | PASS | segment | ok | 1768.8 |  |
| `hq-sam` | foundation_segment | PASS | segment | ok | 4747.3 |  |
| `medsam` | foundation_segment | PASS | segment | ok | 6987.6 |  |
| `mobilesam` | foundation_segment | PASS | segment | ok | 1889.0 |  |
| `mock-foundation-segment` | foundation_segment | PASS | segment | ok | 843.0 |  |
| `sam-vit-base` | foundation_segment | PASS | segment | ok | 5682.7 |  |
| `sam-vit-huge` | foundation_segment | PASS | segment | ok | 15362.7 |  |
| `sam-vit-large` | foundation_segment | PASS | segment | ok | 10157.1 |  |
| `sam2-hiera-base-plus` | foundation_segment | PASS | segment | ok | 5909.3 |  |
| `sam2-hiera-large` | foundation_segment | PASS | segment | ok | 8637.8 |  |
| `sam2-hiera-small` | foundation_segment | PASS | segment | ok | 5228.0 |  |
| `sam2-hiera-tiny` | foundation_segment | PASS | segment | ok | 4466.5 |  |
| `sam2.1-hiera-base-plus` | foundation_segment | PASS | segment | ok | 5898.1 |  |
| `sam2.1-hiera-large` | foundation_segment | PASS | segment | ok | 8476.3 |  |
| `sam2.1-hiera-small` | foundation_segment | PASS | segment | ok | 4785.9 |  |
| `sam2.1-hiera-tiny` | foundation_segment | PASS | segment | ok | 4442.0 |  |
| `grounded-sam` | grounded_segment | PASS | segment | ok | 13854.2 |  |
| `grounded-sam2` | grounded_segment | PASS | segment | ok | 9717.5 |  |
| `mock-grounded-segment` | grounded_segment | PASS | segment | ok | 852.4 |  |
| `mock-obb` | obb | PASS | detect | ok | 854.8 |  |
| `grounding-dino-original-swin-b` | open_vocab_detect | PASS | detect | ok | 8547.8 |  |
| `grounding-dino-original-swin-t` | open_vocab_detect | PASS | detect | ok | 6752.9 |  |
| `grounding-dino-swin-b` | open_vocab_detect | PASS | detect | ok | 7343.8 |  |
| `grounding-dino-swin-t` | open_vocab_detect | PASS | detect | ok | 6757.5 |  |
| `grounding-dino-tiny` | open_vocab_detect | PASS | detect | ok | 6492.5 |  |
| `mock-open-vocab` | open_vocab_detect | PASS | detect | ok | 852.8 |  |
| `owlv2-base-patch16` | open_vocab_detect | PASS | detect | ok | 6689.3 |  |
| `owlv2-large-patch14` | open_vocab_detect | PASS | detect | ok | 13643.4 |  |
| `owlvit-base-patch32` | open_vocab_detect | PASS | detect | ok | 5787.5 |  |
| `owlvit-large-patch14` | open_vocab_detect | PASS | detect | ok | 11908.0 |  |
| `mock-pose` | pose | PASS | predict | ok | 853.3 |  |
| `mock-segment` | segment | PASS | segment | ok | 838.7 |  |
| `oneformer-convnext-large` | segment | SKIP_BLOCKED | segment | — | 1474.8 | WEIGHTS_DOWNLOAD_UNAVAILABLE: Hugging Face download failed for 'oneformer-convnext-large': 404 Client Error. ( |
| `oneformer-dinat-large` | segment | SKIP_BLOCKED | segment | — | 5236.7 | DEPENDENCY_MISSING: cannot import name 'natten2dav' from 'natten.functional' (/home/arash/miniconda3 |
| `oneformer-swin-large` | segment | PASS | segment | ok | 9468.6 |  |
| `rfdetr-seg-medium` | segment | PASS | segment | ok | 4058.7 |  |
| `rfdetr-seg-nano` | segment | PASS | segment | ok | 4010.0 |  |
| `rfdetr-seg-small` | segment | PASS | segment | ok | 3892.0 |  |
| `florence-2-base` | vlm | FAIL | predict | — | 1307.7 | MissingDependencyError: Florence-2 is incompatible with transformers 5.10.2. The model's custom code use |
| `florence-2-large` | vlm | FAIL | predict | — | 1300.1 | MissingDependencyError: Florence-2 is incompatible with transformers 5.10.2. The model's custom code use |

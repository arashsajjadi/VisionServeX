# VisionServeX Legal Model Audit (v3.18)

Every model's license is gated before any runtime/training is enabled. This
document is generated from `model_capabilities()` — see
`docs/qa/v318_full_model_truth/legal_matrix.json` for the machine-readable form.

## Hard rules enforced

- **AGPL / GPL / SSPL** on a runtime/training path → `LICENSE_BLOCKED` (never default-safe).
- **Non-commercial / research-only** → `NON_COMMERCIAL_BLOCKED` (never commercial-safe default).
- **Gated** models → BYOT only: the user supplies their own token and accepts the
  upstream license; VisionServeX never ships weights or tokens.
- **Unknown / custom** license with no curated policy row → hidden pending review.
- **No Ultralytics / AGPL import** on any runtime or training path (benchmark-only
  comparison code is optional and never imported by the package runtime).

## License class distribution

| class | count |
|---|---:|
| permissive | 148 |
| custom_unknown | 3 |

- **Copyleft (AGPL/GPL/SSPL) models:** 0 — none
- **Non-commercial models:** 0 — none
- **Gated (BYOT) models:** 2 — ['grounding-dino-1.6', 'sam3-base']
- **Unknown / custom license (hidden pending review):** 3 — ['grounding-dino-1.5', 'grounding-dino-1.6', 'sam3-base']

VisionServeX deliberately ships a **permissive-only** catalog: the registry
contains no AGPL/GPL/SSPL and no non-commercial weights (e.g. Ultralytics YOLO
and Deci YOLO-NAS are intentionally absent — LibreYOLO is the permissive
replacement). The copyleft/non-commercial gates therefore correctly bind on an
empty set today; the tests keep them binding forever.

## Per-model legal status

| Model | License | Class | Commercial-safe | Gated | Runtime allowed |
|---|---|---|:-:|:-:|:-:|
| `clip-vit-base-patch32` | MIT | permissive | yes | — | yes |
| `clip-vit-large-patch14` | MIT | permissive | — | — | yes |
| `co-dino-inst-vit-l-coco` | Apache-2.0 | permissive | — | — | yes |
| `co-dino-inst-vit-l-lvis` | Apache-2.0 | permissive | — | — | yes |
| `convnextv2-base` | Apache-2.0 | permissive | — | — | yes |
| `convnextv2-large` | Apache-2.0 | permissive | — | — | yes |
| `convnextv2-tiny` | Apache-2.0 | permissive | — | — | yes |
| `deim-m` | Apache-2.0 | permissive | — | — | yes |
| `deim-s` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-atto` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-femto` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-l` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-m` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-n` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-pico` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-s` | Apache-2.0 | permissive | — | — | yes |
| `deimv2-x` | Apache-2.0 | permissive | — | — | yes |
| `dfine-l` | Apache-2.0 | permissive | — | — | yes |
| `dfine-l-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-l-o365-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-m` | Apache-2.0 | permissive | — | — | yes |
| `dfine-m-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-m-o365-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-n` | Apache-2.0 | permissive | — | — | yes |
| `dfine-n-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-s` | Apache-2.0 | permissive | — | — | yes |
| `dfine-s-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-s-o365-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-x` | Apache-2.0 | permissive | — | — | yes |
| `dfine-x-coco` | Apache-2.0 | permissive | — | — | yes |
| `dfine-x-o365-coco` | Apache-2.0 | permissive | — | — | yes |
| `dinov2-base` | Apache-2.0 | permissive | yes | — | yes |
| `dinov2-giant` | Apache-2.0 | permissive | yes | — | yes |
| `dinov2-large` | Apache-2.0 | permissive | yes | — | yes |
| `dinov2-small` | Apache-2.0 | permissive | yes | — | yes |
| `efficientsam` | Apache-2.0 | permissive | yes | — | yes |
| `florence-2-base` | MIT | permissive | yes | — | yes |
| `florence-2-large` | MIT | permissive | yes | — | yes |
| `grounded-sam` | Apache-2.0 | permissive | — | — | yes |
| `grounded-sam2` | Apache-2.0 | permissive | — | — | yes |
| `grounding-dino-1.5` | not released | custom_unknown | — | — | yes |
| `grounding-dino-1.6` | Custom | custom_unknown | — | yes | yes |
| `grounding-dino-original-swin-b` | Apache-2.0 | permissive | — | — | yes |
| `grounding-dino-original-swin-t` | Apache-2.0 | permissive | — | — | yes |
| `grounding-dino-swin-b` | Apache-2.0 | permissive | yes | — | yes |
| `grounding-dino-swin-t` | Apache-2.0 | permissive | yes | — | yes |
| `grounding-dino-tiny` | Apache-2.0 | permissive | yes | — | yes |
| `hq-sam` | Apache-2.0 weights / HQSeg-44K dataset partly NC | permissive | — | — | yes |
| `internimage-b` | MIT | permissive | — | — | yes |
| `internimage-h` | MIT | permissive | — | — | yes |
| `internimage-l` | MIT | permissive | — | — | yes |
| `internimage-s` | MIT | permissive | — | — | yes |
| `internimage-t` | MIT | permissive | — | — | yes |
| `libreyolo-dfine-l` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-dfine-m` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-dfine-n` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-dfine-s` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-dfine-x` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-rtdetr-r101` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-rtdetr-r50` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-yolov9-c` | MIT | permissive | yes | — | yes |
| `libreyolo-yolov9-m` | MIT | permissive | yes | — | yes |
| `libreyolo-yolov9-s` | MIT | permissive | yes | — | yes |
| `libreyolo-yolox-l` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-yolox-m` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-yolox-s` | Apache-2.0 | permissive | yes | — | yes |
| `libreyolo-yolox-x` | Apache-2.0 | permissive | yes | — | yes |
| `maskdino-r50-coco` | Apache-2.0 | permissive | — | — | yes |
| `maskdino-r50-panoptic` | Apache-2.0 | permissive | — | — | yes |
| `maxvit-tiny-tf-224` | Apache-2.0 | permissive | — | — | yes |
| `medsam` | Apache-2.0 weights / medical dataset provenance | permissive | — | — | yes |
| `mobilesam` | Apache-2.0 | permissive | yes | — | yes |
| `mock-classify` | Apache-2.0 | permissive | — | — | yes |
| `mock-detect` | Apache-2.0 | permissive | — | — | yes |
| `mock-foundation-segment` | Apache-2.0 | permissive | — | — | yes |
| `mock-grounded-segment` | Apache-2.0 | permissive | — | — | yes |
| `mock-obb` | Apache-2.0 | permissive | — | — | yes |
| `mock-open-vocab` | Apache-2.0 | permissive | — | — | yes |
| `mock-pose` | Apache-2.0 | permissive | — | — | yes |
| `mock-segment` | Apache-2.0 | permissive | — | — | yes |
| `oneformer-convnext-large` | MIT weights / training-data review | permissive | — | — | yes |
| `oneformer-dinat-large` | MIT weights / training-data review | permissive | — | — | yes |
| `oneformer-swin-large` | MIT | permissive | — | — | yes |
| `owlv2-base-patch16` | Apache-2.0 | permissive | — | — | yes |
| `owlv2-large-patch14` | Apache-2.0 | permissive | — | — | yes |
| `owlvit-base-patch32` | Apache-2.0 | permissive | yes | — | yes |
| `owlvit-large-patch14` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-base` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-large` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-medium` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-nano` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-seg-2xlarge` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-seg-large` | Apache-2.0 | permissive | yes | — | yes |
| `rfdetr-seg-medium` | Apache-2.0 | permissive | yes | — | yes |
| `rfdetr-seg-nano` | Apache-2.0 | permissive | yes | — | yes |
| `rfdetr-seg-small` | Apache-2.0 | permissive | yes | — | yes |
| `rfdetr-seg-xlarge` | Apache-2.0 | permissive | — | — | yes |
| `rfdetr-small` | Apache-2.0 | permissive | — | — | yes |
| `rtdetrv4-l` | Apache-2.0 | permissive | — | — | yes |
| `rtdetrv4-m` | Apache-2.0 | permissive | — | — | yes |
| `rtdetrv4-s` | Apache-2.0 | permissive | — | — | yes |
| `rtdetrv4-x` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r-l` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r-m` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r-s` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r-t` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r2-l` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r2-m` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r2-s` | Apache-2.0 | permissive | — | — | yes |
| `rtmdet-r2-t` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-l` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-l-384x288` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-m` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-m-384x288` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-s` | Apache-2.0 | permissive | — | — | yes |
| `rtmpose-t` | Apache-2.0 | permissive | — | — | yes |
| `sam-vit-base` | Apache-2.0 | permissive | yes | — | yes |
| `sam-vit-huge` | Apache-2.0 | permissive | yes | — | yes |
| `sam-vit-large` | Apache-2.0 | permissive | yes | — | yes |
| `sam2-hiera-base-plus` | Apache-2.0 | permissive | yes | — | yes |
| `sam2-hiera-large` | Apache-2.0 | permissive | yes | — | yes |
| `sam2-hiera-small` | Apache-2.0 | permissive | yes | — | yes |
| `sam2-hiera-tiny` | Apache-2.0 | permissive | yes | — | yes |
| `sam2.1-hiera-base-plus` | Apache-2.0 | permissive | yes | — | yes |
| `sam2.1-hiera-large` | Apache-2.0 | permissive | yes | — | yes |
| `sam2.1-hiera-small` | Apache-2.0 | permissive | yes | — | yes |
| `sam2.1-hiera-tiny` | Apache-2.0 | permissive | yes | — | yes |
| `sam3-base` | SAM License (Meta custom, gated) | custom_unknown | — | yes | yes |
| `seem-davit-d3` | Apache-2.0 | permissive | — | — | yes |
| `seem-focal-t` | Apache-2.0 | permissive | — | — | yes |
| `siglip-base-patch16-224` | Apache-2.0 | permissive | — | — | yes |
| `siglip2-base-patch16-224` | Apache-2.0 | permissive | — | — | yes |
| `siglip2-large-patch16-256` | Apache-2.0 | permissive | — | — | yes |
| `siglip2-so400m-patch14-384` | Apache-2.0 | permissive | — | — | yes |
| `swinv2-base` | MIT | permissive | — | — | yes |
| `swinv2-large` | MIT | permissive | — | — | yes |
| `swinv2-small` | MIT | permissive | — | — | yes |
| `swinv2-tiny` | MIT | permissive | — | — | yes |
| `torchvision-alexnet` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-convnext-tiny` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-densenet121` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-efficientnet-b0` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-mobilenet-v2` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-mobilenet-v3-large` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnet101` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnet152` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnet18` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnet34` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnet50` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-resnext50-32x4d` | BSD-3-Clause | permissive | yes | — | yes |
| `torchvision-wide-resnet50-2` | BSD-3-Clause | permissive | yes | — | yes |

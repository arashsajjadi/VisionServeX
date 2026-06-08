# Commercial-safe core

The commercial-safe core is the set of models with permissive licenses
(Apache-2.0 / MIT) that are `default_safe`, `commercial_safe`, and
`production_allowed`. Weights are pulled from the official source on demand —
VisionServeX never bundles them into the wheel.

The authoritative, always-current list is generated into
`notebook/99_final_report/reports/v38_license_policy_matrix.csv`
(`final_policy = commercial_safe_core`). At v3.8 this is **39 models**:

| family | models |
|---|---|
| SAM v1 | `sam-vit-base`, `sam-vit-large`, `sam-vit-huge`, `mobilesam`, `efficientsam` |
| SAM 2 / 2.1 | `sam2-hiera-{tiny,small,base-plus,large}`, `sam2.1-hiera-{tiny,small,base-plus,large}` |
| DINOv2 | `dinov2-{small,base,large,giant}` |
| Open GroundingDINO | `grounding-dino-tiny`, `grounding-dino-base`, `grounding-dino-swin-{t,b}` |
| Florence-2 | `florence-2-base`, `florence-2-large` (MIT) |
| CLIP / OWL | `clip-vit-base-patch32` (MIT), `owlvit-base-patch32`, `owlv2-base-patch16-ensemble` |
| Depth | `depth-anything-small` (Apache; the larger V2 is non-commercial) |
| RF-DETR-Seg | `rfdetr-seg-{nano,small,medium,large}` (Apache; XL/2XL held for review) |
| EfficientViT-SAM | `efficientvit-sam-{l0,l1,l2}` |
| Interactive | `ritm` (MIT) |
| Detection sidecars* | `maskdino`, `co-dino`, `rt-detrv4`, `rtmdet` |

\* The detection sidecars are commercial-safe by license (Apache-2.0) and become
runnable when their isolated sidecar runtime (Detectron2 / mmdet) builds; they are
classified core because the licensing is clean.

## Why these and not others

- **SAM v1 / SAM 2 / EfficientViT-SAM / EfficientSAM** are Apache-2.0. They are
  trained/distilled on SA-1B (a research-only *dataset*), which Meta, MIT, and
  Huawei all treat as Apache-compatible for the released *weights* — the same
  basis on which Meta released SAM itself. `dataset_risk` is recorded per row.
- **DINOv2** is Apache-2.0 (distinct from the custom-licensed DINOv3 → BYOT).
- **Florence-2 / CLIP** are MIT; **OWL-ViT / OWLv2** are Apache-2.0.
- **RF-DETR-Seg nano…large** are Apache-2.0 with a DINOv2 backbone. The XL/2XL
  *segmentation* checkpoints were reported Apache by v3.7 research but are held in
  `legal_review` pending confirmation of current Roboflow terms.

## Usage

No token needed:

```bash
visionservex model pull sam-vit-base
visionservex sam sam2.1-hiera-small segment image.jpg --box 10,20,200,220
```

```python
from visionservex import VSX
VSX.sam("sam2.1-hiera-small").segment("image.jpg", box=[10, 20, 200, 220])
VSX.dino("dinov2-base").embed("image.jpg")
```

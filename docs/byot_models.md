# BYOT models (Bring Your Own Token)

These models are **gated** or carry a **custom upstream license**. VisionServeX
never redistributes their weights. You connect your own Hugging Face token, accept
the upstream license yourself, and the weights download into your own HF cache.

> This model is gated or uses a custom upstream license. You must use your own
> Hugging Face token and accept the upstream license yourself. VisionServeX does
> not redistribute the weights. Commercial use depends on the upstream license
> terms you accepted.

## Which models

`final_policy = byot_license_required` (see the live matrix at
`notebook/99_final_report/reports/v38_license_policy_matrix.csv`):

- **SAM 3** — `sam3-base`, `sam3-image`, `sam3-video`, `sam3-text-prompt`,
  `sam3-visual-prompt`, `sam3-exemplar-prompt`, `sam3-open-vocabulary`,
  `sam3-tracking` → Hub repo `facebook/sam3` (custom **SAM License**, gated).
- **SAM 3.1** — `sam3.1-base`, `sam3.1-image`, `sam3.1-video`,
  `sam3.1-open-vocabulary`, `sam3.1-text-prompt`, `sam3.1-visual-prompt`,
  `sam3.1-real-time-tracking` → Hub repo `facebook/sam3.1`.
- **DINOv3** — `dinov3-vits16/vitb16/vitl16/vit7b16` and
  `dinov3-convnext-tiny/small/base/large` → `facebook/dinov3-*-pretrain-lvd1689m`
  (custom **DINOv3 License**: commercial use permitted with "Built with DINOv3"
  attribution + acceptable-use + no-compete-training conditions).
- **INSID3** — `insid3-small`, `insid3-base`, `insid3-large` (alias: `insid3`)
  Training-free in-context segmentation (CVPR 2026 Oral, arXiv 2603.28480).
  INSID3 code is Apache-2.0 (visinf/INSID3); uses frozen DINOv3 backbone.
  No INSID3-specific weights shipped. Backbone: DINOv3 License (Meta custom),
  requires "Built with DINOv3" attribution. See [docs/insid3.md](insid3.md).

> **Note:** SAM 3 / SAM 3.1 use a *custom* SAM License, not Apache-2.0 like SAM 1/2.
> DINOv3 uses a *custom* DINOv3 License, not Apache-2.0 like DINOv2. Commercial use
> depends on the terms **you** accept. Provenance of the very newest SAM 3.x
> releases is flagged as unverified — review the actual license text yourself.

## Usage

```bash
# 1. Connect your token (one of):
huggingface-cli login
visionservex hf connect --token-env HF_TOKEN

# 2. Accept the upstream license on the model page (one click):
#    https://huggingface.co/facebook/sam3
#    https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m

# 3. Confirm access (no download), then pull:
visionservex hf check-model facebook/sam3
visionservex model pull sam3-base --accept-upstream-license --hf-token-env HF_TOKEN
visionservex model doctor sam3-base
```

```python
from visionservex import VSX

VSX.hf.status()
VSX.model("sam3-base").access()                       # 'access_granted' once accepted
VSX.model("sam3-base").pull(accept_upstream_license=True)
# Inference (transformers Sam3Model backend) once weights are cached:
VSX.sam("sam3-base").segment("image.jpg", text="person")
emb = VSX.dino("dinov3-vitb16").embed("image.jpg")    # BYOT DINOv3 embedding
```

## What VisionServeX does and does not do

- ✅ Detects your token, checks access (metadata only), downloads to *your* cache.
- ✅ Redacts the token in every log line, JSON payload, and error message.
- ❌ Never bundles weights into the wheel / repo / Docker image.
- ❌ Never accepts a license on your behalf.
- ❌ Never marks a BYOT model `default_safe` or auto-downloads it without
  `--accept-upstream-license`.

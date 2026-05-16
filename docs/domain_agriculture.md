# VisionServeX Agriculture Domain

Agriculture computer vision workflows using VisionServeX. All models listed are permissive-license unless noted.

---

## Overview

VisionServeX supports agriculture workflows through:
1. **General object detection** (RF-DETR, D-FINE) fine-tuned for crops/weeds
2. **Open-vocabulary detection** (Grounding DINO, OWLv2) for zero-shot plant detection
3. **Segmentation** (SAM v1/2/2.1) for precision leaf/plant area measurement
4. **VLM captioning** (Florence-2) for plant disease description
5. **Agriculture-specific models** (AgriCLIP — audit-only)

---

## Weed / Crop Detection

Use RF-DETR as the recommended backbone for fine-tuning on agricultural detection datasets.

### Quick recipe (using RF-DETR)

```bash
# Install
pip install 'visionservex[rfdetr]'

# Zero-shot open-vocab detection (no fine-tuning needed)
visionservex open-vocab owlv2-base-patch16 field_image.jpg \
  --prompt "weed,crop,bare soil,water stress"

# Or use Grounding DINO for more accurate zero-shot
visionservex open-vocab grounding-dino-swin-b field_image.jpg \
  --prompt "weed plant, healthy crop, diseased leaf"

# For production: train RF-DETR on your labeled dataset
# (Requires YOLO or COCO format annotations)
visionservex training capabilities --model rfdetr-large
```

### Agriculture CLI

```bash
# All agriculture commands:
visionservex agriculture --help

# Weed detection recipe:
visionservex agriculture detect field_image.jpg --type weed

# Crop health check:
visionservex agriculture detect field_image.jpg --type crop-health
```

---

## Plant Disease Detection

Use Florence-2 or Grounding DINO for zero-shot plant disease description.

### VLM approach (Florence-2)

```bash
pip install 'visionservex[hf]'

# Describe disease symptoms
visionservex florence2 predict florence-2-large diseased_leaf.jpg \
  --task '<MORE_DETAILED_CAPTION>'

# Detect lesion regions
visionservex florence2 predict florence-2-large diseased_leaf.jpg \
  --task '<DENSE_REGION_CAPTION>'
```

### Open-vocab detection approach

```bash
# Detect visible disease signs
visionservex open-vocab grounding-dino-swin-b leaf_image.jpg \
  --prompt "leaf spot, rust lesion, powdery mildew, blight, yellowing"
```

### Segmentation for measurement

```bash
# Segment diseased region (use SAM 2.1 for best quality)
visionservex sam-family smoke-test sam2.1-hiera-large diseased_leaf.jpg \
  --box 50,80,200,250

# Or MedSAM-style multi-box for multiple lesions
visionservex medical segment medsam diseased_leaf.jpg \
  --box 50,80,200,250 \
  --box 100,200,180,280 \
  --out /tmp/lesion_masks
```

---

## AgriCLIP

AgriCLIP is a CLIP fine-tuned on agricultural imagery.

### Status

`unavailable` — license and HF model availability not verified at time of audit.

**Blocker:** License terms unclear; HF model card not verified live.

**Next action:** Re-audit https://github.com/umair1221/AgriCLIP for license and model weights.

**Alternative now:** Use `siglip2-base-patch16-224` or `clip-vit-large-patch14` for embedding-based crop/disease classification.

---

## SCOLD (Plant Disease)

SCOLD and similar specialized agricultural models are under evaluation.

### Status

`unavailable` — not yet in SOURCE_MANIFEST; pending license and access audit.

**Interim:** Use Florence-2 VLM captions for qualitative disease description.

---

## Recommended Workflow

For production agriculture deployments:

1. **Zero-shot detection**: `grounding-dino-swin-b` or `owlv2-large-patch14` with crop/weed/disease text prompts
2. **Fine-tuning**: Label 100-500 images with Roboflow, export YOLO format, train `rfdetr-large`
3. **Segmentation**: Use `sam2.1-hiera-large` with box prompts from step 1 or 2 for precision area measurement
4. **Disease captioning**: `florence-2-large` for detailed region description
5. **Embedding search**: `dinov2-base` or `clip-vit-large-patch14` for crop variety similarity

---

## Privacy

All inference is local by default. No images are transmitted to external services unless you explicitly use the `grounding-dino-1.5/1.6` API-gated models.

---

## Related Commands

```bash
visionservex agriculture --help
visionservex model-zoo sources --domain agriculture
visionservex model-zoo gap-report --json | jq '.runnable'
```

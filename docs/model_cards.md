# Model Cards

VisionServeX provides structured model cards for major model families.

## Usage

```bash
# Show a model card
visionservex model-card show dfine-s-o365-coco
visionservex model-card show dfine-s-o365-coco --format markdown
visionservex model-card show dfine-s-o365-coco --json

# List all models with available cards
visionservex model-card list
visionservex model-card list --task detect

# Export all cards to markdown
visionservex model-card export --out docs/generated_model_cards.md
```

## Card structure

Each card contains:

| Field | Description |
|-------|-------------|
| `model_id` | Registry model ID |
| `strength_category` | demo_fast / production_recommended / accuracy_grade / … |
| `recommended_for` | Use cases where this model excels |
| `not_recommended_for` | Use cases to avoid |
| `replaces_or_competes_with` | Ultralytics/other model comparisons |
| `install_command` | Exact pip install command |
| `quick_command` | Example CLI command |
| `expected_hardware` | CPU/CUDA/Colab suitability, VRAM |
| `official_benchmark_note` | Upstream AP claims (not VisionServeX-verified) |
| `visionservex_benchmark_status` | What's been tested in this build |
| `known_limitations` | Honest caveats |

## Honesty policy

- `demo_fast` cards explicitly warn against using them for AP benchmarks.
- `SAM/SAM2` cards warn against mixing with detection mAP.
- `official_benchmark_note` states upstream paper claims, not VisionServeX results.
- `visionservex_benchmark_status` states what was actually tested.

## Models with full cards

Detection: dfine-n, dfine-s, dfine-s/m/l/x-o365-coco, rfdetr-nano/small/medium/large

Segmentation: rfdetr-seg-nano/small/medium, grounded-sam, grounded-sam2,
sam-vit-base, sam2-hiera-tiny, oneformer-swin-large

Classification: swinv2-tiny, swinv2-base, internimage-t

Open-vocabulary: grounding-dino-tiny, grounding-dino-swin-b

Expert: rtmpose-s

All other models fall back to registry-derived cards with limited supplementary data.

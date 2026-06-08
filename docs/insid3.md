# INSID3 — Training-Free In-Context Segmentation with DINOv3

## Overview

INSID3 is a training-free, one-shot instance segmentation algorithm introduced at CVPR 2026 Oral.
It uses a frozen DINOv3 backbone to transfer a segmentation mask from a reference image to a
query image — no fine-tuning or task-specific weights required.

- **Paper**: arXiv 2603.28480
- **Code**: https://github.com/visinf/INSID3 — **Apache-2.0**
- **Backbone**: DINOv3 (Meta custom license — see license section below)
- **VisionServeX family**: `insid3`
- **Task**: `in_context_segmentation`
- **Policy**: `byot_license_required`

INSID3 has no weights of its own — it uses a frozen DINOv3 ViT backbone directly.
No INSID3-specific model files are downloaded or stored. VisionServeX does not redistribute
DINOv3 backbone weights; they load from the user's own Hugging Face cache.

---

## Algorithm

1. **Feature extraction**: DINOv3 patch-level features from reference and query images.
2. **Positional debiasing**: SVD-based projection removes position-dependent components.
3. **Prototype construction**: Mean embedding of reference patches covered by the input mask.
4. **Agglomerative clustering**: Query patches are grouped (ward linkage, sklearn).
5. **Cluster matching**: The cluster whose centroid has highest cosine similarity to the prototype is selected.
6. **Upsampling**: The winning cluster binary mask is upsampled to original image resolution.

---

## License

| Component | License |
|-----------|---------|
| INSID3 code | Apache-2.0 (visinf/INSID3) |
| DINOv3 backbone weights | DINOv3 License (Meta custom) |

**DINOv3 License conditions** (abbreviated):
- Commercial use is permitted.
- Attribution required: **"Built with DINOv3"** must appear in any product or publication.
- Acceptable-use policy applies (no harmful use).
- No competing foundational model training on DINOv3 outputs.

VisionServeX marks all INSID3 variants as `byot_license_required` and `can_ship_weights=False`
because of the DINOv3 backbone license. The code license (Apache-2.0) does not change this.

---

## Model Variants

| model_id | Backbone | HF repo |
|----------|----------|---------|
| `insid3-small` | DINOv3 ViT-S/16 ~21M | facebook/dinov3-vits16-pretrain-lvd1689m |
| `insid3-base` | DINOv3 ViT-B/16 ~86M | facebook/dinov3-vitb16-pretrain-lvd1689m |
| `insid3-large` (default) | DINOv3 ViT-L/16 ~300M | facebook/dinov3-vitl16-pretrain-lvd1689m |

Aliases: `insid3` and `insid3-default` both resolve to `insid3-large`.

---

## Setup

```bash
# 1. Accept DINOv3 upstream license at:
#    https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
#    (publicly accessible — accept the terms to confirm)

# 2. Authenticate
huggingface-cli login

# 3. Install dependencies
pip install 'visionservex[hf]' scikit-learn Pillow

# 4. Verify access
visionservex insid3 doctor --model-id insid3-large
```

---

## CLI Usage

```bash
# Status (policy rows)
visionservex insid3 status

# Doctor (HF token + backbone access)
visionservex insid3 doctor --model-id insid3-large

# Run one-shot segmentation
visionservex insid3 run query.jpg reference.jpg reference_mask.png \
    --model-id insid3-large \
    --device cpu \
    --n-clusters 6 \
    --out-dir results/

# Feature correspondence between two images
visionservex insid3 correspond image_a.jpg image_b.jpg \
    --model-id insid3-large \
    --out-dir results/
```

Outputs saved to `--out-dir`:
- `pred_mask.png` — binary segmentation mask (uint8, 0/255)
- `overlay.png` — query image with semi-transparent red overlay on predicted region
- `metadata.json` — timings, mask area, policy, attribution notice

---

## Python API

```python
from visionservex.vsx import VSX

# Check policy
handle = VSX.insid3('insid3-large')
print(handle.explain())

# Run segmentation
result = handle.segment(
    query_image='query.jpg',
    reference_image='reference.jpg',
    reference_mask='reference_mask.png',
    device='cpu',
    n_clusters=6,
    out_dir='results/',
)

if result['status'] == 'ok':
    print('Mask area:', result['mask_area_px'], 'px')
    print('Attribution:', result['attribution_required'])  # "Built with DINOv3"
```

Or directly:

```python
from visionservex.insid3_runtime import insid3_segment

result = insid3_segment(
    'query.jpg', 'reference.jpg', 'reference_mask.png',
    model_id='insid3-large',
    out_dir='results/',
)
```

---

## Output Schema

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | `"ok"` or `"blocked"` |
| `state` | str | `"benchmark_passed_byot_mask"` when successful |
| `mask_area_px` | int | Number of foreground pixels in predicted mask |
| `best_cluster_sim` | float | Cosine similarity of winning cluster to prototype |
| `load_ms` | float | DINOv3 model load time (milliseconds) |
| `feat_ms` | float | Feature extraction time (milliseconds) |
| `seg_ms` | float | Clustering + matching time (milliseconds) |
| `attribution_required` | str | `"Built with DINOv3"` |
| `saved_paths` | dict | Paths to `pred_mask`, `overlay`, `metadata` if `out_dir` set |

---

## Limitations

- **Zero-shot accuracy**: INSID3 is training-free; accuracy depends on DINOv3 feature quality.
  Complex scenes with many similar objects may produce imprecise masks.
- **Resolution**: DINOv3 processes images at a fixed patch grid. Very small objects (< 2 patches)
  may not be captured accurately.
- **Clustering dependency**: `scikit-learn` is required for agglomerative clustering.
  If not installed, falls back to cosine-similarity thresholding (less accurate).
- **Commercial use**: Requires "Built with DINOv3" attribution. BYOT models are not
  automatically commercial-safe beyond the attribution requirement.

---

## Tutorial Notebooks

See `notebook/tutorials/v311_insid3_in_context_segmentation/`:
- `01_insid3_natural_one_shot.ipynb` — basic usage with synthetic demo images

---

## Added in

VisionServeX **v3.11.0** — INSID3 In-Context Segmentation sprint.

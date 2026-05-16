# VisionServeX Industrial Anomaly Detection Domain

Industrial anomaly detection using the Anomalib integration. Detect manufacturing defects, surface anomalies, and quality control issues without labeled defect data (unsupervised learning).

---

## Architecture

VisionServeX wraps [Anomalib](https://github.com/open-edge-platform/anomalib) for industrial anomaly detection. Anomalib is kept outside the core package to avoid dependency conflicts. Use the `create-env` command for a clean setup.

**Required extra:** `pip install 'visionservex[anomaly]'` or `pip install anomalib`

**Validated versions:** `anomalib>=1.0,<3.0`

---

## Quick Start

```bash
# Option 1: Install in current env
pip install 'visionservex[anomaly]'

# Option 2: Isolated conda env (recommended)
visionservex anomaly create-env --name vsx-anomaly --python 3.11
# Run each command from the output

# Check doctor
visionservex anomaly doctor
```

---

## PatchCore (Recommended)

PatchCore is the most reliable unsupervised anomaly detector for industrial inspection. It builds a memory bank of normal image patches and compares test images against it.

**Paper:** [Towards Total Recall in Industrial Anomaly Detection (CVPR 2022)](https://arxiv.org/abs/2106.08265)

### Training

```bash
# Train on a folder of NORMAL (good) images
visionservex anomaly train patchcore \
  --data /path/to/normal_images \
  --out /tmp/vsx_patchcore

# Dry-run to see the plan:
visionservex anomaly train patchcore \
  --data /path/to/normal_images \
  --out /tmp/vsx_patchcore \
  --dry-run

# With custom image size:
visionservex anomaly train patchcore \
  --data /path/to/normal_images \
  --out /tmp/vsx_patchcore \
  --image-size 320
```

### Prediction

```bash
# Score a test image:
visionservex anomaly predict /tmp/vsx_patchcore test_image.jpg

# Save heatmap:
visionservex anomaly predict /tmp/vsx_patchcore test_image.jpg \
  --save-heatmap /tmp/anomaly_heatmap.png
```

### Data format

```
/path/to/normal_images/
  good_image_001.png
  good_image_002.png
  good_image_003.jpg
  ...
```

Recommended minimum: 50+ normal images. More is better (PatchCore scales well).

---

## Other Anomalib Algorithms

All algorithms are trained on normal images only (no defect labels needed).

### PaDiM (Patch Distribution Modeling)

**Paper:** [PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization](https://arxiv.org/abs/2011.08785)

```bash
visionservex anomaly train padim \
  --data /path/to/normal_images \
  --out /tmp/vsx_padim
```

### FastFlow (Normalizing Flows)

**Paper:** [FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows](https://arxiv.org/abs/2111.07677)

```bash
visionservex anomaly train fastflow \
  --data /path/to/normal_images \
  --out /tmp/vsx_fastflow
```

### EfficientAD (Millisecond latency)

**Paper:** [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535)

```bash
visionservex anomaly train efficientad \
  --data /path/to/normal_images \
  --out /tmp/vsx_efficientad
```

EfficientAD is the recommended choice for latency-critical applications.

### WinCLIP (Zero/few-shot)

**Paper:** [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814)

WinCLIP does not require training on normal images — it uses CLIP to detect anomalies from text prompts.

```bash
visionservex anomaly train winclip \
  --data /path/to/normal_images \
  --out /tmp/vsx_winclip
```

### DRAEM (Reconstruction-based)

**Paper:** [DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection](https://arxiv.org/abs/2108.07610)

```bash
visionservex anomaly train draem \
  --data /path/to/normal_images \
  --out /tmp/vsx_draem
```

### Reverse Distillation

**Paper:** [Anomaly Detection via Reverse Distillation from One-Class Embedding (CVPR 2022)](https://arxiv.org/abs/2201.10703)

```bash
visionservex anomaly train reverse_distillation \
  --data /path/to/normal_images \
  --out /tmp/vsx_reverse_distillation
```

---

## MVTec AD Support

MVTec AD is the standard industrial anomaly detection benchmark with 15 object categories.

Expected dataset layout for benchmark:
```
/path/to/mvtec/
  bottle/
    train/
      good/
        image_001.png
        ...
    test/
      good/
        ...
      broken_large/
        ...
    ground_truth/
      broken_large/
        image_001_mask.png
        ...
```

Benchmark command (planned for future release):
```bash
visionservex anomaly benchmark \
  --dataset mvtec:/path/to/mvtec \
  --model patchcore
```

---

## Structured Errors

If anomalib is not installed:
```json
{
  "code": "ANOMALIB_REQUIRED",
  "message": "anomalib is not installed",
  "fix": "pip install 'visionservex[anomaly]'   (or: pip install anomalib)"
}
```

If dataset is empty:
```json
{
  "code": "DATASET_REQUIRED",
  "message": "Dataset directory /path has no images",
  "fix": "Add training images: cp *.png /path/  (recommended: at least 50 normal images)"
}
```

---

## Install Help

```bash
# Show all install options (native pip, conda, docker):
visionservex anomaly install-help
visionservex anomaly install-help --json
```

---

## Related Commands

```bash
visionservex anomaly --help
visionservex anomaly list
visionservex anomaly doctor
visionservex anomaly create-env --help
visionservex model-zoo sources --domain industrial
```

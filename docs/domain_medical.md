# VisionServeX Medical Imaging Domain

> **RESEARCH AND EDUCATION ONLY.** VisionServeX medical commands do not provide medical diagnosis, treatment recommendation, or clinical guidance. Do not use these outputs for patient care.

---

## Supported Models

| Model | Status | Install | Notes |
|-------|--------|---------|-------|
| MedSAM | runnable | `pip install 'visionservex[hf]'` | Real mask output, multi-box support |
| TotalSegmentator | expert_sidecar | `pip install TotalSegmentator` | CT/MR multi-organ segmentation |
| nnU-Net v2 | expert_sidecar | `pip install nnunetv2` | Self-configuring framework |
| MedSAM2 | expert_sidecar | Clone repo | Volumetric/video prompted segmentation |
| SAM-Med2D | expert_sidecar | Clone repo | 2D medical SAM adapter |
| MONAI Model Zoo | expert_sidecar | `pip install monai` | Curated bundle library |

---

## MedSAM

MedSAM is a promptable medical image segmentation model based on SAM, fine-tuned on diverse medical imaging data. It is wired in VisionServeX via the `sam_hf` engine using the `wanglab/medsam-vit-base` HF checkpoint.

### Install

```bash
pip install 'visionservex[hf]'
# Weights are downloaded automatically from HuggingFace on first use
```

### Single-box segmentation

```bash
visionservex medical segment medsam ct_slice.png \
  --box 50,100,200,300 \
  --out /tmp/medsam_out
```

### Multi-box segmentation (v2.5.0+)

Pass `--box` multiple times to segment multiple regions in one call:

```bash
visionservex medical segment medsam ct_slice.png \
  --box 50,100,200,300 \
  --box 250,150,400,350 \
  --box 10,400,80,480 \
  --out /tmp/medsam_out
```

JSON output:

```bash
visionservex medical segment medsam ct_slice.png \
  --box 50,100,200,300 \
  --out /tmp/medsam_out \
  --json
```

Example output:
```json
{
  "model_id": "medsam",
  "input": "ct_slice.png",
  "boxes": ["50,100,200,300"],
  "out": "/tmp/medsam_out",
  "masks_saved": [
    {
      "mask_path": "/tmp/medsam_out/mask_000.png",
      "iou_score": 0.93,
      "box": [50, 100, 200, 300]
    }
  ],
  "n_masks": 1,
  "status": "ok",
  "disclaimer": "RESEARCH AND EDUCATION ONLY..."
}
```

### Structured errors

If `transformers` is not installed:
```json
{
  "code": "MEDICAL_EXTRA_REQUIRED",
  "message": "Missing modules: ['transformers']",
  "fix": "pip install 'visionservex[hf]'"
}
```

If checkpoint is not cached:
```json
{
  "code": "CHECKPOINT_REQUIRED",
  "message": "MedSAM checkpoint not cached",
  "fix": "visionservex model pull medsam"
}
```

### License

Apache-2.0. Research-grade. Not validated for clinical decisions.

---

## TotalSegmentator

TotalSegmentator segments 104+ anatomical structures in CT/MR volumes.

### Status

Not wired in VisionServeX engine. VisionServeX delegates to upstream.

### Install

```bash
pip install TotalSegmentator
# Requires: nibabel, nnunetv2, torch, SimpleITK
# GPU strongly recommended (CPU possible but slow)
```

### Example

```bash
# Check if installed:
visionservex medical validate totalsegmentator

# Segment (VisionServeX returns next_step command):
visionservex medical segment totalsegmentator ct_volume.nii.gz --out /tmp/ts_out

# Actual segmentation (upstream CLI):
TotalSegmentator -i ct_volume.nii.gz -o /tmp/ts_out
```

### License

Apache-2.0 code. Some model weight subsets have non-commercial/research-only conditions. Verify upstream README before commercial use.

---

## nnU-Net v2

nnU-Net v2 is a self-configuring medical image segmentation framework.

### Status

Not wired in VisionServeX engine. Requires dataset-specific trained weights.

### Install

```bash
pip install nnunetv2
```

### Example

```bash
# VisionServeX returns delegation instructions:
visionservex medical segment nnunet-v2 volume.nii.gz --out /tmp/nnunet_out

# Actual prediction (nnU-Net CLI):
nnUNetv2_predict -i /path/to/input -o /tmp/nnunet_out -d <DATASET_ID> -c 3d_fullres
```

### License

Apache-2.0 framework; no shipped weights.

---

## MedSAM2

MedSAM2 extends SAM 2 for volumetric and video medical segmentation.

### Status

`expert_sidecar` — not wired in VisionServeX.

### Install

```bash
git clone https://github.com/bowang-lab/MedSAM2.git
cd MedSAM2 && pip install -e .
```

---

## SAM-Med2D

SAM adapter fine-tuned on 2D medical images across 31 imaging modalities.

### Status

`expert_sidecar` — not wired in VisionServeX.

### Install

```bash
git clone https://github.com/OpenGVLab/SAM-Med2D.git
cd SAM-Med2D && pip install -e .
```

---

## Dependency check

```bash
# Check all medical model deps:
visionservex medical doctor
visionservex medical doctor --json

# Check specific model:
visionservex medical validate medsam
visionservex medical validate totalsegmentator
```

---

## Model recommendation

```bash
visionservex medical recommend --goal ct-segmentation
visionservex medical recommend --goal "prompt-based medical segment"
visionservex medical recommend --goal "3d volumetric"
```

---

## Disclaimer

VisionServeX medical commands are RESEARCH AND EDUCATION ONLY. They do not provide medical diagnosis, treatment recommendation, or clinical guidance. Do not use the outputs for patient care. All models listed here are research-grade and have not been validated for clinical use.

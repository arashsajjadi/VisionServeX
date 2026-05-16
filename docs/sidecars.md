# VisionServeX Expert Sidecar Workflows

Expert sidecars are models that cannot be installed in the core VisionServeX environment because they require conflicting dependencies, special build steps, or large framework installs (OpenMMLab, Detectron2, etc.).

VisionServeX provides dedicated CLI commands to create isolated conda environments, validate installations, and run inference through the sidecar. All sidecar commands return structured errors (with `code`, `message`, `fix` keys) if the sidecar is not set up.

---

## OpenMMLab (RTMDet, RTMPose, InternImage, Co-DINO)

OpenMMLab models require `mmcv`, `mmdet`, and optionally `mmrotate`, `mmpose`. These are incompatible with some versions of PyTorch and should be installed in an isolated environment.

### Quick start

```bash
# Check if OpenMMLab is installed
visionservex openmmlab validate

# Create isolated conda env
visionservex openmmlab create-env --name vsx-openmmlab --python 3.10

# Activate and install
conda activate vsx-openmmlab
pip install -U openmim
mim install mmcv
mim install mmdet
```

### Workflow: RTMDet-R (aerial OBB)

```bash
# In the vsx-openmmlab env:
mim install mmrotate
# Run OBB inference via VisionServeX delegate command:
visionservex aerial predict rtmdet-r2-s aerial_image.jpg --out /tmp/aerial_out
# Returns: structured error with next_step if not set up
```

### Workflow: RTMPose

```bash
# In the vsx-openmmlab env:
mim install mmpose
# Pose inference:
visionservex aerial predict rtmpose-s image.jpg --out /tmp/pose_out
```

### Workflow: Co-DINO (instance segmentation)

```bash
# Co-DINO requires OpenMMLab mmdet + mmcv
# In vsx-openmmlab env:
mim install mmdet
# Run via expert command:
visionservex expert openmmlab-smoke-test co-dino-inst-vit-l-coco image.jpg
```

### Structured error when not set up

```json
{
  "code": "OPENMMLAB_REQUIRED",
  "message": "mmdet is not installed",
  "fix": "pip install -U openmim && mim install mmcv mmdet"
}
```

---

## Detectron2 (MaskDINO)

MaskDINO requires Facebook's Detectron2, which is a large framework with its own PyTorch build requirements.

### Install

```bash
# Create a dedicated env
conda create -n vsx-detectron2 python=3.10 -y
conda activate vsx-detectron2

# Install Detectron2 (GPU-specific — check your CUDA version)
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Clone and install MaskDINO
git clone https://github.com/IDEA-Research/MaskDINO.git
cd MaskDINO && pip install -e .
```

### MaskDINO workflow

```bash
# Download MaskDINO-SwinL COCO checkpoint:
# https://github.com/IDEA-Research/MaskDINO#model-zoo

# Inference (Detectron2 native):
python demo/demo.py \
  --config-file configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml \
  --input image.jpg \
  --opts MODEL.WEIGHTS /path/to/maskdino_checkpoint.pth

# VisionServeX will return delegate instructions:
visionservex expert maskdino image.jpg --out /tmp/out
```

### Structured error when not set up

```json
{
  "code": "DETECTRON2_REQUIRED",
  "message": "Detectron2 is not installed.",
  "fix": "See docs/sidecars.md for MaskDINO setup instructions."
}
```

---

## Florence-2 (Isolated environment)

Florence-2 requires `flash-attn` and/or specific `transformers` versions that can conflict with other models. An isolated environment is recommended.

### Quick start

```bash
# Use the florence2 create-env command:
visionservex florence2 create-env --name vsx-florence2 --python 3.11

# This outputs the exact conda commands. Then:
conda activate vsx-florence2

# Validate:
visionservex florence2 doctor
```

### Manual install

```bash
conda create -n vsx-florence2 python=3.11 -y
conda run -n vsx-florence2 pip install 'visionservex[hf]'
conda run -n vsx-florence2 pip install flash-attn --no-build-isolation
```

### Running Florence-2

```bash
# In the vsx-florence2 env:
visionservex florence2 predict florence-2-base image.jpg --task '<OD>'
visionservex florence2 predict florence-2-large image.jpg --task '<CAPTION>'
visionservex florence2 predict florence-2-base image.jpg --task '<OCR_WITH_REGION>'

# Available tasks: <CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>,
#   <OD>, <DENSE_REGION_CAPTION>, <CAPTION_TO_PHRASE_GROUNDING>,
#   <OCR>, <OCR_WITH_REGION>
```

### Structured error when not set up

```json
{
  "code": "FLORENCE2_EXTRA_REQUIRED",
  "message": "florence2 extra not installed",
  "fix": "pip install 'visionservex[hf]' && pip install flash-attn --no-build-isolation"
}
```

---

## Anomalib (PatchCore, industrial anomaly)

Anomalib is kept out of VisionServeX core because it pulls many heavy ML dependencies. Use the `create-env` command to set up a dedicated environment.

### Quick start

```bash
# Generate and run conda recipe:
visionservex anomaly create-env --name vsx-anomaly --python 3.11 --json
# Then run each command in the output

# Or step-by-step:
conda create -n vsx-anomaly python=3.11 -y
conda run -n vsx-anomaly pip install anomalib
conda run -n vsx-anomaly pip install 'visionservex[anomaly]'
```

### Train PatchCore on your data

```bash
conda activate vsx-anomaly

# Train (needs a folder of NORMAL images for unsupervised anomaly detection)
visionservex anomaly train patchcore \
  --data /path/to/normal_images \
  --out /tmp/vsx_patchcore

# Dry-run to see the plan without training:
visionservex anomaly train patchcore \
  --data /path/to/normal_images \
  --out /tmp/vsx_patchcore \
  --dry-run
```

### Predict

```bash
visionservex anomaly predict /tmp/vsx_patchcore image_to_score.jpg
```

### Install options

```bash
# Show all install options (native pip, conda, docker):
visionservex anomaly install-help
visionservex anomaly install-help --json
```

### Structured error when not set up

```json
{
  "code": "ANOMALIB_REQUIRED",
  "message": "anomalib is not installed",
  "fix": "pip install 'visionservex[anomaly]'   (or: pip install anomalib)"
}
```

### Supported algorithms

- `patchcore` — PatchCore (CVPR 2022); memory-bank approach; no training needed
- `padim` — PaDiM; patch distribution modeling
- `fastflow` — FastFlow; normalizing flows
- `efficientad` — EfficientAD; millisecond latency
- `winclip` — WinCLIP; zero/few-shot
- `draem` — DRAEM; reconstruction-based
- `reverse_distillation` — Reverse distillation (CVPR 2022)

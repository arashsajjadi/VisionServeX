# VisionServeX Pose Estimation and OBB Detection

Pose estimation and Oriented Bounding Box (OBB) detection workflows. Both require the OpenMMLab sidecar for full capability.

---

## Supported Models

| Model | Task | Status | Install |
|-------|------|--------|---------|
| RTMPose (s/m/l/x) | Pose estimation | expert_sidecar | OpenMMLab MMPose |
| RTMDet-R-s | OBB detection | expert_sidecar | OpenMMLab MMRotate |
| RTMDet-R2-s | OBB detection | expert_sidecar | OpenMMLab MMRotate |
| Oriented R-CNN | OBB detection | expert_sidecar | OpenMMLab MMRotate |

---

## RTMPose (Pose Estimation)

RTMPose is OpenMMLab's real-time multi-person pose estimation model. It achieves state-of-the-art accuracy on COCO pose benchmarks.

### Status

`expert_sidecar` — requires OpenMMLab MMPose.

### Install

```bash
# Create OpenMMLab env
conda create -n vsx-openmmlab python=3.10 -y
conda activate vsx-openmmlab

pip install -U openmim
mim install mmcv
mim install mmdet   # for person detector
mim install mmpose  # for RTMPose

# Validate
visionservex openmmlab validate
```

### Available checkpoints

RTMPose checkpoints are available in the MMPose model zoo:
- `rtmpose-s-coco` — 68.5 AP @ COCO, fast
- `rtmpose-m-coco` — 74.9 AP @ COCO, balanced
- `rtmpose-l-coco` — 76.1 AP @ COCO, accurate
- `rtmpose-x-coco` — 76.8 AP @ COCO, highest accuracy

### Inference

```bash
# VisionServeX delegate command (in vsx-openmmlab env):
visionservex aerial predict rtmpose-s image.jpg --out /tmp/pose_out

# If sidecar not set up, returns:
# {"code": "OPENMMLAB_REQUIRED", "message": "mmpose not installed", ...}

# Direct MMPose inference:
python mmpose/demo/topdown_demo_with_mmdet.py \
  mmdetection/configs/rtmdet/rtmdet_nano_320-8xb32_coco-person.py \
  /path/to/person_det.pth \
  mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py \
  /path/to/rtmpose_s.pth \
  --input image.jpg \
  --output-root /tmp/pose_out
```

### COCO Pose benchmark

RTMPose uses 17-keypoint COCO format. OKS (Object Keypoint Similarity) is the primary metric.

---

## RTMDet-R / RTMDet-R2 (OBB Detection)

RTMDet-R and RTMDet-R2 are oriented bounding box detectors for aerial and rotated-object detection.

### Status

`expert_sidecar` — requires OpenMMLab MMRotate.

### Install

```bash
# In vsx-openmmlab env:
mim install mmrotate
```

### Inference

```bash
# VisionServeX aerial command:
visionservex aerial detect aerial_image.jpg \
  --model rtmdet-r2-s \
  --out /tmp/obb_out

# Direct MMRotate inference:
python mmrotate/demo/image_demo.py \
  aerial.jpg \
  configs/rotated_rtmdet/rotated_rtmdet-r_s_9x_dota.py \
  --checkpoint /path/to/rtmdet_r2_s.pth \
  --out-file /tmp/result.jpg
```

### Output format

OBB predictions use the `(cx, cy, w, h, angle)` format where `angle` is in radians from the positive x-axis. VisionServeX normalizes these to `obb_result` objects when the engine is wired.

---

## DOTA Support

DOTA (Dataset for Object deTection in Aerial images) is the primary OBB benchmark:
- DOTA v1.0: 15 categories, ~188,000 instances
- DOTA v1.5: adds small vehicle
- DOTA v2.0: 18 categories, ~1,793,658 instances

RTMDet-R2-s achieves competitive mAP75 on DOTA. Run the benchmark:

```bash
# (requires vsx-openmmlab env with mmrotate)
visionservex aerial benchmark \
  --dataset dota:/path/to/dota \
  --model rtmdet-r2-s \
  --split val
```

---

## Oriented R-CNN

Classic two-stage oriented detector. Slower than RTMDet-R but often more accurate at large scale.

### Install

Same as RTMDet-R (MMRotate).

### Inference

```bash
# In vsx-openmmlab env:
python mmrotate/demo/image_demo.py \
  aerial.jpg \
  configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py \
  --checkpoint /path/to/oriented_rcnn_r50.pth
```

---

## Zero-Shot OBB Alternative

When OpenMMLab is not available, use OWLv2 or Grounding DINO for axis-aligned detection of aerial objects:

```bash
pip install 'visionservex[hf]'

# Open-vocab detection (axis-aligned boxes, not OBB):
visionservex open-vocab owlv2-large-patch14 aerial.jpg \
  --prompt "ship,airplane,vehicle,helicopter,storage tank"

# More detailed with Grounding DINO:
visionservex open-vocab grounding-dino-swin-b aerial.jpg \
  --prompt "ship on water, airplane on runway, military vehicle"
```

---

## Related Commands

```bash
visionservex aerial --help
visionservex openmmlab validate
visionservex openmmlab create-env
visionservex model-zoo blockers --family rtmdet
visionservex model-zoo sources --domain aerial
```

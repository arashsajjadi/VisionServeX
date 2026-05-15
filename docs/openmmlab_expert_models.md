# OpenMMLab expert models

The following VisionServeX models require the OpenMMLab toolchain:

| Model IDs | Task | OpenMMLab package |
|-----------|------|-------------------|
| `rtmpose-t/s/m/l` + 384 variants | Pose/Keypoints | mmpose |
| `rtmdet-r-t/s/m/l`, `rtmdet-r2-t/s/m/l` | OBB | mmdet + mmrotate |
| `co-dino-inst-vit-l-coco`, `co-dino-inst-vit-l-lvis` | Instance segmentation | mmdet |
| `internimage-t/s/b/l/h` | Classification | mmpretrain + custom CUDA ops |

These are marked `difficulty=expert` in the registry. They are **not auto-installed**
and **not recommended for beginners**. Use `rfdetr-*`, `dfine-*`, `grounding-dino-*`,
`swinv2-*`, or `oneformer-*` for most tasks.

---

## Option A: Docker (recommended)

The easiest and most reliable path is the VisionServeX OpenMMLab Docker image,
which pre-installs the full toolchain via `openmim`.

### Build

```bash
docker build -t visionservex-openmmlab:latest \
    -f docker/openmmlab/Dockerfile .
```

### Run

```bash
docker run --gpus all --rm \
    -p 127.0.0.1:8080:8080 \
    -v $HOME/.cache/visionservex:/cache/visionservex \
    -e VISIONSERVEX_AUTH__ENABLED=false \
    visionservex-openmmlab:latest
```

Or with Docker Compose:

```bash
docker compose -f docker/openmmlab/docker-compose.yml up
```

### Verify

```bash
curl http://127.0.0.1:8080/models | jq '.models[] | select(.family == "rtmpose") | .id'
```

---

## Option B: Native install (advanced)

```bash
pip install openmim
mim install mmengine
mim install mmcv
mim install mmdet           # for Co-DINO and RTMDet-R
mim install mmpose          # for RTMPose
mim install mmrotate        # for RTMDet-R/R2 OBB
mim install mmpretrain      # for InternImage via MMPreTrain
```

**InternImage requires custom CUDA ops** that must be compiled on the host:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage/classification
pip install -r requirements.txt
cd ops_dcnv3 && python setup.py install && cd ..
```

This is only needed for InternImage. RTMPose, RTMDet-R, and Co-DINO work
with standard mmcv/mmengine.

---

## RTMPose quick start (after install)

```python
from visionservex import VisionModel

# Requires mmpose installed
m = VisionModel("rtmpose-s", device="cuda")
result = m.predict("person.jpg")
for kp in result.persons[0].keypoints:
    print(kp.name, kp.x, kp.y, kp.score)
```

CLI:

```bash
visionservex predict rtmpose-s person.jpg --save outputs/pose.jpg
```

---

## RTMDet-R/R2 quick start (OBB)

```python
from visionservex import VisionModel

m = VisionModel("rtmdet-r2-s", device="cuda")
result = m.predict("aerial.jpg")
for det in result.detections:
    print(det.label, det.score, det.box.corners())
```

> OBB integration is cautious. We make no benchmark superiority claims.
> Verify upstream checkpoint licenses before commercial use.

---

## Why are these expert models?

1. **Heavy dependency chain.** `mmcv` requires compilation and PyTorch version
   pinning. The toolchain is fragile on non-standard platforms.
2. **Large model sizes.** Co-DINO requires 16 GB+ VRAM.
3. **Custom CUDA ops.** InternImage requires compiling `ops_dcnv3`.
4. **Licensing uncertainty.** Some OpenMMLab checkpoints are research-only.
   Review the upstream license for each checkpoint before deployment.

---

## Alternatives for each task

| Instead of | Use |
|-----------|-----|
| RTMPose | `mock-pose` (testing) |
| RTMDet-R/R2 | `mock-obb` (testing) |
| Co-DINO-Inst | `rfdetr-seg-*`, `oneformer-swin-large` |
| InternImage | `swinv2-*` (similar accuracy, much easier) |

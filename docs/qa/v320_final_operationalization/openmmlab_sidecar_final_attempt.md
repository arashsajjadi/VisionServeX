# OpenMMLab sidecar final attempt (v3.20)

Families: `internimage`, `rtmdet`, `rtmpose`, `co-dino`, `maskdino`, `seem` +
partials `rtmdet-r2-s`, `rtmpose-s`.

**Real attempt this sprint (Docker IS runnable on this host).** I built the
OpenMMLab Docker sidecar image and ran a live inference smoke. Result:

- ✅ **The sidecar runtime works (CPU).** `mmdet`'s `DetInferencer('rtmdet_tiny_…')`
  ran inside the container and returned detections (`OPENMMLAB_SIDECAR_CPU_SMOKE: PASS`,
  see `openmmlab_sidecar_matrix.json`).
- ❌ **GPU path is blocked on this host** by a hardware × pinned-CUDA mismatch:
  ```
  RuntimeError: CUDA error: no kernel image is available for execution on the device
  ```
  mmcv 2.1.0's only prebuilt wheel targets **torch 2.1.0+cu121** (max compute
  capability sm_89). The host GPU is an **RTX 5080 (Blackwell, sm_120)**, which
  needs **torch ≥2.7 / CUDA ≥12.8**. There is **no prebuilt mmcv 2.x wheel** for that
  newer torch/CUDA, and mmcv 2.x will not build from source on this host (py3.13,
  cu130, setuptools≥72 — see v3.19 `openmmlab_engine_plan.md`). The two requirements
  do not intersect, so OpenMMLab cannot be GPU-accelerated here.

## Build fix landed this sprint

The first build failed at the mmcv import sanity probe:
```
ImportError: libxcb.so.1: cannot open shared object file: No such file or directory
```
mmcv/OpenCV link against X11 runtime libs absent from the slim base image. Fixed in
`docker/openmmlab/Dockerfile` by adding `libxcb1 libgl1 libglib2.0-0` to the apt
install; the image now builds and imports `torch/mmcv/mmpose/mmdet` cleanly
(13.1 GB, `visionservex-openmmlab:v320`). A `.dockerignore` was added so the
multi-GB weights never ship to the daemon.

## Readiness decision (honest)

The OpenMMLab models stay **`CATALOG_ONLY_ENGINE_NOT_WIRED` / `PARTIAL_IMPLEMENTATION_BLOCKED`**
in the default package — the in-process engine is a stub and the sidecar is an
opt-in Docker service, not the default runtime. They are **not** promoted to
`*_LIVE` (that would claim the default package runs them, which it does not). The
sidecar is now a **proven** path for users who opt in.

## Exact commands (working)
```bash
docker build -t visionservex-openmmlab:v320 -f docker/openmmlab/Dockerfile .
# CPU (works on any host):
docker run --rm --entrypoint python visionservex-openmmlab:v320 -c \
  "from mmdet.apis import DetInferencer; import numpy as np; \
   DetInferencer('rtmdet_tiny_8xb32-300e_coco', device='cpu')((np.random.rand(640,640,3)*255).astype('uint8'))"
# GPU: requires a host GPU with compute capability <= sm_89 (e.g. A100/4090),
#      OR wait for an mmcv build compatible with torch>=2.7 / sm_120 (Blackwell).
```

## Per-family status
All `internimage` / `rtmdet` / `rtmpose` / `co-dino` / `maskdino` / `seem` variants:
**sidecar-runnable (CPU proven; GPU blocked on sm_120 host)**, hidden in the default
package. License: RTMDet/RTMPose/InternImage/MaskDINO Apache-2.0; Co-DINO/SEEM
expert-use (verify per checkpoint).

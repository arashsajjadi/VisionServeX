# OpenMMLab engine plan (v3.19)

Families `internimage`, `rtmdet`, `rtmpose`, `co-dino`, `maskdino`, `seem` +
partials `rtmdet-r2-s`, `rtmpose-s`. **None operationalized in-process this
sprint** — host-native OpenMMLab is infeasible. They stay
`CATALOG_ONLY_ENGINE_NOT_WIRED` / `PARTIAL_IMPLEMENTATION_BLOCKED`. The Docker
sidecar is the supported path (already scaffolded).

## Why host-native (in-process) is INFEASIBLE this sprint

`mmengine 0.10.7` is installed; **`mmcv`, `mmdet`, `mmpose`, `mmsegmentation` are
NOT.** Installing them on this host fails on three independent walls:

1. **No prebuilt `mmcv` wheel** for `torch 2.11 / cu130` — OpenMMLab's wheel index
   stops far below; building from source needs `nvcc` + the CUDA toolkit.
2. **`mmcv` 2.x refuses `setuptools>=72`** (removed `pkg_resources`); the host has
   modern setuptools.
3. **Python version:** `mmcv` 2.x / `mmdet` 3.x target Python 3.8–3.11; the host
   is **3.13**. PyPI tops out at `mmcv 2.2.0` (sdist only), `mmdet 3.3.0`,
   `mmpose 1.3.2` — none ship cu130/py313 binaries.

Forcing any of these would pollute the base install and likely break it — a hard
violation of the "optional, no base-install breakage" rule.

## The supported path — Docker sidecar (FIXABLE_MODERATE, already built)

`engines/openmmlab_sidecar.py` is an HTTP proxy that forwards by task to a Docker
container (`docker/openmmlab/{Dockerfile,docker-compose.yml,sidecar_app.py}`).
The container pins the verified stack (python 3.10, torch 2.1.0+cu121, mmcv 2.1.0
from OpenMMLab's prebuilt index, mmdet 3.3.0, mmpose 1.3.2) and keeps the host
clean. It was previously smoke-verified for **RTMPose-m + RTMDet-tiny**.

Already cached and ready for the sidecar:
- `~/.cache/torch/hub/checkpoints/`: `rtmdet_tiny_*`, `rtmdet_m_*`,
  `rtmpose-tiny_*`, `rtmpose-m_*` `.pth`.
- `~/.cache/visionservex/sidecars/mmdetection/` (config clone),
  `~/.cache/visionservex/openmmlab/{rtmdet-r2-s,rtmpose-s}/`.

### Exact next step
```
visionservex openmmlab docker-build && visionservex openmmlab docker-run
export VISIONSERVEX_OPENMMLAB_SIDECAR_URL=http://localhost:8090
```
Then `rtmdet`/`rtmpose` can serve pose/obb immediately from the cached
checkpoints. **Do not** attempt host-native `mmcv` on this machine.

## License
RTMDet / RTMPose / InternImage / MaskDINO are **Apache-2.0** (verify per
checkpoint). Co-DINO / SEEM vary (registry flags "expert use, manual checkpoint").
No commercial blocker for the rtm* family once the sidecar is up.

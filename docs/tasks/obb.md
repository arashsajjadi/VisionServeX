# Oriented bounding boxes (OBB)

VisionServeX provides cautious OBB integrations via the OpenMMLab toolchain
(`mmrotate`). We make no claims of OBB benchmark superiority.

## Models

- `rtmdet-r-m` (Apache-2.0, license uncertain on some checkpoints)
- `rtmdet-r2-m` (Apache-2.0, license uncertain on some checkpoints)
- `mock-obb`

## Install

```bash
pip install -U openmim
mim install mmengine mmcv mmdet mmrotate
```

## Python

```python
from visionservex import VisionModel

m = VisionModel("rtmdet-r-m")
r = m.predict("aerial.jpg")
for det in r.detections:
    print(det.label, det.score, det.box.cx, det.box.cy, det.box.theta)
```

`OrientedBox` exposes `corners()` for downstream drawing.

## Notes

Always validate OBB integrations against your own data and verify upstream
checkpoint licensing before commercial use.

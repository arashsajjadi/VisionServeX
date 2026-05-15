# Pose / keypoints

## Models

- `rtmpose-m` (Apache-2.0): RTMPose via MMPose. Strong permissive choice.
- `mock-pose`: deterministic test fixture.

## Install

```bash
pip install -U openmim
mim install mmengine mmcv mmpose
```

## Python

```python
from visionservex import VisionModel

m = VisionModel("rtmpose-m")
r = m.predict("person.jpg")
for person in r.persons:
    for kp in person.keypoints:
        print(kp.name, kp.x, kp.y, kp.score)
r.save("annotated.jpg")
```

## HTTP

`POST /pose` with multipart upload + `model_id`.

## Notes

Keypoint names depend on the upstream RTMPose preset (COCO 17, Halpe 26,
WholeBody, etc.). VisionServeX returns whatever the upstream model produces
under `Keypoint.name`. Pin a specific preset in your application code if you
need stability across model upgrades.

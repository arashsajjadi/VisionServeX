# Grounded / universal segmentation

VisionServeX composes two engines for prompt-driven segmentation:

1. **Grounding DINO** localizes boxes from text prompts.
2. **SAM 2 / SAM 2.1** segments each box.

This pairing — `grounded-sam2` in the registry — is **capability-matched to
prompt-based universal segmentation**. We do not advertise it as a
guaranteed benchmark winner.

Additional integrations:

- `seem-base` (Apache-2.0, license uncertain): expert path; install upstream
  manually.
- `oneformer-coco` (MIT): panoptic / instance / semantic.

## Install

```bash
pip install 'visionservex[grounding]' 'visionservex[sam2]'
# Plus the upstream sam2 package and checkpoints (see installation.md).
```

## Python

```python
from visionservex import VisionModel

m = VisionModel("grounded-sam2")
r = m.predict("image.jpg", prompts=["dog", "leash"])
for seg in r.segments:
    print(seg.label, seg.score, seg.mask.shape)
r.save("annotated.jpg")
```

## HTTP

```bash
curl -X POST http://127.0.0.1:8080/grounded-segment \
  -H "Content-Type: application/json" \
  -d '{
        "model_id": "grounded-sam2",
        "image_b64": "<base64 jpeg>",
        "prompts": ["dog", "leash"]
      }'
```

## Notes

This composed pipeline is heavier than either model alone. Expect
multi-second latencies on CPU and tune `runtime.per_model_concurrency` if
you serve multiple clients concurrently.

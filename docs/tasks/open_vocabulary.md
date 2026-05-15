# Open-vocabulary detection

Detect objects described in natural language without retraining.

## Models

- `grounding-dino-tiny` (Apache-2.0): edge / laptop friendly.
- `grounding-dino-base` (Apache-2.0): GPU recommended.
- `grounding-dino-1.5` (Custom): **external / API-only**. Disabled by default
  in the registry. Review the upstream terms before using.

## Install

```bash
pip install 'visionservex[grounding]'
```

## Python

```python
from visionservex import VisionModel

m = VisionModel("grounding-dino-tiny")
r = m.predict("image.jpg", prompts=["red bicycle", "person on the left"])
for det in r.detections:
    print(det.label, det.score, det.box.to_xyxy())
```

## HTTP

```bash
curl -X POST http://127.0.0.1:8080/open-vocab/detect \
  -H "Content-Type: application/json" \
  -d '{
        "model_id": "grounding-dino-tiny",
        "image_b64": "<base64 jpeg>",
        "prompts": ["red bicycle", "person on the left"]
      }'
```

The endpoint accepts:

- `image_b64`: base64-encoded image bytes;
- or `image_url`: HTTPS URL (requires `inputs.allow_url_inputs=true`, and
  SSRF guard rejects private hosts).

## Notes

`prompts` is a list of strings. The mapping back to `Detection.label` is
done by the upstream model and may include the prompt verbatim or a
post-processed phrase. Treat `label` as descriptive, not as a stable
category id.

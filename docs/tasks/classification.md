# Classification

## Models

- `swinv2-base` (MIT): Swin Transformer V2.
- `internimage-base` (MIT, license uncertain): optional, requires custom
  CUDA ops via OpenMMLab toolchain.
- `mock-classify`: test fixture.

## Python

```python
from visionservex import VisionModel

m = VisionModel("swinv2-base")
r = m.predict("image.jpg")
for label, score in r.top_k[:5]:
    print(label, score)
```

## HTTP

`POST /classify` with multipart upload + `model_id`.

## Notes

`top_k` is a list of `(label, score)` tuples ordered by score descending.
In server JSON, this is flattened as `results: [{"label": ..., "score": ...}, ...]`.

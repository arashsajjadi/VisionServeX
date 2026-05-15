# Python API

```python
from visionservex import VisionModel
```

## `VisionModel(model_id, *, task=None, device=None, precision=None, auto_pull=False)`

Constructs a model wrapper. Looks the registry up, resolves the device and
precision, and creates the engine.

```python
m = VisionModel("mock-detect")
m = VisionModel("grounding-dino-tiny", auto_pull=True, device="cuda", precision="fp16")
```

`auto_pull=True` downloads weights on first use if necessary. The
server-side `auto_pull` is a separate config (`models.auto_pull`).

## Inference

```python
result = m.predict("image.jpg")               # path
result = m.predict(image_bytes)               # bytes
result = m.predict(pil_image)                 # PIL.Image
result = m.predict("image.jpg", prompts=["cat", "dog"])
```

Batched:

```python
results = m.batch_predict(["a.jpg", "b.jpg"])
for r in m.stream(images):
    process(r)
```

## Lifecycle

```python
m.warmup()
m.unload()

with VisionModel("mock-detect") as m:
    r = m.predict("a.jpg")
```

## Introspection

```python
m.info()
# {
#   "id": "mock-detect", "task": "detect", "device": "cpu",
#   "precision": "fp32", "engine": "mock", "backend": "mock",
#   "status": "stable", "implementation_status": "wired",
#   "license": "Apache-2.0", "loaded": False,
#   "cache_path": None, "auto_pull": False
# }
m.entry              # full ModelEntry
m.entry.license      # "Apache-2.0"
m.entry.warnings     # list[str]
```

## Benchmark

```python
m.benchmark("image.jpg", n=20)
# {"n": 20, "p50_ms": ..., "p90_ms": ..., "p99_ms": ..., "device": "cpu", "backend": "..."}
```

## Result schema

All `predict` results inherit from `BaseResult` and expose:

- `kind`, `model_id`, `task`, `image_size`, `device`, `precision`,
  `backend`, `latency_ms`, `model_loaded_from`, `cache_path`,
  `fallback_reason`, `metadata`, `warnings`.
- Methods: `.to_dict()`, `.to_json(indent=None)`, `.to_coco()` (where
  applicable), `.plot()`, `.save(path)`, `.summary()`.

Concrete classes:

```
DetectionResult         { detections: list[Detection] }
SegmentationResult      { segments:   list[Segment] }
PoseResult              { persons:    list[PoseInstance] }
ClassificationResult    { top_k:      list[(label, score)] }
OrientedDetectionResult { detections: list[OrientedDetection] }
OpenVocabularyResult    { prompts: list[str], detections: list[Detection] }
```

## Settings

```python
from visionservex.config import get_settings, reload_settings

s = get_settings()
print(s.server.host, s.auth.enabled, s.models.auto_pull)
reload_settings(server={"host": "0.0.0.0"})
```

## Downloads

```python
from visionservex.registry import default_registry
from visionservex.runtime.downloads import download

entry = default_registry().get("grounding-dino-tiny")
path = download(entry)                        # uses huggingface_hub when present
```

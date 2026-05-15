# VisionServeX Syntax Contract

This document defines the complete CLI/Python/API syntax that VisionServeX guarantees.
All 222 examples are documented and covered by automated tests.

## A. Install / Doctor / Getting Started

```bash
# 1
pip install visionservex
# 2
pip install "visionservex[server]"
# 3
pip install "visionservex[server,hf,rfdetr]"
# 4
pip install "visionservex[all]"
# 5-9
visionservex doctor
visionservex doctor --json
visionservex doctor --fix-suggestions
visionservex getting-started
visionservex status
# 10-12
visionservex version
visionservex devices
visionservex devices --json
# 13-15
visionservex gpu doctor
visionservex gpu smoke-test
visionservex mps smoke-test
# 16-17
visionservex downloads audit
visionservex models-audit
# 18-24
visionservex list-models
visionservex list-models --friendly
visionservex list-models --json
visionservex list-models --task detect
visionservex list-models --task segment
visionservex list-models --can-run
visionservex list-models --easy
# 25-30
visionservex recommend --task detect --simple
visionservex recommend --task segment --device auto
visionservex recommend --task open-vocab --vram 8
visionservex recommend --task classify --device cuda
visionservex recommend --task pose --include-docker
visionservex recommend --task obb --include-docker
```

## B. Model Download / Cache / Suites

```bash
# 31-39: pull individual models
visionservex pull dfine-n
visionservex pull rfdetr-nano
visionservex pull swinv2-tiny
visionservex pull sam2-hiera-tiny
visionservex pull grounding-dino-tiny
visionservex pull grounded-sam2
visionservex pull oneformer-swin-large
visionservex pull rtmpose-s        # shows exact docker/native install command
visionservex pull rtmdet-r2-s      # shows exact docker/native install command
# 40-44: pull flags
visionservex pull dfine-n --dry-run
visionservex pull dfine-n --verify
visionservex pull dfine-n --force
visionservex pull dfine-n --device auto
visionservex pull dfine-n --json
# 45-51: suites
visionservex pull-easy --yes
visionservex pull-recommended --task detect --yes
visionservex pull-recommended --task segment --yes
visionservex pull-suite beginner
visionservex pull-suite gpu-demo
visionservex pull-suite server-demo
visionservex pull-suite full-auto --yes-i-understand-large-downloads
# 52-58: cache
visionservex cache path
visionservex cache list
visionservex cache verify
visionservex cache repair
visionservex cache clean
visionservex cache clean --model dfine-n
visionservex cache clean --all --yes
```

## C. CLI Prediction

```bash
# 59-68
visionservex predict dfine-n image.jpg
visionservex predict dfine-n image.jpg --save outputs/dfine.jpg
visionservex predict rfdetr-nano image.jpg --save outputs/rfdetr.jpg
visionservex predict swinv2-tiny image.jpg
visionservex predict swinv2-tiny image.jpg --top-k 5
visionservex predict sam2-hiera-tiny image.jpg --point 120,180 --save outputs/sam2_point.png
visionservex predict sam2-hiera-tiny image.jpg --box 50,60,300,400 --save outputs/sam2_box.png
visionservex predict grounding-dino-tiny image.jpg --prompt "car, person, dog"
visionservex predict grounding-dino-tiny image.jpg --prompt "red car" --threshold 0.25
visionservex predict grounded-sam2 image.jpg --prompt "car, person" --save outputs/grounded_sam2.jpg
# 70-72
visionservex predict oneformer-swin-large image.jpg --task semantic
visionservex predict oneformer-swin-large image.jpg --task panoptic
visionservex predict rtmpose-s person.jpg --save outputs/pose.jpg
# 73-88: flags
visionservex predict dfine-n image.jpg --device auto
visionservex predict dfine-n image.jpg --device cuda
visionservex predict dfine-n image.jpg --device cpu
visionservex predict swinv2-tiny image.jpg --device mps
visionservex predict dfine-n image.jpg --precision auto
visionservex predict dfine-n image.jpg --precision fp16
visionservex predict dfine-n image.jpg --json
visionservex predict dfine-n image.jpg --save-json outputs/result.json
visionservex predict dfine-n image.jpg --save-image outputs/annotated.jpg
visionservex predict dfine-n image.jpg --auto-pull
visionservex predict dfine-n image.jpg --no-auto-pull
visionservex predict dfine-n image.jpg --debug
visionservex predict dfine-n image.jpg --timeout 30
visionservex batch-predict dfine-n images/ --save-dir outputs/detections
visionservex batch-predict swinv2-tiny images/ --top-k 3 --save-json outputs/classes.json
visionservex batch-predict grounded-sam2 images/ --prompt "car, person" --save-dir outputs/grounded
```

## D. Gateway Commands

```bash
visionservex gateway start
visionservex gateway start --host 127.0.0.1 --port 8080
visionservex gateway start --profile laptop
visionservex gateway start --profile gpu-workstation
visionservex gateway start --profile cpu-safe
visionservex gateway start --profile public-tunnel-safe
visionservex gateway start --auto-pull
visionservex gateway start --preload dfine-n,swinv2-tiny,sam2-hiera-tiny
visionservex gateway status
visionservex gateway doctor
visionservex gateway loaded-models
visionservex gateway memory
visionservex gateway preload dfine-n,swinv2-tiny
visionservex gateway unload dfine-n
visionservex gateway openapi
visionservex gateway client-example
visionservex gateway stop
```

## E. HTTP API Endpoints

```
GET  /health                        GET  /ready
GET  /version                       GET  /devices
GET  /models                        GET  /models/{id}
POST /models/{id}/pull              POST /models/{id}/load
POST /models/{id}/unload
POST /predict                       POST /batch-predict
POST /detect                        POST /segment
POST /segment/b64                   POST /pose
POST /obb                           POST /classify
POST /open-vocab/detect             POST /grounded-segment
POST /predict/annotated
GET  /jobs/{id}                     GET  /jobs/{id}/events
GET  /jobs/{id}/events?sse=true     DELETE /jobs/{id}
GET  /metrics                       GET  /metrics/prometheus
GET  /gateway/status                POST /gateway/warmup
```

## F. Python API — VisionModel

```python
from visionservex import VisionModel

# Basic predict
result = VisionModel("dfine-n").predict("image.jpg")
# Device + precision
result = VisionModel("dfine-n", device="cuda", precision="fp32").predict("image.jpg")
# SAM2 box prompt
result = VisionModel("sam2-hiera-tiny").predict("image.jpg", box=[50, 60, 300, 400])
# SAM2 point prompt (two aliases)
result = VisionModel("sam2-hiera-tiny").predict("image.jpg", points=[[120, 180]], labels=[1])
result = VisionModel("sam2-hiera-tiny").predict("image.jpg", points=[[120, 180]], point_labels=[1])
# Grounding DINO (comma-separated)
result = VisionModel("grounding-dino-tiny").predict("image.jpg", prompt="car, person, dog")
# Grounding DINO threshold
result = VisionModel("grounding-dino-tiny").predict("image.jpg", prompt="red car", threshold=0.25)
# Grounded-SAM2
result = VisionModel("grounded-sam2").predict("image.jpg", prompt="car, person")
# OneFormer task
result = VisionModel("oneformer-swin-large").predict("image.jpg", task="semantic")
# Classification top-k
result = VisionModel("swinv2-tiny").predict("image.jpg", top_k=5)
# Auto-pull
result = VisionModel("dfine-n", auto_pull=True).predict("image.jpg")
# Batch
results = VisionModel("dfine-n").batch_predict(["a.jpg", "b.jpg", "c.jpg"])
# Lifecycle
model = VisionModel("dfine-n"); model.warmup(); result = model.predict("image.jpg"); model.unload()
# Properties
print(model.info(), model.device, model.loaded)
# Result serialization
result.to_dict(); result.to_json(); result.save("out.jpg")
result.save_json("outputs/result.json"); result.save_image("outputs/result.jpg")
```

## G. Python Client

```python
from visionservex import Client, AsyncClient

# Sync
c = Client("http://127.0.0.1:8080")
c.health(); c.models(); c.gateway_status()
c.detect("dfine-n", "image.jpg"); c.classify("swinv2-tiny", "image.jpg")
c.segment("sam2-hiera-tiny", "image.jpg", box=[50, 60, 300, 400])
c.open_vocab_detect("grounding-dino-tiny", "image.jpg", prompts=["car"])
c.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
c.pull("dfine-n"); c.load("dfine-n"); c.unload("dfine-n")
for event in c.job_events("JOB_ID"): print(event)

# With auth and retry
c = Client("http://127.0.0.1:8080", api_key="key", max_retries=3)

# Async
import asyncio
c = AsyncClient("http://127.0.0.1:8080")
result = asyncio.run(c.detect("dfine-n", "image.jpg"))
results = asyncio.run(c.batch_detect("dfine-n", ["a.jpg", "b.jpg"]))
```

## H. Typed Exceptions

```python
from visionservex import (
    VisionServeXError, ModelNotFoundError, InputNotFoundError,
    DeviceUnavailableError, ModelMissingWeightsError, SidecarNotRunningError,
    ExternalModelError, ManualModelError, EngineDependencyError,
)
try:
    VisionModel("dfine-n").predict("image.jpg")
except VisionServeXError as exc:
    print(exc.code, exc.message, exc.hint)
```

## M. Expected Error Behavior

| Scenario | Expected Code |
|----------|---------------|
| Unknown model | `MODEL_NOT_FOUND` |
| Missing input file | `INPUT_NOT_FOUND` |
| CUDA broken | `DEVICE_UNAVAILABLE` or CPU fallback with `fallback_reason` |
| OpenMMLab sidecar down | `SIDECAR_NOT_RUNNING` |
| External/API model | `EXTERNAL_MODEL` |
| Manual model | `MANUAL_MODEL` |
| Server overloaded | `BUSY` (HTTP 503) + `Retry-After` header |
| SSRF attempt | `BAD_URL` (HTTP 422) |
| Image too large | `REQUEST_TOO_LARGE` (HTTP 413) |

All errors include: `code`, `message`, `hint`, `details`. No raw tracebacks unless `--debug`.

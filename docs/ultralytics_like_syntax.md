# Ultralytics-Like Workflow in VisionServeX

VisionServeX provides a familiar Ultralytics-style interface without depending on Ultralytics as a package.

## Python API

```python
from visionservex import VisionModel

# Load a model
model = VisionModel("dfine-x-o365-coco")

# Factory methods
model = VisionModel.from_pretrained("dfine-x-o365-coco")
model = VisionModel.from_registry("rfdetr-large")

# Download weights
model.pull()

# Move to device
model.to("cuda")

# Run inference
results = model.predict("image.jpg", conf=0.25)
results = model.predict("image.jpg", conf=0.25, device="cuda", save="out.jpg")

# Inspect results
results.save("outputs/out.jpg")     # annotated image
results.save("outputs/out.json")    # JSON
results.plot()                       # returns PIL Image
results.to_json()                    # JSON string
results.to_csv()                     # CSV string
results.debug()                      # debug string

# Check capabilities
model.supports("predict")    # {"supported": True, ...}
model.supports("train")      # {"supported": False, "reason": "...", "hint": "..."}
model.supports("val")        # detect task only
model.training_info()        # per-family training capabilities
model.export_info()          # ONNX/TRT/other export support

# Evaluate AP (detection only)
model.val(dataset="yolo:/path/to/coco128", max_images=100)

# Cache and checkpoint info
model.cache_info()
model.checkpoint_info()
model.clear_cache()
model.names  # COCO80 class names for detection models

# Unsupported operations (return structured errors, not tracebacks)
model.train(data="data.yaml")  # VisionModel.train() for most backends
```

## CLI task aliases

```bash
# Detection
visionservex detect dfine-x-o365-coco image.jpg --conf 0.25 --device cuda

# Segmentation
visionservex segment rfdetr-seg-medium image.jpg --save-image out.jpg

# Classification
visionservex classify swinv2-base image.jpg --top-k 5

# Open-vocabulary
visionservex open-vocab grounding-dino-swin-b image.jpg --prompt "car, person, bicycle"

# Grounded segmentation
visionservex grounded-segment grounded-sam2 image.jpg --prompt "person"

# Evaluate AP (detection only)
visionservex val dfine-x-o365-coco --dataset yolo:/path/to/coco128 --max-images 128

# Training (structured error for most models)
visionservex train rfdetr-small --data data.yaml --epochs 50  # rf-detr may support this
visionservex train dfine-s-o365-coco --data data.yaml        # returns TRAINING_NOT_SUPPORTED

# Model lifecycle
visionservex model pull dfine-x-o365-coco --dry-run
visionservex model info dfine-x-o365-coco
visionservex model checkpoint-info dfine-x-o365-coco
visionservex model cache dfine-x-o365-coco
visionservex model list-local

# Capability matrices
visionservex training capabilities --model dfine-x-o365-coco
visionservex export-cmd capabilities --model rfdetr-large

# Video (roadmap v1.5)
visionservex video predict mock-detect video.mp4  # returns VIDEO_NOT_IMPLEMENTED
```

## Differences from Ultralytics

| Feature | Ultralytics | VisionServeX |
|---------|------------|--------------|
| Package dependency | Required | Not required |
| Model families | YOLO series | D-FINE, RF-DETR, SAM, SAM2, Grounding DINO, SwinV2, OneFormer, … |
| Training | Supported | Only RF-DETR (rfdetr package); others return TRAINING_NOT_SUPPORTED |
| Video | Supported | Roadmap v1.5 |
| Export | Multi-format | ONNX (experimental for most); rfdetr supports ONNX |
| AP evaluation | Built-in | Via `model.val()` or `benchmark-competitiveness --dataset` |
| Open-vocabulary | YOLO-World/YOLOE | Grounding DINO, Grounded-SAM2 |
| License | AGPL-3.0 or commercial | Apache-2.0 (all core) |

## Unsupported operations

VisionServeX returns structured errors (not raw tracebacks) for unsupported operations:

```python
model.supports("train")
# {"supported": False, "reason": "TRAINING_NOT_SUPPORTED_IN_HF_BACKEND", "hint": "..."}

model.train("data.yaml")
# raises NotImplementedError with structured message: "TRAINING_NOT_SUPPORTED"
```

Use `model.supports("operation")` before calling to avoid surprises.

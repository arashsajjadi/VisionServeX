# VisionServeX Training Matrix

_Train-ready models only (full lifecycle train→checkpoint→reload→predict→export)._

| Model ID | Family | Dataset format | Reload API | Predict after reload | Export | Validated | Blocker |
|---|---|---|---|:--:|---|:--:|---|
| torchvision-alexnet | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-convnext-tiny | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-densenet121 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-efficientnet-b0 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-mobilenet-v2 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-mobilenet-v3-large | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnet101 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnet152 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnet18 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnet34 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnet50 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-resnext50-32x4d | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| torchvision-wide-resnet50-2 | torchvision-classify | imagefolder | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| libreyolo-rtdetr-r50 | libreyolo | yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| libreyolo-yolov9-s | libreyolo | yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| libreyolo-yolox-s | libreyolo | yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-base | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-large | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-medium | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-nano | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-small | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-seg-medium | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-seg-nano | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |
| rfdetr-seg-small | rfdetr | coco-json, yolo | `VisionModel.from_checkpoint(ckpt, model_id=...)` | ✅ | onnx | ✅ | — |

**24 train-ready models.** All other detectors/classifiers are inference-ready or blocked — see `docs/model_matrix.md` for the exact per-variant blocker.


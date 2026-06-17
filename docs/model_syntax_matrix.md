# VisionServeX Model Syntax Matrix (v3.17.0)

Canonical public API per task. Every model is constructed the same way; the
task-specific method validates the model's registered task and raises a typed
`TaskNotSupportedError` (never silently mis-routes). Discover ids with
`from visionservex.core.model import list_models; list_models(task="detect")` and
the exact per-model syntax with `VisionModel(id).capabilities()["validated_syntax"]`.

## Generic / detection inference

```python
from visionservex import VisionModel

model = VisionModel("libreyolo-yolox-s", device="cuda")
result = model.predict("image.jpg", threshold=0.25, nms_iou=0.5)
# result.metadata -> {raw_count, post_nms_count, nms_applied, checkpoint}
# result.detections -> [Detection(label, score, box, class_id), ...]  (post-NMS)
raw = model.predict("image.jpg", return_raw=True)   # raw proposals (debug)
```

Detection model IDs (task `detect`/`obb`/`open_vocab_detect`): `libreyolo-*`,
`rfdetr-*`, `dfine-*`, `grounding-dino-*`, `owlv2-*`, `owlvit-*`, … (see
`docs/model_matrix.md`).

## Text-prompt grounding / open-vocabulary detection

```python
model = VisionModel("grounding-dino-tiny", device="cuda")
result = model.predict("image.jpg", prompts=["black car", "watch"], threshold=0.25)
```

## Promptable / foundation / semantic segmentation

```python
model = VisionModel("sam2-hiera-small", device="cuda")
result = model.segment("image.jpg", boxes=[[x, y, w, h]], points=[[px, py, 1]])
# result.segments -> [Segment(label, score, box, mask, class_id), ...]
```

Segmentation IDs (task `foundation_segment`/`segment`/`grounded_segment`): `sam-*`,
`sam2-*`, `sam2.1-*`, `efficientsam-*`, `mobilesam`, `hq-sam`, `oneformer-*`,
`rfdetr-seg-*`, `grounded-sam*`.

## Classification

```python
model = VisionModel("torchvision-resnet50", device="cuda")
result = model.classify("image.jpg", top_k=5)
# result.top_k -> [(label, score), ...]
```

Classifier IDs (task `classify`): `torchvision-*`, `convnextv2-*`, `swinv2-*`,
`maxvit-*`, (`internimage-*` are catalog-only — OpenMMLab not wired).

## Embedding / feature extraction + similarity

```python
model = VisionModel("dinov2-base", device="cuda")
ref = model.embed("reference.jpg")        # EmbeddingResult (embedding, shape, normalized)
tgt = model.embed("target.jpg")
score = model.similarity(ref, tgt)        # cosine similarity (float)
```

Embedding IDs (task `embed`): `dinov2-*`, `clip-*`, `siglip-*`, `siglip2-*`.

## Semantic correspondence / in-context segmentation

`VisionModel.correspond(...)` raises `TaskNotSupportedError` — semantic
correspondence is the dedicated INSID3 API (it is region/prototype matching, not
generic segmentation):

```python
from visionservex.vsx import VSX
result = VSX.insid3("insid3-base").segment(query_image, ref_image, ref_mask)
```

## Training / fine-tuning (train-ready models only)

```python
model = VisionModel("libreyolo-rtdetr-r50", device="cuda")
res = model.train("data.yaml", epochs=50, imgsz=640)   # YOLO data.yaml
ckpt = res["checkpoint"]                                # always an existing file
```

Classifier fine-tune (ImageFolder):

```python
res = VisionModel("torchvision-resnet18").train("imagefolder/", epochs=5)
```

Train-ready IDs: `libreyolo-yolox-s`, `libreyolo-yolov9-s`, `libreyolo-rtdetr-r50`,
`rfdetr-*`, `torchvision-*`. Everything else is inference-ready or blocked — see
`docs/training_matrix.md`.

## Checkpoint reload + export

```python
trained = VisionModel.from_checkpoint(ckpt, model_id="libreyolo-rtdetr-r50", device="cuda")
pred = trained.predict("image.jpg")                       # confident, post-NMS
path = trained.export(format="onnx", output_path="m.onnx")
```

## Unsupported methods

Calling a task method on the wrong model raises a typed error:

```python
VisionModel("torchvision-resnet50").embed("x.jpg")
# -> TaskNotSupportedError: [TASK_NOT_SUPPORTED] embed() is not supported for
#    'torchvision-resnet50' (task='classify').
```

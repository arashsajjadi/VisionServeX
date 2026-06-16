# LibreYOLO Training (v3.13.0 → reliability-hardened in v3.14.0)

VisionServeX adds **detector training / fine-tuning** for the permissive
LibreYOLO families, on top of the inference engine shipped in v3.12.0. Training
is backed entirely by the permissive [`libreyolo`](https://github.com/LibreYOLO/libreyolo)
package (MIT code; Apache-2.0 / MIT weights). **No Ultralytics runtime** is
imported on any training, checkpoint, or export path.

> **v3.14.0 — trained-checkpoint reload fixed.** A trained checkpoint whose class
> count differs from COCO-80 reloads via libreyolo's `_rebuild_for_new_classes`,
> which left the inner module in *training* mode — so `predict()` crashed
> (`'NoneType' object has no attribute 'sum'`). v3.14.0 forces eval after load,
> so the **full lifecycle `train → checkpoint → reload → predict → export` is now
> validated live** for every variant below. See the proof artifact
> [`docs/qa/v314_train_reload_matrix.json`](qa/v314_train_reload_matrix.json)
> (regenerate with `python tools/qa/v314_train_reload_matrix.py`).

## What is trainable (full lifecycle validated live, v3.14.0)

| Model id               | Family  | Inference | Train / fine-tune | Reload → predict | Export | License            |
| ---------------------- | ------- | :-------: | :---------------: | :--------------: | :----: | ------------------ |
| `libreyolo-yolox-s`    | YOLOX   |     ✅     |         ✅         |        ✅         |  ONNX  | Apache-2.0         |
| `libreyolo-yolov9-s`   | YOLOv9  |     ✅     |         ✅         |        ✅         |  ONNX  | MIT (MMTechLab)    |
| `libreyolo-rtdetr-r50` | RT-DETR |     ✅     |         ✅         |        ✅         |  ONNX  | Apache-2.0         |
| `libreyolo-dfine-n`    | D-FINE  |     ✅     |         ✅         |        ✅         |  ONNX  | Apache-2.0         |

`libreyolo-dfine-n` is a registry-wired variant added in v3.14.0 (weights from
`LibreYOLO/LibreDFINEn`). D-FINE via LibreYOLO is trainable because it routes
through `LibreYOLOEngine`. The **standalone** Hugging Face `dfine-*` models
(`DFINEEngine`) remain **inference-only** — HF Transformers does not expose a
D-FINE training loop, so those report `TRAINING_NOT_SUPPORTED_IN_HF_BACKEND` and
are not faked.

`_training_capabilities(model_id)["trained_checkpoint_predict_supported"]` is the
truth flag: `train_supported` is never `True` for a family unless a trained
checkpoint can actually be reloaded and used for inference.

### Not trainable (by design)

- **YOLO-NAS** (`libreyolo-yolonas-*`) — Deci proprietary, **non-commercial**.
  Excluded from the trainable families and hard-rejected by both the capability
  table and `LibreYOLOEngine.train()`. It is never commercially trainable here.
- Any non-LibreYOLO family without a real, lawful training backend.

## Dataset format

Training consumes a **YOLO `data.yaml`** describing the dataset:

```yaml
# data.yaml
path: /abs/or/relative/dataset/root   # optional; defaults to the yaml's dir
train: images/train                   # dir of images (or a .txt manifest)
val: images/val                       # 'val' or 'valid'
nc: 3
names: [person, car, dog]             # list, or {0: person, 1: car, 2: dog}
```

VisionServeX validates this contract with `visionservex.data.yolo_dataset`
(`validate_yolo_yaml`) using `yaml.safe_load` only. A `download:` block is
**reported but never executed** (training runs with `allow_download_scripts=False`).
Pass either the `data.yaml` path or a directory containing one.

## Python API

```python
from visionservex import VisionModel

model = VisionModel("libreyolo-yolox-s")
result = model.train("path/to/dataset", epochs=50, batch=16, imgsz=640, device="cuda")
# device omitted -> auto-detect GPU

print(result["best_checkpoint"])   # .../runs/train/<name>/weights/best.pt
print(result["metrics"]["best_mAP50"])
```

`result` contract:

```python
{
  "status": "ok",
  "model_id": "libreyolo-yolox-s",
  "family": "libreyolo",
  "variant": "yolox-s",
  "dataset_format": "yolo",
  "best_checkpoint": ".../weights/best.pt",
  "last_checkpoint": ".../weights/last.pt",
  "save_dir": ".../runs/train/<name>",
  "metrics": {"best_mAP50": ..., "best_mAP50_95": ..., "best_epoch": ...,
              "epochs_completed": ..., "final_loss": ..., "training_time_hours": ...},
  "artifacts": {"weights_dir": ..., "results_csv": ... | null, "args_yaml": ... | null},
}
```

Training **fine-tunes from the COCO-pretrained base weights** that the engine
downloads on demand (never bundled). The class head is rebuilt to the dataset's
`nc` automatically.

## Trained-checkpoint reload (public API, v3.14.0)

Reload a trained checkpoint through the clean public API — no reaching into
internal engine classes:

```python
from visionservex import VisionModel

res = VisionModel("libreyolo-yolox-s").train("data.yaml", epochs=50, device="cuda")
ckpt = res["best_checkpoint"] or res["last_checkpoint"]

# Option A — classmethod factory:
trained = VisionModel.from_checkpoint(ckpt, model_id="libreyolo-yolox-s", device="cuda")
det = trained.predict(image)        # trained weights, normalized DetectionResult

# Option B — in place:
m = VisionModel("libreyolo-yolox-s")
m.load_checkpoint(ckpt, device="cuda")
det = m.predict(image)
```

The family and input size come from `model_id`, never guessed from the file.
A missing checkpoint raises a clean error; there is **no** silent fall back to
the base/pretrained weights. `from_checkpoint`/`load_checkpoint` also work for
`rfdetr-*` (reload via the rfdetr package's `pretrain_weights`); engines without
checkpoint-reload support raise a structured `CHECKPOINT_LOAD_UNSUPPORTED`.

### RF-DETR note

RF-DETR trains/fine-tunes via the mature `rfdetr` package's own API
(`model.train(dataset_dir=...)`, COCO format) — VisionServeX does not wrap that
loop, so `VisionModel("rfdetr-nano").train(...)` returns a structured
`TRAIN_VIA_NATIVE_API` pointer rather than running. A resulting checkpoint
reloads for inference here via `VisionModel.from_checkpoint(ckpt,
model_id="rfdetr-nano")`.

## Export

```python
model = VisionModel("libreyolo-yolox-s")
path = model.export(format="onnx", output_path="out/yolox_s.onnx")
```

ONNX is the supported, tested export format in v3.13.0. The libreyolo package
also supports TorchScript / TensorRT / OpenVINO; those are *backend-supported but
not surfaced/tested* in VisionServeX v3.13 and are reported as such by
`visionservex export capabilities --model <id>`.

## CLI

```bash
# Validate config + dataset without training (CI-safe; NOT proof of training):
visionservex training train libreyolo-yolox-s --data path/to/data.yaml --dry-run --json

# Real training:
visionservex training train libreyolo-yolox-s --data path/to/data.yaml \
    --epochs 50 --batch 16 --imgsz 640 --device cuda --json

# Capability probe:
visionservex training capabilities --model libreyolo-yolox-s --json
visionservex export capabilities --model libreyolo-yolox-s --json
```

Unsupported families return a structured `TRAINING_NOT_SUPPORTED` error (exit 2),
never a raw traceback.

## Live smoke (optional)

A real one-epoch training smoke is gated behind an env flag and is **not** part
of CI (it needs a GPU and pulls base weights):

```bash
VSX_LIVE_LIBREYOLO_TRAIN=1 visionservex training train libreyolo-yolox-s \
    --data tests/assets/tiny_yolo --epochs 1 --imgsz 320 --device cuda
```

## Legal posture

- **No Ultralytics.** The training, checkpoint, and export paths import neither
  `ultralytics` nor any AGPL/GPL runtime. LibreYOLO is the permissive
  alternative to AGPL Ultralytics YOLO.
- Weights are pulled on demand from the LibreYOLO Hugging Face org and are
  **never bundled** (`can_ship_weights=False`).
- YOLO-NAS / non-commercial families are not commercially trainable here.
- Standalone HF D-FINE stays **inference-only** until a real lawful training
  backend exists — this is not faked.

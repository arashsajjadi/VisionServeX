# Anastig upgrade notes — VisionServeX v3.16.0 (LibreYOLO reliability)

## Package pin

```bash
pip install --upgrade 'visionservex[libreyolo]==3.16.0'
```

## What v3.16.0 fixes (the eval≠predict gap)

A LibreYOLO checkpoint that trained + evaluated fine but returned **0 boxes** (or
raw/duplicate floods) through `predict()` after reload. Four package-level causes
fixed: EMA lag, train/infer imgsz mismatch, a missing `best.pt`, and missing NMS.
Result: after adequate training, `predict()` at the **default 0.25 threshold**
returns confident, post-NMS boxes.

## Model picker — which IDs to show

Read truth from `model_capabilities(model_id)` (or `VisionModel(id).capabilities()`).

**Show as TRAIN-READY** (lifecycle validated — `readiness=="train-ready"`,
`validated_lifecycle==True`):

- `libreyolo-yolox-s`
- `libreyolo-yolov9-s`
- `libreyolo-rtdetr-r50`

**Show as INFERENCE-ONLY** (`readiness=="inference-ready"`, `train_supported==False`):

- Larger YOLO/RT-DETR variants — `libreyolo-yolox-m/l/x`, `libreyolo-yolov9-m/c`,
  `libreyolo-rtdetr-r101` — `exact_blocker=="VARIANT_NOT_LIFECYCLE_VALIDATED"`.
  (Same validated engine; training works but isn't individually certified this
  release. You may enable training for these at your own risk, or request
  validation in v3.17.)
- All D-FINE — `libreyolo-dfine-n/s/m/l/x` —
  `exact_blocker=="UPSTREAM_DFINE_FDR_TOPK_CRASH"` (libreyolo D-FINE *training*
  crashes upstream; inference is fine).

**Keep BLOCKED:** `libreyolo-yolonas-*` (Deci non-commercial),
`exact_blocker=="LIBREYOLO_NONCOMMERCIAL_FAMILY"`.

## Health gate — can it be relaxed?

Yes, for the three validated variants. Gate on the capability fields, not a custom
heuristic:

```python
cap = model_capabilities(model_id)
train_ok = cap["readiness"] == "train-ready" and cap["validated_lifecycle"] \
           and cap["trained_checkpoint_predict_supported"] and cap["post_nms_predict_supported"]
```

After training, verify the produced checkpoint predicts ≥1 box at the default
threshold on a held-out image before enabling Save/Run (recommended smoke check).

## Fields to read from `model_capabilities`

| Field | Meaning |
|---|---|
| `readiness` | `train-ready` / `inference-ready` / `catalog-only` / `blocked` |
| `train_supported` | training is certified for this variant |
| `trained_checkpoint_predict_supported` | reload→predict works |
| `post_nms_predict_supported` | predict returns final post-NMS boxes (not raw) |
| `validated_lifecycle` | full lifecycle live-validated |
| `exact_blocker` | why not train-ready (when applicable) |

## Training + reload API (unchanged shape, now reliable)

```python
from visionservex import VisionModel

res = VisionModel("libreyolo-rtdetr-r50").train("data.yaml", epochs=50, imgsz=640, device="cuda")
ckpt = res["checkpoint"]          # NEW: always an existing file (best.pt or last.pt)

m = VisionModel.from_checkpoint(ckpt, model_id="libreyolo-rtdetr-r50", device="cuda")
pred = m.predict(image)           # confident, post-NMS boxes at default threshold
```

- EMA is **off by default** (usable short-fine-tune checkpoints). Pass `ema=True`
  to `train(...)` for long runs.
- predict() automatically infers at the **training imgsz** recorded in the checkpoint.

## NMS / post-NMS output contract

`predict()` returns FINAL detections. The result carries:

```python
pred.metadata == {
  "raw_count": 300,        # proposals before NMS
  "post_nms_count": 2,     # final boxes shown
  "nms_applied": True,
  "checkpoint": "<path or None>",
}
```

- Display `detections` (already post-NMS). Tune with `predict(image, threshold=...,
  nms_iou=..., max_det=...)`.
- For debugging raw proposals: `predict(image, return_raw=True)` (no NMS).

## Summary for the UI

- Flip `libreyolo-yolox-s / yolov9-s / rtdetr-r50` to **Training ready**.
- Show larger YOLO/RT-DETR + all D-FINE as **Inference only** with the `exact_blocker`
  tooltip.
- Keep YOLO-NAS hidden/blocked.
- Render `post_nms_count` / `raw_count` if you surface detection diagnostics.

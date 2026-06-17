# v3.16.0 Preflight

Phase-0 freeze for the LibreYOLO trained-checkpoint reliability sprint.

| Field | Value |
|---|---|
| branch | `main` |
| HEAD (start) | `a65a150` |
| tag at HEAD | `v3.15.0` |
| origin/main | `a65a150` (in sync) |
| current version | `3.15.0` |
| libreyolo | 1.1.1 |

Dirty files: only the pre-existing unrelated `notebook/*` + `scripts/v310_sam3_debug.py`
(outside src/tests/docs/tools/pyproject; excluded from all v3.16 commits).

Legal grep: runtime engines/core/data **CLEAN** (no Ultralytics). Decision: **SAFE TO PROCEED.**

## Root causes reproduced (tools/qa/v316_libreyolo_train_predict_matrix.py)

On a tiny learnable synthetic YOLO dataset, the trained checkpoints showed the exact
eval≠predict gap Anastig reported. Four distinct package-level causes:

1. **EMA lag** — the trainer saved the EMA model (decay 0.9998). For short
   fine-tunes (~48 updates) the EMA is ~99% the *initial* weights → near-base,
   low-confidence predictions while eval mAP (rank-based) looks fine.
2. **Train/infer imgsz mismatch** — a checkpoint trained at imgsz 320 was inferred
   at the model's native 640 → max confidence ~0.25 (below the default threshold).
   Predicting at the trained imgsz → confidence ~0.95.
3. **best.pt missing** — best.pt is only written when val mAP improves; `train()`
   returned a `best_checkpoint` path pointing at a non-existent file.
4. **No NMS on DETR output** — RT-DETR/D-FINE emit up to 300 set-based boxes with
   no NMS; at low thresholds `predict()` flooded duplicates.

## Fixes (all in this release)

- EMA **off by default** in the engine trainer (`ema=True` available for long runs).
- predict() infers at the **trained imgsz** (read from the checkpoint config).
- `best_checkpoint` **falls back to last.pt**; new `checkpoint` key = the usable file.
- predict() applies a **class-aware NMS** safety net (+ `raw_count`/`post_nms_count`,
  `return_raw=True`).

## Validation (25 epochs, CPU)

| Variant | eval mAP50 | predict@0.25 | max score | verdict |
|---|--:|--:|--:|---|
| libreyolo-yolox-s | 1.00 | 1 | 0.959 | PREDICT_OK |
| libreyolo-yolov9-s | 0.95 | 1 | 0.483 | PREDICT_OK |
| libreyolo-rtdetr-r50 | 0.44 | 2 | 0.319 | PREDICT_OK |
| libreyolo-dfine-n | — | — | — | CRASHED (upstream FDR topk) |

→ yolox-s / yolov9-s / rtdetr-r50 are **train-ready** (lifecycle validated).
→ D-FINE training is **BLOCKED** upstream (`UPSTREAM_DFINE_FDR_TOPK_CRASH`); inference-only.

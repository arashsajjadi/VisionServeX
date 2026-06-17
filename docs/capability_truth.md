# VisionServeX Capability Truth (v3.17.0)

`model_capabilities(model_id)` (and `VisionModel(id).capabilities()`) is the single
source of truth for what a model can do. It is derived from the registry, the
licensing policy, the training/export tables, and the engine registry — and is
enforced by the v3.13–v3.17 test suites.

## Schema

```python
{
  "model_id", "family", "task", "tasks": ["detect"], "engine", "backend",
  "readiness": "train-ready | inference-ready | embedding-ready(=inference for embed) | catalog-only | blocked",
  "legal_status", "commercial_safe", "gated", "license_code", "license_weights",
  "pretrained_inference_supported", "pretrained_load_supported",
  "train_supported", "finetune_supported",
  "checkpoint_save_supported", "checkpoint_load_supported",
  "trained_checkpoint_predict_supported", "post_nms_predict_supported",
  "export_supported": ["onnx"], "supported_dataset_formats",
  "validated_lifecycle", "validated_variants", "known_blockers", "exact_blocker",
  "validated_syntax": {"predict": "...", "classify": "...", "train": "...", ...},
}
```

## Rules (test-enforced)

- A model is **train-ready** only if `train_supported and trained_checkpoint_predict_supported
  and validated_lifecycle` (full lifecycle train→checkpoint→reload→predict→NMS→export proven).
- If inference works but training does not → **inference-ready** (`train_supported=False`,
  `exact_blocker` set).
- If the engine is not wired (`implementation_status=stub`) → **catalog-only** with
  `exact_blocker="CATALOG_ONLY: ..."`.
- `commercial_safe=True` only when a curated policy row grants default-safe.
- Embedding models (`task=embed`) are inference-ready when `embed()` works; they are
  **not** detectors/classifiers — use `embed()`/`similarity()`.
- Correspondence is the INSID3 API; `VisionModel.correspond()` raises
  `TaskNotSupportedError` (it is region/prototype matching, not segmentation).
- Detection `predict()` returns **post-NMS** boxes by default; raw proposals only via
  `return_raw=True`.

## Which field to trust for each UI state

| UI label | Condition |
|---|---|
| Training ready | `readiness=="train-ready"` |
| Inference ready | `readiness=="inference-ready"` |
| Embedding ready | `readiness=="inference-ready" and task=="embed"` |
| Similarity ready | embedding-ready (use `similarity()`) |
| Correspondence ready | INSID3 ids only (`insid3-*`) |
| Blocked by license | `legal_status in {noncommercial_restricted, enterprise_license_required}` |
| Needs token | `gated==True` |
| Experimental | `validated_lifecycle==False` (train-not-certified variants) |
| Hidden / admin only | `readiness=="catalog-only"` |

See `docs/model_matrix.md` (full table) and `docs/qa/v317_full_model_matrix/` (matrix JSON).

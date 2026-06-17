# Anastig upgrade notes — VisionServeX v3.17.0 (full model matrix)

## Package pin

```bash
pip install --upgrade "visionservex[all]==3.17.0"
```

## What v3.17.0 adds

- A complete, evidence-backed **model matrix** for all 151 registered models
  (`docs/model_matrix.md`, `docs/qa/v317_full_model_matrix/`).
- Task-specific public API with **typed errors**: `classify()`, `embed()`,
  `segment()`, `similarity()`, `correspond()`, plus `list_models(task=...)`.
- `model_capabilities()` now carries `tasks` + `validated_syntax`.

## Drive the UI from `model_capabilities(model_id)`

Discover ids per task with `list_models(task="detect")`. Build the picker from the
capability object — do not hardcode.

| UI label | Condition | Method to call |
|---|---|---|
| **Training ready** | `readiness=="train-ready"` | `model.train(...)` → `VisionModel.from_checkpoint(...)` |
| **Inference ready** | `readiness=="inference-ready"`, task detect/segment | `model.predict(...)` / `model.segment(...)` |
| **Embedding ready** | `readiness=="inference-ready"`, `task=="embed"` | `model.embed(...)`, `model.similarity(a,b)` |
| **Similarity ready** | embedding-ready | `model.similarity(model.embed(a), model.embed(b))` |
| **Correspondence ready** | `insid3-*` only | `VSX.insid3(id).segment(query, ref, ref_mask)` |
| **Blocked by license** | `legal_status` non-commercial/enterprise | hide / disable |
| **Needs token** | `gated==True` | BYOT token (server-side only) |
| **Experimental** | `train_supported and not validated_lifecycle` | inference now; train at own risk |
| **Hidden / admin only** | `readiness=="catalog-only"` | hide |

## Exact model IDs

- **Train** (validated lifecycle): `libreyolo-yolox-s`, `libreyolo-yolov9-s`,
  `libreyolo-rtdetr-r50`; `rfdetr-*`; `torchvision-*` (ImageFolder classifier
  fine-tune). 24 train-ready total — see `docs/training_matrix.md`.
- **Inference (detect)**: `libreyolo-*` (incl. larger inference-only variants),
  `rfdetr-*`, `dfine-*`, `grounding-dino-*`, `owlv2-*`, `owlvit-*`.
- **Inference (segment)**: `sam-*`, `sam2-*`, `sam2.1-*`, `efficientsam-*`,
  `mobilesam`, `hq-sam`, `oneformer-*`, `rfdetr-seg-*`.
- **Classification**: `torchvision-*`, `convnextv2-*`, `swinv2-*`, `maxvit-*`.
- **Embedding / Similarity**: `dinov2-*`, `clip-*`, `siglip-*`, `siglip2-*`.
- **Correspondence**: `insid3-*` (INSID3 API only).
- **Blocked (partial impl — engine registered, not inference-wired)**: the 3 rows
  `maxvit-tiny-tf-224`, `rtmdet-r2-s`, `rtmpose-s` — `exact_blocker` =
  `NOT_INFERENCE_READY: implementation_status=partial, engine_registered=True`.
- **Hidden / catalog-only**: OpenMMLab `_stub` families (`internimage-*`, most
  `rtmdet-*` / `rtmpose-*`) → `CATALOG_ONLY: engine 'openmmlab' not wired`; and the
  `deim-*` / `deimv2-*` rows → custom loader required from the official repo.
- **Needs token (gated)**: `grounding-dino-1.6` (registry license) and `sam3-base`
  (BYOT license required) — token server-side only, never logged/committed.

## Capability fields to trust

`readiness`, `train_supported`, `trained_checkpoint_predict_supported`,
`post_nms_predict_supported`, `validated_lifecycle`, `gated`, `commercial_safe`,
`exact_blocker`, `validated_syntax`.

## API syntax

See `docs/model_syntax_matrix.md`. Detection `predict()` returns **post-NMS** boxes;
`result.metadata` has `raw_count`/`post_nms_count`/`nms_applied`. Unsupported task
methods raise `TaskNotSupportedError` (code `TASK_NOT_SUPPORTED`) — handle it, don't
expect a silent empty result.

## Token handling

Gated models use a server/worker-side HF token only. The token is never printed,
logged, or committed by VisionServeX.

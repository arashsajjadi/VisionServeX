# Anastig upgrade notes — VisionServeX v3.18 (live truth + visibility contract)

## Package pin

```bash
pip install --upgrade "visionservex[all]==3.18.0"
```

## What v3.18 changes for Anastig

v3.18 makes the catalog **live-verified and self-describing**. Anastig no longer
needs any heuristics about which models to show — it switches on two new fields.

- **`readiness_state`** — the precise readiness (22 states; see
  `docs/capability_truth.md`). This supersedes the coarse `readiness`, which is
  kept byte-stable for backward compatibility.
- **`anastig_visibility`** — the verb Anastig acts on directly.
- **`live_verified_inference` / `live_verified_train`** — whether the model was
  actually run this sprint (not merely capability-derived).
- **`requires_token`**, **`license`**, **`license_class`**, **`blocker`**,
  **`legal_review_required`** — explicit legal/UX fields.
- New top-level imports: `visionservex.list_models`, `visionservex.model_capabilities`.
- New typed method: `VisionModel(id).detect(...)` (open-vocab detectors take `prompts=`).

## Drive the entire UI from `anastig_visibility`

| `anastig_visibility` | Anastig behaviour |
|---|---|
| `show_train` | Show train **and** inference UI (`TRAIN_READY_LIVE`). |
| `show_inference` | Show inference UI (`INFERENCE_READY_LIVE` / `OPEN_VOCAB_READY_LIVE` / `VLM_READY_LIVE`, or a train-derived model whose inference is live). |
| `show_embedding` | Show embedding/similarity UI (`EMBEDDING_READY_LIVE`). |
| `show_segmentation` | Show segmentation UI (`SEGMENTATION_READY_LIVE`). |
| `show_token_required` | Show BYOT/token UI (`GATED_TOKEN_REQUIRED`). Never run without the user's token. |
| `hide` | Hide entirely (catalog-only, custom-loader, partial, dependency/weights/crash/oom, **or derived-not-yet-live**). |
| `blocked_admin_only` | Hide from end users; visible to admins (legal-review, license-blocked, unknown-review). |

**Hard rule:** only `*_READY_LIVE` states are usable-by-default. A
`*_DERIVED_NEEDS_LIVE_CONFIRMATION` model is capability-derived and **not yet
live-verified** — keep it hidden until it is promoted to `*_LIVE`.

## The machine-readable contract

`docs/anastig_model_allowlist_v318.json` is generated straight from
`model_capabilities()`. Buckets: `train_ready_live`, `inference_ready_live`,
`embedding_ready_live`, `segmentation_ready_live`, `open_vocab_ready_live`,
`gated_token_required`, `hidden_catalog_only`, `blocked`, `license_blocked`.
Full prose: `docs/anastig_model_contract_v318.md`.

## v3.18 live results (what is actually proven)

- **16 `TRAIN_READY_LIVE`** (full train→checkpoint→reload→predict→ONNX export):
  `libreyolo-yolox-s` / `libreyolo-yolov9-s` / `libreyolo-rtdetr-r50` + all 13
  `torchvision-*` classifiers.
- **8 `TRAIN_READY_DERIVED`** (inference-live, native trainer not smoke-run):
  `rfdetr-*`. These show as `show_inference`.
- **~85 live inference-ready** across detect / segment / embed / open-vocab.
- **1 `GATED_TOKEN_REQUIRED`**: `sam3-base` (BYOT).
- **Hidden/blocked**: OpenMMLab `_stub` families, `deim-*`/`deimv2-*`/`rtdetrv4-*`
  (custom loader), 3 partial rows, Florence-2 (`DEPENDENCY_MISSING`), OneFormer
  ConvNeXt (`WEIGHTS_MISSING`), `medsam`/`hq-sam` (live but `blocked_admin_only`
  pending legal review).

## Token handling (unchanged, still strict)

Gated models use a server/worker-side HF token only. The token is never printed,
logged, committed, or embedded in any `model_capabilities()` payload (enforced by
`tests/live/test_v318_gated_models.py`).

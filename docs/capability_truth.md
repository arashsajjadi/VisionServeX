# Capability Truth (v3.18)

`visionservex.model_capabilities(model_id)` is the **single source of truth** for
every model. Anastig and any downstream UI drive their entire model experience
off this one object — there is no hardcoded allowlist beyond optional product
ranking.

```python
import visionservex as vsx
cap = vsx.model_capabilities("libreyolo-yolox-s")
```

## The object

| Field | Meaning |
|---|---|
| `model_id`, `family`, `task`, `engine` | identity |
| `readiness` | legacy coarse bucket: `train-ready` / `inference-ready` / `catalog-only` / `blocked` (kept byte-stable for backward compatibility) |
| **`readiness_state`** | the precise v3.18 state (see taxonomy below) — **canonical** |
| **`anastig_visibility`** | `show_train` / `show_inference` / `show_embedding` / `show_segmentation` / `show_token_required` / `hide` / `blocked_admin_only` |
| `blocker` | one human-readable reason the model is not plug-and-play usable (`None` when live-ready) |
| `license`, `license_class` | canonical effective license + class (`permissive`/`copyleft`/`noncommercial`/`custom_unknown`/`unknown`) |
| `commercial_safe` | granted only by a curated `commercial_safe_core` policy row; never true for copyleft/non-commercial |
| `gated`, `requires_token` | gated weights require the user's own token + license acceptance (BYOT) |
| `legal_review_required` | license flagged for review → hidden from end users |
| `predict_supported`, `live_verified_inference`, `live_verified_train` | what actually ran |
| `train_supported`, `checkpoint_load_supported`, `trained_checkpoint_predict_supported`, `export_supported` | training/reload/export truth |
| `validated_syntax` | the exact public-API call for each supported method |

The v3.13–v3.17 fields (`legal_status`, `license_code`, `license_weights`,
`pretrained_inference_supported`, `post_nms_predict_supported`,
`validated_lifecycle`, `exact_blocker`, `tasks`, …) are all still present.

## Readiness taxonomy (`readiness_state`)

The cardinal rule, enforced by `tests/test_v318_no_ready_without_live_or_derived_flag.py`:

> No state promises readiness unless it is either **live-verified this sprint**
> (`*_READY_LIVE`) or **explicitly flagged derived** (`*_DERIVED_NEEDS_LIVE_CONFIRMATION`).

**Live-ready** (the only states a UI may treat as usable by default):
`TRAIN_READY_LIVE`, `INFERENCE_READY_LIVE`, `EMBEDDING_READY_LIVE`,
`SEGMENTATION_READY_LIVE`, `OPEN_VOCAB_READY_LIVE`, `VLM_READY_LIVE`,
`CORRESPONDENCE_READY_LIVE`.

**Derived** (capability-derived, awaiting live proof → hidden until promoted):
`TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION`,
`INFERENCE_READY_DERIVED_NEEDS_LIVE_CONFIRMATION`.

**Legal / access:** `GATED_TOKEN_REQUIRED`, `LICENSE_BLOCKED`,
`NON_COMMERCIAL_BLOCKED`.

**Not wired / blocked:** `CATALOG_ONLY_ENGINE_NOT_WIRED`, `CUSTOM_LOADER_REQUIRED`,
`DEPENDENCY_MISSING`, `WEIGHTS_MISSING`, `UPSTREAM_CRASH`, `OOM_BLOCKED`,
`TASK_NOT_SUPPORTED`, `PARTIAL_IMPLEMENTATION_BLOCKED`, `UNKNOWN_REVIEW_REQUIRED`.

The coarse `readiness` is a pure, compatible view of `readiness_state` — the two
can never contradict (a `*_LIVE` precise state is always coarse train/inference).

## Evidence chain — how `*_LIVE` is earned

```
tools/qa/v318_live_inference_matrix.py        ─┐  (really runs each model on CPU)
tools/qa/v318_live_train_lifecycle_matrix.py  ─┤
                                               ├─► docs/qa/v318_full_model_truth/*.json   (EVIDENCE, committed)
tools/qa/v318_sync_live_evidence.py           ─┘
                                               └─► src/visionservex/readiness/live_evidence.py  (CONCLUSIONS, baked)
                                                        └─► model_capabilities() reads the baked frozensets
```

`model_capabilities()` never loads a model — it reads the baked frozensets, so it
is fast and weight-free. `tests/test_v318_capability_truth_contract.py` fails CI
if the baked conclusions ever drift from the committed matrices.

A model that was **live-tested and FAILED** is never reported as an optimistic
`*_DERIVED`; it gets its true blocker (`DEPENDENCY_MISSING`, `WEIGHTS_MISSING`,
`UPSTREAM_CRASH`, …). Example: Florence-2 fails under transformers 5.10.2, so it
is `DEPENDENCY_MISSING`, not "inference-ready-derived".

## Honest nuance: train-derived but inference-live

RF-DETR trains via its own package's native COCO trainer, which is too heavy for
a CPU smoke, so its train lifecycle stays `TRAIN_READY_DERIVED_NEEDS_LIVE_CONFIRMATION`.
But its *inference* IS live-verified, so `anastig_visibility` is `show_inference`
(`live_verified_inference=True`, `live_verified_train=False`). The headline state
tells the train story; visibility reflects what is actually usable now.

## v3.18 live verification summary

- **101 / 105** wired, legal, non-gated models passed real CPU inference.
- **16** models passed the full train lifecycle (train → checkpoint → reload →
  predict-after-reload → schema → ONNX export): 3 LibreYOLO detectors + 13
  torchvision classifiers.
- **8** RF-DETR variants are `TRAIN_READY_DERIVED` (native trainer, inference-live).
- Honest blockers: Florence-2 ×2 (`DEPENDENCY_MISSING`), OneFormer DiNAT
  (`DEPENDENCY_MISSING`), OneFormer ConvNeXt (`WEIGHTS_MISSING`, upstream 404).

See `docs/qa/v318_full_model_truth/` for the full machine-readable matrices.

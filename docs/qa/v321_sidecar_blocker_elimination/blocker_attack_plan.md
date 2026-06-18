# v3.21 Blocker attack plan

Every blocked/hidden family with a target outcome before work starts. 49 models
across 13 families are not live in the v3.20 baseline.

| Family/model | Current blocker | Legal? | In-process? | Sidecar? | ONNX? | Adapter? | Target outcome this sprint |
|---|---|---:|---:|---:|---:|---:|---|
| `florence-2-*` (2) | DEPENDENCY_MISSING (py3.13 × transformers≤4.49 tokenizers) | MIT ✓ | no (py3.13 wall) | **yes (py3.11 Docker)** | no | n/a | **Build py3.11 sidecar → VLM_READY_LIVE_SIDECAR** if caption smoke passes |
| `rtmdet-*` (8) | CATALOG_ONLY / PARTIAL | Apache ✓ | no (mmcv/py3.13) | **yes (built+CPU-proven)** | maybe | n/a | **Wire sidecar engine → SIDECAR_CPU_READY_LIVE** (GPU blocked sm_120) |
| `rtmpose-*` (6) | CATALOG_ONLY / PARTIAL | Apache ✓ | no | **yes (sidecar)** | maybe | n/a | **SIDECAR_CPU_READY_LIVE** (pose) |
| `internimage-*` (5) | CATALOG_ONLY | Apache ✓ | no | maybe (sidecar) | no | n/a | sidecar feasibility; else CATALOG_ONLY + exact reason |
| `maskdino-*` (2), `co-dino-*` (2), `seem-*` (2) | CATALOG_ONLY | Apache/expert | no | maybe | no | n/a | exact blocker (config/checkpoint per-model); not promoted without smoke |
| `deim-*` (10) | CUSTOM_LOADER_REQUIRED | Apache/none/DINOv3-caveat | no | maybe (torch2.5.1 env) | maybe (carpedm20) | n/a | real attempt; per-variant DETECT_READY_LIVE_SIDECAR/ONNX or exact blocker |
| `rtdetrv4-*` (4) | CUSTOM_LOADER_REQUIRED | Apache (unverified) | no | maybe | no | n/a | exact blocker (Drive-only weights, no HF) |
| `oneformer-convnext-large` (1) | WEIGHTS_MISSING (HF 404) | MIT | no | no | no | n/a | **WEIGHTS_MISSING_PERMANENT** (no weights exist anywhere loadable) |
| `oneformer-dinat-large` (1) | DEPENDENCY_MISSING (NATTEN API) | MIT | no (GPU-only) | maybe (NATTEN sidecar) | no | n/a | NATTEN sidecar feasibility; else exact GPU/dep blocker |
| `rfdetr-seg-{large,xlarge,2xlarge}` (3) | CATALOG_ONLY (impl=stub) | Apache ✓ | not wired | n/a | n/a | n/a | keep CATALOG_ONLY (variant not wired) |
| `sam3-base` (1) | GATED_TOKEN_REQUIRED | byot | yes (w/ token) | n/a | n/a | n/a | stays gated (no license acceptance) — BYOT contract verified |
| `grounding-dino-1.5/1.6` (2) | external/gated, no local weights | Custom | no | no | no | n/a | stays CATALOG_ONLY/gated (no open weights) |

Cross-cutting fine-tune pressure (separate from the blocked list):
- Foundation segmenters (SAM/SAM2/MobileSAM/HQ-SAM/EfficientSAM): investigate
  mask-decoder / LoRA fine-tune; honest `NOT_TRAINABLE_BY_DESIGN` /
  `FINE_TUNE_NOT_IMPLEMENTED_UPSTREAM` if not feasible.
- Embedding (dinov2/clip/siglip): investigate full-backbone / adapter beyond the
  v3.20 linear probe; `fine_tune_kind` becomes explicit.

**Primary wins targeted:** Florence-2 sidecar (2), OpenMMLab rtmdet+rtmpose sidecar
(14). Everything else gets an exact, reproduced outcome.

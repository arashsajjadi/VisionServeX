# v3.17.0 Preflight — full model matrix sprint

| Field | Value |
|---|---|
| branch | `main` |
| HEAD (start) | `eaf5624` |
| tag at HEAD | (post-v3.16.0 CI patch) |
| origin/main | `eaf5624` (in sync) |
| current version | `3.16.0` |
| registry size | 151 models |

Dirty files: only pre-existing unrelated `notebook/*` + `scripts/v310_sam3_debug.py`
(excluded from all v3.17 commits). Legal grep on runtime engines/core/data/runtime:
**CLEAN**. Decision: **SAFE TO PROCEED.**

## Scope + honesty statement

The full catalog (151 models) cannot all be live weight-downloaded/gated-tested in one
run (resource-safety + gated access). The matrix is therefore explicit:

- **Live-verified for ALL models (no downloads):** instantiate, capabilities, engine
  registration, license, dependency importability.
- **Live-verified inference (fresh, this run):** `torchvision-resnet50` (classification,
  top-5) and `libreyolo-yolox-s` (detection, 1 box) — a classifier + a detector across
  the two main runnable families.
- **Capability-DERIVED (not claimed live):** inference/train/reload/export for the
  remaining models — backed by engine wiring + the repo's `real_model`-marked smoke
  suite (`tests/test_model_smoke_matrix.py`, `tests/test_real_model_smoke.py`) and the
  v3.15/v3.16 lifecycle proofs. The matrix marks `live_verified: true|false` per stage.

Nothing is marked ready beyond what `model_capabilities()` (test-enforced) asserts.

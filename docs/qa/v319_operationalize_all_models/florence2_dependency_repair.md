# Florence-2 dependency repair (v3.19)

**Models:** `florence-2-base`, `florence-2-large` (Microsoft, **MIT**, task `vlm`).
**Outcome this sprint:** **NOT operationalized — kept `DEPENDENCY_MISSING`** with the
exact blocker below. Not faked, not force-shimmed (a fragile multi-shim would
destabilize the SAM3 path).

## Root cause — Florence-2 remote code targets transformers 4.x

Florence-2 ships custom modeling code (`trust_remote_code=True`) written against
the transformers **4.x** surface. The environment runs **transformers 5.10.2**
(required by SAM3 — `[hf]` extra pins `transformers>=5.0`). The remote code breaks
on a *cascade* of 5.x removals. Reproduced live this sprint, in order:

1. `from transformers import BartTokenizerFast` — **removed from the package root**
   in v5 (import-time `ImportError`). *(Not addressed by the existing engine shims.)*
2. `Florence2LanguageConfig.forced_bos_token_id` — moved to `GenerationConfig`;
   `PretrainedConfig` no longer stores it (`AttributeError` at config build).
3. `Florence2ForConditionalGeneration._supports_sdpa` — removed/renamed in v5
   (`AttributeError` during model `__init__`).
4. Legacy tuple `past_key_values` protocol + custom `_reorder_cache` — the
   `from_legacy_cache`/`to_legacy_cache` bridge was removed in v5; only `Cache`/
   `DynamicCache`/`EncoderDecoderCache` objects exist (fails inside `.generate()`).

The existing `engines/florence2.py` shims cover #2 and #3 partially; #1 and #4 are
**not** handled, and patching one reveals the next (whack-a-mole).

## Why a version pin is infeasible

`transformers>=4.40,<5.0` exists and would satisfy Florence-2, **but SAM3
(`Sam3Model`) only exists in transformers 5.x** — the two are **mutually
exclusive in one install**. Downgrading transformers would break SAM3 / SAM2 /
CHMv2 (DINOv3 depth) and every other 5.x-only path. This is why the engine
currently hard-blocks rather than downgrades.

## Reproduction

```python
# transformers 5.10.2 — fails at step (1) without a BartTokenizerFast alias,
# then (2), then (3), then (4) inside generate():
from transformers import AutoProcessor, AutoModelForCausalLM
proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
# -> ImportError: cannot import name 'BartTokenizerFast' from 'transformers'
```

## The two real paths (next work)

- **(A) Vendored multi-shim, keep transformers 5.x — FIXABLE_MODERATE but fragile.**
  Patch all four APIs before/around load: register a `BartTokenizerFast`/
  `BartTokenizer` alias on the `transformers` root **before** the remote
  `processing_florence2` import; add `forced_bos_token_id`/`forced_eos_token_id`
  and `_supports_sdpa` shims; and bridge the cache (force `use_cache=False`, or map
  the legacy tuple protocol onto `EncoderDecoderCache`). High maintenance — breaks
  again on each transformers bump; not shipped to avoid silent regressions.
- **(B) Isolated `transformers<5.0` sidecar — CLEAN but heavier.** Run Florence-2
  in its own conda/venv pinned to `transformers>=4.40,<5.0`, exposed over the
  existing sidecar pattern (`sidecars/manager.py`). No conflict with the main
  env's SAM3. Recommended if Florence-2 becomes a product requirement.

**License:** MIT — clean for commercial use once a runtime path exists. The
blocker is purely the transformers-version cascade, not legal.

# Florence-2 final attempt (v3.20)

**Models:** `florence-2-base`, `florence-2-large` (Microsoft, MIT, task `vlm`).
**Outcome:** **NOT operationalized — `DEPENDENCY_MISSING`.** Two real attempts this
sprint, both reproduced a hard wall. The only viable path is a **Python 3.10/3.11
sidecar**, which this Python-3.13 host cannot host in-process.

## Attempt 1 — isolated venv on transformers 5.x (host's version)
Florence-2's remote code (`trust_remote_code`) breaks on a cascade of 5.x removals
(BartTokenizerFast import → `forced_bos_token_id` → `_supports_sdpa` → legacy cache),
documented in v3.19. Not pursued further (would need a fragile multi-shim and risks
SAM3).

## Attempt 2 — isolated venv pinned to transformers < 5.0 (this sprint)
```bash
python -m venv --system-site-packages florence2_venv
florence2_venv/bin/pip install "transformers>=4.40,<5.0" einops timm
```
- The resolver installed **transformers 4.57.6** (newest 4.x). Florence-2 still failed:
  ```
  AttributeError: 'Florence2ForConditionalGeneration' object has no attribute '_supports_sdpa'
  ```
  — `_supports_sdpa` was already removed in late-4.x (the attention refactor), so the
  remote code needs an **early** 4.x (≤ ~4.49).
- Pinning `transformers==4.44.2` (a known-good Florence-2 version) **fails to install on
  Python 3.13**:
  ```
  × Failed to build installable wheels for some pyproject.toml based projects
  ╰─> tokenizers
  ```
  The old `tokenizers` that transformers 4.44.x pins has **no Python-3.13 wheel** and its
  Rust build fails in this environment.

## Conclusion (reproduced, evidence-backed)
The intersection of "transformers old enough for Florence-2's remote code
(`_supports_sdpa` + legacy cache, ≤ ~4.49)" and "a `tokenizers` build that works on
**Python 3.13**" is empty on this host. Florence-2 is therefore not in-process runnable
here at all.

## Exact next work
Run Florence-2 in a **Python 3.10/3.11 sidecar** (conda/venv or container) pinned to
`transformers>=4.40,<4.50` + a matching `tokenizers` (prebuilt wheels exist for py3.10/3.11):
```bash
conda create -n vsx-florence2 python=3.11 -y
conda run -n vsx-florence2 pip install "transformers==4.44.2" einops timm pillow torch
# expose over the existing sidecar pattern (sidecars/manager.py)
```
The `florence2` / `vlm-legacy` extras in `pyproject.toml` already pin
`transformers>=4.40,<5.0` for exactly this isolated-env use. License (MIT) is clean —
the blocker is purely the Python-3.13 × old-transformers/tokenizers incompatibility.

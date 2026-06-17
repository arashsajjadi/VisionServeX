# Gated models final contract (v3.20)

No license bypass. BYOT only. Token never printed/logged/committed (verified).

| Model | readiness_state | anastig_visibility | requires_token | commercial_safe |
|---|---|---|:-:|:-:|
| `sam3-base` | `GATED_TOKEN_REQUIRED` | `show_token_required` | yes | no |
| `grounding-dino-1.6` | `CATALOG_ONLY_ENGINE_NOT_WIRED` | `hide` | yes | no |

- **`sam3-base`** — curated BYOT policy + a real token-gated runtime; Anastig shows the
  token UI. Runs only after the user supplies their own HF token and accepts Meta's SAM3
  license. Not live-run this sprint (no license acceptance available — honestly gated).
- **`grounding-dino-1.6`** — `requires_auth` but no wired local engine; a token alone
  would not make it run → hidden, with `requires_token` still flagged.

## Token-safety (verified this sprint)
- `hf_redact_token()` shows only first-3/last-2 chars; middle never revealed
  (`tests/test_v320_token_redaction.py`).
- No `model_capabilities()` payload contains a raw token (`raw_token in repr(cap) == False`).
- `token_never_logged == True` for all 151 models.
- No token string is printed, logged, or committed anywhere in this sprint's output
  (the local token is read but never echoed).

## Anastig BYOT UI
Show the token form **only** for `anastig_visibility == "show_token_required"`. Never
enable a gated model by default; never treat `gated=True` as commercial-safe; never
show `anastig_train_visibility == show_train` for a gated model.

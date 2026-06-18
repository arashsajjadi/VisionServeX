# Gated-model contract (v3.19)

VisionServeX never bypasses a license gate, never enables a gated model by
default, and never logs/commits/exposes a token.

## Gated models

| Model | readiness_state | anastig_visibility | requires_token | commercial_safe |
|---|---|---|:-:|:-:|
| `sam3-base` | `GATED_TOKEN_REQUIRED` | `show_token_required` | yes | no |
| `grounding-dino-1.6` | `CATALOG_ONLY_ENGINE_NOT_WIRED` | `hide` | yes | no |

- **`sam3-base`** carries a curated BYOT policy row and a real token-gated runtime
  (the BYOT path), so it is `GATED_TOKEN_REQUIRED` — Anastig shows the BYOT/token
  UI; the model only runs once the user supplies their own HF token **and** accepts
  Meta's SAM3 license. Not live-run this sprint (no license acceptance available).
- **`grounding-dino-1.6`** is `requires_auth` but has no wired local engine, so a
  token alone would not make it run — honestly `CATALOG_ONLY_ENGINE_NOT_WIRED`
  (hidden), with `requires_token` still flagged.

## Token-safety guarantees (verified this sprint)

- `hf_get_token(redact=True)` / `hf_redact_token()` reveal only the first 3 and
  last 2 characters; every user-visible path uses the redacted form.
- A `model_capabilities()` payload for a gated model **never** contains the raw
  token (verified: `raw_token in repr(capability) == False`).
- No raw token is printed, logged, or committed anywhere in this sprint's output.
- Live gated tests are env-gated (`VSX_LIVE_GATED_MODELS=1`) and never run by
  default; the unconditional contract assertions are weight-free.

## Anastig BYOT UI contract

- Show the BYOT/token form **only** for `anastig_visibility == "show_token_required"`.
- Never enable a gated model by default; never treat `gated=True` as commercial-safe.
- After the user supplies a token + accepts the upstream license, route `sam3-base`
  through the BYOT runtime — do not assume the standard engine path.

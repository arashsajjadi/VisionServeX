# Gated / Auth Models — Bring Your Own Token (BYOT)

Some high-value models are **gated**: their weights are not publicly
redistributable, or they are served only via a paid API. VisionServeX supports
them through a strict **BYOT** policy:

- **No mirrored gated weights.** VisionServeX never bundles, caches publicly, or
  re-hosts gated checkpoints.
- **Token per user, at runtime.** You provide your own access token / API key via
  an environment variable. Tokens are redacted from all logs, reports, and
  artifacts (`visionservex.security` redaction; verified by
  `tests/test_security_privacy.py` and `v3_token_leak_scan.json`).
- **`final_state` stays `auth_required` / `external_api_only`** until you supply
  valid credentials and the provider's terms permit the run.

## Gated targets

| model | mechanism | env var | weights license | core status |
|---|---|---|---|---|
| `sam3-base` | HF gated access request | `HF_TOKEN` | Meta custom "SAM License" (restrictive) | `auth_required` (BYOT) |
| `grounding-dino-1.5` | DeepDataSpace API | `DDS_API_TOKEN` | Apache-2.0 client; API-served | `auth_required` (BYOT) |
| `grounding-dino-1.6` | DeepDataSpace API | `DDS_API_TOKEN` | Apache-2.0 client; API-served | `auth_required` (BYOT) |
| `grounding-dino-1.5-pro` | DeepDataSpace API | `DDS_API_TOKEN` | proprietary/closed (API only) | `external_api_only` (BYOT) |
| `grounding-dino-1.6-pro` | DeepDataSpace API | `DDS_API_TOKEN` | proprietary/closed (API only) | `external_api_only` (BYOT) |
| `dino-x-api` | DeepDataSpace API | `DDS_API_TOKEN` | proprietary/closed (API only) | `external_api_only` (BYOT) |

## Workflow

```bash
# 1. obtain access (HF gate or API token) yourself — VisionServeX cannot do this for you
export HF_TOKEN=...            # for sam3-base (request access on the HF model page first)
export DDS_API_TOKEN=...       # for grounding-dino 1.5/1.6 (cloud.deepdataspace.com/apply-token)

# 2. check access (does not run a benchmark, does not leak the token)
visionservex sam3 status
```

If the token is missing the model reports a structured blocker
(`HF_SAM3_ACCESS_NOT_APPROVED`, `DEEPDATASPACE_API_KEY_MISSING`,
`EXTERNAL_API_REQUIRED`) with the exact next step — never a silent failure and
never a fabricated benchmark.

> The "1.5-pro" / "1.6-pro" / DINO-X models are **API-only and proprietary**:
> the weights are never published, so they can only ever be external baselines
> consumed through your own paid API quota — they are not, and cannot be, part of
> the commercial-safe local core.

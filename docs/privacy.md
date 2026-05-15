# Privacy in VisionServeX

## Does VisionServeX provide end-to-end encryption?

**No — and this is by design.**

VisionServeX is an inference server. It must decode your image into pixel
tensors and feed them to the model to produce a result. There is no
cryptographic scheme that allows a server to run neural network inference
on ciphertext without seeing the plaintext data.

What we _do_ provide:

| Protection | Default | Notes |
|-----------|---------|-------|
| Local-only binding | ✅ `127.0.0.1` | Data never leaves your machine |
| Encrypted transport (TLS) | ⚪ Optional | Configure `TLS_CERT_FILE` / `TLS_KEY_FILE` |
| No data retention | ✅ `metadata_only` | Images never written to disk |
| No prompt logging | ✅ Off | `VISIONSERVEX_PRIVACY__SAVE_PROMPTS=false` |
| Optional encryption-at-rest | ⚪ Optional | SQLite job metadata; Fernet/AES-128 |
| Log redaction | ✅ Always | Tokens, base64, HF keys scrubbed |
| Secure temp files | ✅ Auto-deleted | 0600 permissions, removed after inference |
| Auth for public mode | ✅ Required | API key mandatory when exposing publicly |

## Data flow

```
User → [HTTPS/HTTP] → VisionServeX server → decode image → model → result
                                              ↑
                              plaintext tensors here (unavoidable for inference)
```

No image bytes, base64 strings, or mask arrays are written to disk by
default. Job metadata stored in SQLite contains only:
- `request_id`
- `model_id`
- `status` / `timestamps`
- `latency_ms`
- `error code` (if any)

## Privacy configuration

```bash
# Default (already set)
VISIONSERVEX_PRIVACY__RETENTION_MODE=metadata_only
VISIONSERVEX_PRIVACY__SAVE_INPUTS=false
VISIONSERVEX_PRIVACY__SAVE_OUTPUTS=false
VISIONSERVEX_PRIVACY__SAVE_PROMPTS=false

# Optional: encrypt job store metadata at rest
VISIONSERVEX_PRIVACY__ENCRYPT_JOB_STORE=true
visionservex security keygen --out ~/.visionservex/key.bin
export VISIONSERVEX_PRIVACY__ENCRYPTION_KEY_FILE=~/.visionservex/key.bin
```

## Choosing a security mode

```bash
visionservex security mode local_private    # default, safest
visionservex security mode cloudflare_private  # for public APIs
visionservex security audit --json          # current posture
```

## Log redaction

All log output is filtered through a redacting handler that scrubs:
- `Authorization: Bearer ...`
- `CF-Access-Client-Secret: ...`
- `HF_TOKEN=...` / `hf_...` tokens
- Base64 image data
- API keys in any form

To test redaction: `visionservex security test-redaction`

## Recommendations by use case

| Use case | Recommended mode | Additional steps |
|---------|-----------------|-----------------|
| Local development | `local_private` | Nothing extra needed |
| LAN sharing | `lan_private` | Enable auth, use TLS |
| Public API | `cloudflare_private` | Cloudflare Access + service tokens |
| Multi-user SaaS | `production_multi_user` | Encrypted job store, audit logs |

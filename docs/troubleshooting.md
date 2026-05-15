# Troubleshooting

## `MODEL_NOT_FOUND`

The model id is not in the registry.

```bash
visionservex list-models
```

## `MODEL_MISSING`

Weights are not in the cache and auto-pull is disabled.

```bash
visionservex pull <model_id>
# or enable auto-pull on the server:
export VISIONSERVEX_MODELS__AUTO_PULL=true
```

## `MANUAL_DOWNLOAD_REQUIRED`

The model is flagged `manual` — VisionServeX refuses to invent a download
path. Follow the upstream URL printed in the error message, drop the
files into the cache directory, then run
`visionservex cache repair <model_id>`.

## `ENGINE_UNAVAILABLE`

The model's engine cannot initialize. The error message names the missing
package and prints the install command.

Common cases:

- `pip install 'visionservex[grounding]'` for Grounding DINO.
- `pip install 'visionservex[hf]'` for Hugging Face based models.
- Run `visionservex doctor` to see which optional packages are installed.

By default, stub engines refuse to silently fall back to mock output. If
you accept mock predictions for plumbing tests:

```bash
export VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK=true
```

## `BAD_IMAGE` / decompression-bomb error

The image is too large. Raise limits or resize:

```bash
export VISIONSERVEX_LIMITS__MAX_IMAGE_PIXELS=33177600
export VISIONSERVEX_LIMITS__MAX_IMAGE_DIM=8192
```

## `BUSY` (503)

The per-model queue is full. Either raise queue size or back off:

```bash
export VISIONSERVEX_RUNTIME__QUEUE_SIZE=128
export VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY=4
```

## CUDA initialization fails

`visionservex doctor` will tell you whether torch can see your GPU.

- Check `nvidia-smi`.
- Make sure the torch wheel matches your CUDA toolkit.
- Force CPU: `export VISIONSERVEX_RUNTIME__DEVICE_PREFERENCE=cpu`.

## Apple Silicon (MPS)

- Use a recent torch (>= 2.1).
- MPS still has feature gaps; some ops fall back to CPU.
- If results look slow or odd, set `--device cpu` explicitly.

## OOM

- Reduce `runtime.per_model_concurrency`.
- Resize input images before sending.
- Pick a smaller variant.

## Cloudflare Tunnel refuses to run

The CLI requires `--i-understand-this-is-public` AND
`VISIONSERVEX_AUTH__ENABLED=true`.

## Permission errors writing the cache

Override the cache directory:

```bash
export VISION_SERVEX_CACHE_DIR=/var/cache/visionservex
```

## Hugging Face download fails

- The repo may require login: set `HF_TOKEN` (never logged).
- The repo may be gated. Request access on the model page.

## Logs contain a secret

They should not. `RedactingFilter` rewrites Bearer tokens, `api_key=...`,
and Cloudflare service tokens. If you observe a leak, please file a
private security report per `SECURITY.md`.

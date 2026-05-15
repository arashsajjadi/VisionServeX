# Google Colab GPU Worker (optional)

VisionServeX supports running as a **temporary remote GPU worker** on Google Colab.
This is good for demos, benchmarks, and short-lived GPU access — **not** for production.

> ⚠️ **Session limits, not infrastructure.** Colab sessions can disconnect at any time.
> The GPU is not guaranteed. Cache is lost when the session ends unless you mount Google Drive.
> Never store API keys or sensitive data in notebook cells.

---

## When to use Colab worker mode

✅ Good for:
- Demos and tutorials
- Benchmarks and quick comparisons
- Experimenting with GPU models when you have no local GPU
- One-off tasks during a single working session

❌ Not for:
- Production traffic
- Long-running services
- Workloads needing reliability or persistence
- Storing private data

---

## Quickstart in a Colab notebook

```python
# Cell 1 — install
!pip install -U "visionservex[server,hf,rfdetr]"

# Cell 2 — check environment
!visionservex colab doctor

# Cell 3 — optional: persistent cache via Drive
from google.colab import drive
drive.mount('/content/drive')
%env VISIONSERVEX_CACHE_DIR=/content/drive/MyDrive/visionservex_cache

# Cell 4 — pull a small GPU-friendly suite
!visionservex pull-suite gpu-demo --yes

# Cell 5 — start the worker (local to the Colab VM)
!visionservex gateway start --profile colab-gpu-worker &

# Cell 6 — test
!curl -s http://127.0.0.1:8080/health
```

A copy-paste-ready notebook lives at `examples/colab/VisionServeX_Colab_GPU_Worker.ipynb`.

---

## The `colab-gpu-worker` profile

Designed for short-lived Colab GPU sessions. Conservative defaults:

| Setting | Value |
|---------|-------|
| Bind host | `127.0.0.1` (local to Colab VM) |
| Max loaded models | 1 |
| Per-model concurrency | 1 |
| Queue size | 4 |
| Max VRAM fraction | 0.85 |
| Min free VRAM | 1.5 GB |
| Desktop GPU reserve | off (Colab has no desktop GUI) |
| Auto-pull | off |
| Save inputs/outputs | off |
| Retention | `metadata_only` |

Apply with:
```bash
visionservex gateway start --profile colab-gpu-worker
```

---

## CLI commands

```bash
# Environment / GPU diagnostics
visionservex colab doctor          # full report (in-Colab, GPU, Drive, auth)
visionservex colab status          # single-line status
visionservex colab gpu-check       # GPU + safe VRAM budget

# Cache configuration (Drive optional)
visionservex colab mount-drive     # show how to mount Drive in a cell
visionservex colab cache-path      # show recommended cache path
visionservex colab setup-cache --drive   # configure Drive-backed cache

# Auth / token (for tunnel exposure)
visionservex colab token           # generate a fresh API key (shown once)

# Safe tunnel exposure (requires auth + acknowledgement)
visionservex colab tunnel-start --domain api.example.com --i-understand-this-is-public
visionservex colab tunnel-stop

# Cleanup
visionservex colab cleanup --yes

# Test reachability of a remote Colab worker from your laptop
visionservex colab test-remote https://your-tunnel.example.com --api-key $TOKEN
```

JSON output is available everywhere via `--json`.

---

## Persistent cache via Google Drive (optional)

Without Drive, model downloads are lost when the Colab session ends.

To persist:
```python
# In a Colab cell:
from google.colab import drive
drive.mount('/content/drive')
```

```bash
visionservex colab setup-cache --drive
# Then:
%env VISIONSERVEX_CACHE_DIR=/content/drive/MyDrive/visionservex_cache
```

Subsequent sessions can reuse the cache and skip download.

---

## Calling the Colab worker from your laptop

After exposing the worker through Cloudflare Tunnel:

```python
from visionservex import Client

client = Client(
    "https://your-tunnel.example.com",
    api_key="<token from `visionservex colab token`>",
)

result = client.detect("dfine-n", "image.jpg")
print(result.summary())

result = client.classify("swinv2-tiny", "image.jpg")
print(result.top_k[:5])

result = client.grounded_segment(
    "grounded-sam2", "image.jpg", prompt="car, person"
)
```

Reachability check from your laptop:
```bash
visionservex colab test-remote https://your-tunnel.example.com --api-key $TOKEN
```

This probes `/health` and `/models` and returns structured error codes:
- `ok` — worker is reachable and authenticated.
- `AUTH_REQUIRED` — pass `--api-key`.
- `UNREACHABLE` — URL/tunnel/session is dead.
- `ERROR` — other transport-level failure.

---

## Security rules for Colab tunnels

The CLI enforces three rules before allowing a tunnel start:

1. **Auth must be configured.** Run `visionservex colab token` first and set the env vars.
2. **Public exposure must be acknowledged.** Pass `--i-understand-this-is-public`.
3. **`cloudflared` must be installed.** Otherwise `CLOUDFLARED_MISSING` is returned.

The default Colab profile binds to `127.0.0.1` — without a tunnel, nothing leaves the Colab VM.

Cloudflare Access is strongly recommended for any tunnel deployment.
See [cloudflare_tunnel.md](cloudflare_tunnel.md) and [security.md](security.md).

---

## Structured error codes

| Code | Meaning |
|------|---------|
| `COLAB_NOT_DETECTED` | Not running in Colab |
| `COLAB_GPU_UNAVAILABLE` | Colab session has no GPU (change runtime type) |
| `DRIVE_NOT_MOUNTED` | `--drive` requested but Drive not mounted |
| `AUTH_REQUIRED` | Tunnel requires API key configured first |
| `EXPOSURE_NOT_ACKNOWLEDGED` | Public tunnel requires `--i-understand-this-is-public` |
| `CLOUDFLARED_MISSING` | `cloudflared` binary not installed |
| `UNREACHABLE` | Remote worker URL did not respond |
| `MISSING_DEPENDENCY` | Required library not installed |

---

## Privacy notes

- The Colab worker writes no inputs/outputs to disk by default.
- Retention is `metadata_only`.
- Log redaction is always on (API keys, HF tokens, base64).
- Cache directory is **only** populated with model weights, not user images.
- If you mount Drive, only the cache directory is written there — never request bodies.

---

## Known limitations

- Colab sessions can disconnect without warning. Resume work in a fresh session.
- The GPU you receive is allocated by Google and may be a T4, L4, A100 — VisionServeX adapts via the VRAM guard, but performance varies.
- Large models (sam2-hiera-large, oneformer-dinat-large) may not fit on smaller GPUs.
- Cold-start downloads are slow on the first session — Drive cache fixes this.
- Two parallel inferences are not allowed by default in the `colab-gpu-worker` profile. Override only if you understand the VRAM tradeoffs.

# Local Model Gateway

VisionServeX runs as a local HTTP API gateway for computer vision models.
You start it once and call any supported model through one clean API — no
manual backend juggling.

## Quick start

```bash
# Install
pip install 'visionservex[server,hf,rfdetr]'

# Download a beginner suite
visionservex pull-suite beginner --yes

# Start the gateway
visionservex gateway start

# Or with a profile
visionservex gateway start --profile laptop
```

Endpoints available at `http://127.0.0.1:8080`:

```bash
# Health
curl http://127.0.0.1:8080/health

# List models
curl http://127.0.0.1:8080/models | jq '.models[:3]'

# Gateway status (loaded models, device, queue)
curl http://127.0.0.1:8080/gateway/status | jq

# Detect
curl -F "image=@image.jpg" -F "model_id=dfine-n" http://127.0.0.1:8080/detect

# Classify
curl -F "image=@image.jpg" -F "model_id=swinv2-tiny" http://127.0.0.1:8080/classify
```

## Python client

```python
from visionservex import Client

client = Client("http://127.0.0.1:8080")

result = client.detect("dfine-n", "image.jpg")
print(result.device, result.latency_ms)

result = client.classify("swinv2-tiny", "image.jpg")
for label, score in result.results[:5]:
    print(label, score)

result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
print(f"{len(result.results)} segments")
```

## Configuration profiles

| Profile | Bind | Auto-pull | Best for |
|---------|------|-----------|----------|
| `laptop` | 127.0.0.1 | easy_only | Development, single user |
| `gpu-workstation` | 127.0.0.1 | registry_allowed | Multi-model GPU serving |
| `cpu-safe` | 127.0.0.1 | false | Offline / resource-limited |
| `public-tunnel-safe` | 127.0.0.1 | false | Cloudflare Tunnel (requires auth) |

```bash
visionservex gateway start --profile gpu-workstation
visionservex gateway profile laptop          # show env vars
visionservex gateway profile laptop --json   # JSON output
```

## Model suites

```bash
visionservex pull-suite beginner --yes       # dfine-n, swinv2-tiny, sam2-hiera-tiny
visionservex pull-suite gpu-demo --yes       # + rfdetr, grounding-dino, grounded-sam2
visionservex pull-suite server-demo --yes    # dfine-n, sam2-hiera-tiny, grounded-sam2
```

## Scheduler policies

Some models benefit from exclusive GPU access (no concurrent requests):

```bash
visionservex scheduler profile --json        # show all policies
visionservex scheduler recommend --model dfine-n --json
```

| Model | Policy | Max concurrency |
|-------|--------|----------------|
| dfine-n | queue_recommended | 1 |
| swinv2-tiny | acceptable_parallelism | 2 |
| grounded-sam2 | gpu_exclusive | 1 |

## Job tracking

When `auto_pull` is on and a model is missing:

```bash
POST /predict?wait_for_download=false
→ {"status": "downloading", "job_id": "...", "progress_url": "/jobs/..."}

GET /jobs/{id}                     # snapshot
GET /jobs/{id}/events?sse=true     # Server-Sent Events stream
```

## Security

By default, the gateway binds to `127.0.0.1` (loopback only) and requires
no authentication. For Cloudflare Tunnel use, enable auth:

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")
visionservex gateway start --profile public-tunnel-safe
```

See [security.md](security.md) and [cloudflare_tunnel.md](cloudflare_tunnel.md).

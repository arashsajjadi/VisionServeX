# Concurrency and server load control

VisionServeX uses a per-model async scheduler with explicit backpressure. The
goal is to serve concurrent requests efficiently while preventing OOM,
throughput collapse, and request starvation.

## How it works

1. A request arrives at a prediction endpoint.
2. The scheduler checks whether the per-model and global queues have capacity.
3. If capacity exists, the request is enqueued; a semaphore slot is acquired
   and inference runs in a thread pool (non-blocking event loop).
4. If the queue is full, the server immediately responds `503 SERVER_BUSY`
   with a `Retry-After` header.
5. On completion, the semaphore slot is released; the next queued request runs.

Models are loaded once and reused across concurrent requests. A per-model
load lock prevents duplicate downloads or loads when two requests race for
the same model.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY` | 2 | Max concurrent inflight requests per model |
| `VISIONSERVEX_RUNTIME__MAX_GLOBAL_CONCURRENCY` | 8 | Cap on total inflight requests across all models |
| `VISIONSERVEX_RUNTIME__QUEUE_SIZE` | 64 | Maximum queued (waiting) requests per model |
| `VISIONSERVEX_RUNTIME__REQUEST_TIMEOUT_S` | 60 | Per-request inference timeout |
| `VISIONSERVEX_RUNTIME__SERVER_BUSY_RETRY_AFTER_S` | 2 | `Retry-After` seconds on `503` responses |
| `VISIONSERVEX_RUNTIME__BUSY_STATUS_CODE` | 503 | HTTP status code for backpressure rejection |

## Profiles

**Laptop (CPU only)**

```bash
VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY=1
VISIONSERVEX_RUNTIME__MAX_GLOBAL_CONCURRENCY=2
VISIONSERVEX_RUNTIME__QUEUE_SIZE=8
```

**GPU workstation**

```bash
VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY=4
VISIONSERVEX_RUNTIME__MAX_GLOBAL_CONCURRENCY=16
VISIONSERVEX_RUNTIME__QUEUE_SIZE=64
```

**Public tunnel (conservative)**

```bash
VISIONSERVEX_RUNTIME__PER_MODEL_CONCURRENCY=1
VISIONSERVEX_RUNTIME__MAX_GLOBAL_CONCURRENCY=4
VISIONSERVEX_RUNTIME__QUEUE_SIZE=16
VISIONSERVEX_RUNTIME__REQUEST_TIMEOUT_S=30
```

## Server busy response

When the queue is full:

```json
{
  "request_id": "...",
  "error": {
    "code": "BUSY",
    "message": "The server queue for model X is full.",
    "hint": "Server is at capacity. Retry in ~2 seconds or reduce concurrency.",
    "details": {"retry_after_seconds": 2}
  }
}
```

HTTP headers:

```
HTTP/1.1 503 Service Unavailable
Retry-After: 2
```

Clients should respect `Retry-After` and implement exponential backoff.

## Metrics

`GET /metrics` returns:

```json
{
  "scheduler": {
    "total_inflight": 3,
    "models": {
      "rfdetr-nano": {"inflight": 2, "queued": 1}
    }
  },
  "counters": {
    "requests_total": 120,
    "requests_rejected_backpressure": 4,
    "requests_timed_out": 0
  },
  "observations": {
    "latency_ms": {
      "n": 120, "p50": 72.4, "p90": 143.8, "p99": 388.2
    }
  },
  "models_loaded": [...]
}
```

## Preventing duplicate model loads

The `ModelCache` holds a per-model `threading.RLock`. If two requests race
for an unloaded model, only one triggers the download/load; the other waits.
This prevents:

- duplicate weight downloads consuming bandwidth.
- duplicate GPU allocations causing OOM.
- duplicate model objects wasting VRAM.

## Auto-pull during inference

When `models.auto_pull=true` and a model is missing, the server either:

- Blocks until the download completes (`wait_for_download=true`, default).
- Returns a job id immediately (`?wait_for_download=false`), which the
  client can poll via `GET /jobs/{id}`.

During a download, the model's load slot is held so subsequent requests for
the same model wait for the download rather than starting competing downloads.

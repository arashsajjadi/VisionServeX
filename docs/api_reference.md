# HTTP API reference

Base URL: `http://127.0.0.1:8080` by default. OpenAPI at `/openapi.json`,
Swagger at `/docs`, ReDoc at `/redoc`.

All inference endpoints require authentication when
`VISIONSERVEX_AUTH__ENABLED=true`. Health, devices, and metadata do not.

## Stable response envelope (prediction)

```json
{
  "request_id": "...",
  "status": "completed",
  "model_id": "mock-detect",
  "task": "detect",
  "backend": "mock",
  "device": "cpu",
  "precision": "fp32",
  "latency_ms": 3.1,
  "model_loaded_from": null,
  "cache_path": null,
  "fallback_reason": null,
  "results": [...],
  "warnings": [],
  "metadata": {}
}
```

## Stable progress envelope (download / job)

```json
{
  "request_id": "...",
  "job_id": "...",
  "kind": "pull",
  "model_id": "...",
  "status": "downloading",
  "message": "Downloading model weights...",
  "progress": {
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "percent": 0.0,
    "speed_bytes_per_sec": 0
  },
  "result": null,
  "error": null,
  "created_at": 0,
  "updated_at": 0,
  "cancelled": false
}
```

## Stable error envelope

```json
{
  "request_id": "...",
  "error": {
    "code": "MODEL_MISSING",
    "message": "Model weights for 'rfdetr-small' are not available locally.",
    "hint": "Run: visionservex pull rfdetr-small",
    "details": {}
  }
}
```

Documented codes:

| Code                       | HTTP | Meaning                                            |
| -------------------------- | ---- | -------------------------------------------------- |
| `MODEL_NOT_FOUND`          | 404  | The `model_id` is not in the registry.             |
| `TASK_MISMATCH`            | 422  | Model registered for a different task.             |
| `MODEL_MISSING`            | 422  | Weights not cached and auto-pull disabled.         |
| `DOWNLOAD_FAILED`          | 422  | A pull failed for any reason.                      |
| `MANUAL_DOWNLOAD_REQUIRED` | 422  | Model is `manual` — follow upstream instructions.  |
| `BAD_MIME_TYPE`            | 422  | Content-Type not in the allowlist.                 |
| `BAD_IMAGE`                | 422  | Decode/dim/area validation failed.                 |
| `REQUEST_TOO_LARGE`        | 413  | Body exceeds `limits.max_upload_bytes`.            |
| `BAD_BASE64`               | 422  | `image_b64` is not valid base64.                   |
| `BAD_URL`                  | 422  | `image_url` failed SSRF / scheme / host checks.    |
| `URL_FETCH_FAILED`         | 422  | Remote image could not be fetched.                 |
| `UNAUTHENTICATED`          | 401  | Missing or invalid credentials.                    |
| `FORBIDDEN`                | 403  | Operation disabled by configuration.               |
| `RATE_LIMITED`             | 429  | Per-IP rate limit exceeded.                        |
| `BUSY`                     | 503  | Model queue is full; retry with backoff.           |
| `TIMEOUT`                  | 422  | Inference exceeded `runtime.request_timeout_s`.    |
| `ENGINE_UNAVAILABLE`       | 422  | Engine cannot load — usually missing optional dep. |
| `JOB_NOT_FOUND`            | 404  | Unknown or expired `job_id`.                       |
| `INTERNAL_ERROR`           | 500  | Unhandled error; see logs.                         |

## Endpoints

### Meta

```http
GET  /health
GET  /ready
GET  /version
GET  /devices
GET  /metrics
```

### Models

```http
GET    /models
GET    /models/{model_id}
POST   /models/{model_id}/pull?wait=true|false
POST   /models/{model_id}/load
POST   /models/{model_id}/unload
```

### Jobs

```http
GET    /jobs/{job_id}
GET    /jobs/{job_id}/events
DELETE /jobs/{job_id}
```

### Inference

```http
POST /predict                                 multipart: image, model_id, [prompts]
POST /detect                                  multipart: image, model_id
POST /segment                                 multipart: image, model_id
POST /pose                                    multipart: image, model_id
POST /classify                                multipart: image, model_id
POST /open-vocab/detect                       json: {model_id, image_b64|image_url, prompts}
POST /grounded-segment                        json: {model_id, image_b64|image_url, prompts}
POST /batch-predict                           multipart: images[], model_id
POST /predict/annotated                       multipart: image, model_id  → JPEG response
```

All prediction endpoints accept `?wait_for_download=true|false`. When
auto-pull is allowed and the model is missing, `false` returns a
`DownloadingResponse` (status `"downloading"` + `job_id` + `progress_url`).

## Auto-pull configuration

- `VISIONSERVEX_MODELS__AUTO_PULL=true|false` (default false)
- `VISIONSERVEX_MODELS__AUTO_PULL_POLICY=never|easy_only|registry_allowed|all_auto_downloadable`
- `VISIONSERVEX_MODELS__AUTO_PULL_MAX_SIZE_GB=5`
- `VISIONSERVEX_MODELS__AUTO_PULL_REQUIRE_AUTH=true`

When `server.public_mode=true` and `auto_pull_require_auth=true`, the
server refuses to auto-pull on a request that did not pass authentication.

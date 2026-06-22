# Worker Job Lifecycle (v3.22.0)

Long jobs (video inference, downloads) are tracked by `runtime.jobs.JobStore`.

## Job fields
`job_id`, `run_id`, `model_id`, `kind`, `status`, `message`, `created_at`,
`updated_at`, `progress`, `result`, `error`, `cancelled`, `cancel_requested`.
Statuses: `queued → checking_dependencies → downloading → … → running_inference →
completed | failed | cancelled`.

## Endpoints
| Endpoint | Method | Use |
| -------- | ------ | --- |
| `/jobs/{id}` | GET | snapshot |
| `/jobs/{id}/events` | GET | poll / SSE (`?sse=true`) |
| `/jobs/{id}/cancel` | POST | cancel (also `DELETE /jobs/{id}`) |
| `/jobs/{id}/artifacts` | GET | artifacts + result summary |
| `/jobs/{id}/cleanup` | POST | release temp files + GPU cache |

## Real cancellation (the fix)
Before v3.22 the `cancelled` flag was set but never checked. Now `JobStore.cancel`
sets a `threading.Event` (`cancel_event`); the video loop wraps it in a
`CancelToken` and checks it **between waves**. On cancel the worker:
1. stops submitting new waves, 2. drops the frame buffer, 3. releases tensors +
CUDA cache, 4. marks the job `cancelled` with a partial result.

**Proven live:** cancel during a video batch stopped after 8 frames / 2 waves
(vs ~916 for the full clip), VRAM released to 33 MB, status `cancelled`.

## Model switching
Cancel the old `/video/infer` job, then submit the new model. No browser refresh.
A model switch unloads the old model (frees VRAM) before loading the new one.

## Concurrency
One heavy GPU video job per worker (`_heavy_lock`). A second concurrent
`/video/infer` returns **409 WORKER_BUSY** with a retry hint.

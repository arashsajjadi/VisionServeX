# Performance

## Knobs

| Setting                                  | Purpose                                              |
| ---------------------------------------- | ---------------------------------------------------- |
| `runtime.device_preference`              | Force `cuda`, `mps`, or `cpu`.                       |
| `runtime.precision_preference`           | `fp32`, `fp16`, `bf16`, `int8` (engine-dependent).   |
| `runtime.max_loaded_models`              | LRU cap.                                             |
| `runtime.per_model_concurrency`          | Max concurrent inferences per model.                 |
| `runtime.queue_size`                     | Bounded queue; over the cap returns 503 BUSY.        |
| `runtime.request_timeout_s`              | Per-request budget.                                  |
| `runtime.model_idle_unload_s`            | Auto-unload after idleness; 0 disables.              |
| `limits.max_image_pixels`                | Largest accepted image.                              |

## Patterns

- **Warm starts**: call `POST /models/{id}/load` once at boot for the models
  you know you need. The first prediction after that is fast.
- **Reuse the same VisionModel**: in Python, do not re-instantiate per
  request. In server mode, the LRU cache does this for you.
- **Batch where the model supports it**: `m.batch_predict([...])` or
  `/batch-predict`.
- **Pick the right device**: GPU for transformers and SAM 2; CPU is fine
  for small inputs with light models.
- **Tune precision**: fp16 / bf16 on supported GPUs roughly halves memory
  and often improves throughput.

## Anti-patterns

- Spawning a fresh `VisionModel` for every request (cold start every time).
- Driving inference in unbounded background processes (uncontrolled GPU
  contention).
- Increasing `queue_size` to infinity instead of fixing slow inference
  (eventually OOMs).
- Reading large images without re-sizing first.

## Measuring

```bash
visionservex benchmark dfine-small image.jpg --n 50
```

In the server, `GET /metrics` returns latency percentiles, queue stats, and
loaded models.

## CPU vs GPU note

VisionServeX never silently switches to CPU. Either `device_preference="auto"`
picks a sensible device or you set one explicitly. `visionservex doctor`
reports what is available.

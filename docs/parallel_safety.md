# Parallel Inference Safety

VisionServeX uses model-aware concurrency policies to prevent uncontrolled
parallel GPU workloads that can exhaust VRAM and freeze the system.

## Why parallel inference is risky

Each inference request on a GPU model holds VRAM. If multiple requests run
concurrently and each loads a separate model copy, VRAM fills up rapidly.
On a 16 GB desktop GPU with three concurrent D-FINE requests, VRAM can reach
12–14 GB — close to the limit and risking a desktop freeze.

The parallel safety system prevents this by:
1. Enforcing concurrency policies per model family.
2. Queuing excess requests instead of spawning uncontrolled processes.
3. Applying a VRAM safety budget before each GPU load.
4. Running GPU tests serially by default.

---

## Model concurrency policies

| Policy | Meaning | Default models |
|--------|---------|---------------|
| `gpu_exclusive` | Only one concurrent GPU request | rfdetr-nano, sam2-hiera-tiny, grounded-sam2, oneformer |
| `queue_recommended` | Queue instead of parallel; slowdown observed at concurrency ≥ 2 | dfine-n (215% slowdown), grounding-dino-tiny |
| `acceptable_parallelism` | Limited concurrency acceptable | swinv2-tiny (max 2) |
| `cpu_parallel` | CPU parallel allowed | Mock models |

```bash
# Show all model scheduler policies
visionservex scheduler profile
visionservex scheduler profile --json

# Show policy for a specific model
visionservex scheduler profile --model dfine-n

# Get a recommendation
visionservex scheduler recommend --model dfine-n --device cuda

# Override policy (runtime only — not persisted)
visionservex scheduler set-policy dfine-n --policy gpu_exclusive --max-concurrency 1
```

---

## Parallel test benchmarking

```bash
# Test concurrent inference — stops on VRAM risk
visionservex parallel-test dfine-n examples/images/street.jpg \
  --device cuda \
  --concurrency 2 \
  --runs 3 \
  --stop-on-vram-risk

# Test two models concurrently
visionservex parallel-test-pair dfine-n swinv2-tiny examples/images/street.jpg \
  --device cuda
```

Status classification from parallel tests:

| Status | Slowdown threshold |
|--------|--------------------|
| `excellent_parallelism` | ≤ 10% |
| `acceptable_parallelism` | ≤ 25% |
| `queue_recommended` | > 25% |

---

## GPU test serial policy

GPU tests run serially by default. VisionServeX does not use `pytest-xdist`
for GPU tests. Each test:
1. Checks free VRAM before loading a model.
2. Clears the torch cache after each model.
3. Calls `del model; gc.collect(); torch.cuda.empty_cache()` between tests.

To enable GPU tests:
```bash
VISION_SERVEX_RUN_GPU_TESTS=1 pytest -m gpu -q --maxfail=1
```

Never run GPU tests with `pytest -n auto` — this spawns multiple workers,
each reserving VRAM, which can exhaust the GPU.

---

## Server queuing

When a request arrives and the GPU is busy, the server:
1. Checks the model's scheduler policy.
2. If `gpu_exclusive` or `queue_recommended`: queues the request.
3. If the queue is full: returns `SERVER_BUSY` with `Retry-After`.
4. If VRAM budget is exceeded: returns `GPU_MEMORY_GUARD`.

Clients should handle both codes:
```python
from visionservex import Client
from visionservex.exceptions import ServerBusy, GpuMemoryGuard

client = Client("http://127.0.0.1:8080")
try:
    result = client.detect("dfine-n", "image.jpg")
except ServerBusy as exc:
    print(f"Retry after {exc.retry_after}s")
except GpuMemoryGuard as exc:
    print(f"VRAM exhausted: {exc.details}")
```

---

## Benchmark results (RTX 5080, v1.0.0)

| Model | P50 CUDA | P50 CPU | Policy | Max concurrency |
|-------|----------|---------|--------|----------------|
| dfine-n | 9.2 ms | 48.7 ms | queue_recommended | 1 |
| swinv2-tiny | 16.2 ms | 28.3 ms | acceptable_parallelism | 2 |
| sam2-hiera-tiny | 31.5 ms | 504 ms | gpu_exclusive | 1 |
| grounding-dino-tiny | 90.5 ms | 2553 ms | queue_recommended | 1 |
| grounded-sam2 | 206.8 ms | 4697 ms | gpu_exclusive | 1 |

GPU is strongly recommended for sam2, grounding-dino, and grounded-sam2.
CPU is viable for dfine-n and swinv2-tiny for moderate throughput.

# Benchmarks

VisionServeX provides a CLI benchmark matrix to measure latency and
throughput across models and devices.

## Single-model benchmark

```bash
visionservex benchmark mock-detect examples/images/street.jpg --runs 10 --warmup 2
visionservex benchmark rfdetr-nano examples/images/street.jpg --runs 10 --device cuda
```

Output:
- Cold load time (including weight loading).
- Warm p50 / p90 / p99 latency.
- Estimated throughput (req/s) at warm p50.

## Benchmark matrix

```bash
# CPU-only, all wired models
visionservex benchmark-matrix \
    --models rfdetr-nano,dfine-n,swinv2-tiny,sam2-hiera-tiny \
    --devices cpu \
    --runs 5 --warmup 2

# GPU benchmark (requires healthy CUDA or MPS)
visionservex benchmark-matrix \
    --models rfdetr-nano,dfine-n,swinv2-tiny \
    --devices cuda,cpu \
    --runs 10 --warmup 3 \
    --out reports/benchmark.json
```

Output columns: Model, Device, Precision, P50 ms, P95 ms, Req/s, Fallback.

## Parallelism test

```bash
visionservex parallel-test rfdetr-nano examples/images/street.jpg \
    --concurrency 2 --runs 5 --device cpu
```

Output:
- Sequential p50 ms.
- Concurrent wall-time p50 ms.
- Slowdown %.
- Status: `excellent_parallelism` (≤10%), `acceptable_parallelism` (≤25%),
  `scheduler_needs_queueing` (>25%), `protected_throughput` (server busy).

## Server load test

```bash
# Start server first
visionservex serve &

# Benchmark at multiple concurrency levels
visionservex benchmark benchmark-server \
    --url http://127.0.0.1:8080 \
    --model rfdetr-nano \
    --concurrency 1,2,4 \
    --runs 10
```

## CPU baseline (verified 2026-05-15)

| Model | Device | P50 ms | Notes |
|-------|--------|--------|-------|
| `mock-detect` | cpu | 0.14 ms | built-in deterministic |
| `rfdetr-nano` | cpu | ~55 ms | warm; rfdetr package |
| `dfine-n` | cpu | ~45 ms | warm; HF Transformers |
| `swinv2-tiny` | cpu | ~28 ms | warm; HF Transformers |
| `sam2-hiera-tiny` | cpu | ~505 ms | warm; SAM2 HF |
| `grounding-dino-tiny` | cpu | ~2300 ms | warm; text tokenization overhead |

> **GPU and MPS are preferred automatically when healthy.** CPU figures are
> provided as a baseline. Actual GPU performance depends on hardware, VRAM,
> and driver version. Run `visionservex devices --benchmark` to measure your
> specific setup.

## Benchmark artifact hygiene

Generated benchmark reports go to `reports/` which is excluded from Git.
Do not commit `.json` report files unless they are intentionally curated.

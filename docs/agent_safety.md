# Agent Safety — VisionServeX

## Incident context

During iterative development, an agent launched multiple concurrent pytest processes in the background without resource checks. The machine reached:

- RAM: ~61/61 GB (100% usage)
- VRAM: ~15.9/15.9 GB on RTX 5080 (100% usage)
- SSD I/O: saturated
- GUI/mouse: unresponsive (hard freeze)

**Root cause:** Agents running `pytest` via background subprocesses without resource guards, marker filtering, or concurrency control.

---

## Resource guard system

All resource checks are implemented in `src/visionservex/runtime/resource_guard.py`.

### Default safety budgets

| Resource | Threshold | Env override |
|----------|-----------|--------------|
| Free RAM | ≥ 8 GB | `VISIONSERVEX_MIN_FREE_RAM_GB` |
| RAM usage | ≤ 80% | (fixed) |
| Free VRAM | ≥ 2 GB | `VISIONSERVEX_MIN_FREE_VRAM_GB` |
| Free disk | ≥ 10 GB | `VISIONSERVEX_MIN_FREE_DISK_GB` |
| Concurrent pytest | 0 others | `VISIONSERVEX_ALLOW_CONCURRENT_PYTEST=1` |
| Test workers | 1 (real_model/GPU) | `VISIONSERVEX_MAX_TEST_WORKERS` |

### Heavy test opt-in flags

| Flag | Enables |
|------|---------|
| `VISIONSERVEX_RUN_REAL_MODEL_TESTS=1` | `real_model`, `slow` markers |
| `VISIONSERVEX_RUN_GPU_TESTS=1` | `gpu` marker |
| `VISIONSERVEX_RUN_DOWNLOAD_TESTS=1` | `download` marker |
| `VISIONSERVEX_RUN_SIDECAR_TESTS=1` | `sidecar` marker |
| `VISIONSERVEX_RUN_BENCHMARK_TESTS=1` | `benchmark` marker |
| `VISIONSERVEX_RUN_DISK_HEAVY_TESTS=1` | `disk_heavy`, `memory` markers |

---

## Pytest lockfile

A lockfile at `/tmp/visionservex_pytest.lock` (configurable via `VISIONSERVEX_PYTEST_LOCK`)
prevents concurrent pytest runs. The lock contains PID, command, and start time.

- Created in `pytest_sessionstart` (conftest.py hook)
- Released in `pytest_sessionfinish`
- Stale lock (PID no longer exists) is detected and auto-cleaned

---

## Marker filtering

The quick test command filters out all heavy markers:

```
-m "not slow and not real_model and not gpu and not network
    and not sidecar and not release and not benchmark
    and not memory and not disk_heavy and not download"
```

A normal `pytest` run (no `-m`) still skips heavy markers via `pytest_collection_modifyitems`
in conftest.py — it inspects env vars at collection time.

---

## GPU VRAM safety

Before loading any real model:
1. Check free VRAM ≥ required + 2 GB reserve (for desktop/display)
2. Refuse if insufficient (raise `ResourceGuardError`)

After every model test:
```python
model.unload()
del model, processor, outputs, tensors
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
torch.cuda.reset_peak_memory_stats()
```

This is enforced by `cleanup_after_test()` in `resource_guard.py`, called from the
`_cleanup_heavy_test` autouse fixture in conftest.py.

---

## CI design

| Pipeline | Trigger | OS/Python | Timeout |
|----------|---------|-----------|---------|
| Fast CI (lint + quick tests) | push, PR | ubuntu/3.12 | 10 min |
| Full CI (matrix) | release tag, workflow_dispatch | 3 OS × 3 Python | 30 min |

Concurrency: old CI runs on the same branch auto-cancel.

---

## Agent rules summary

See `AGENT_RULES.md` in the repo root for the concise rule set.

Key prohibitions:
- Never run pytest in background
- Never run multiple pytest processes
- Never run real_model/gpu/download tests without explicit env var
- Stop if RAM > 80% or VRAM < 2 GB free
- Check `visionservex dev resources` before any heavy operation

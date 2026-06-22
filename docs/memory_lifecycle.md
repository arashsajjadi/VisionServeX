# Memory Lifecycle (v3.22.0)

VisionServeX measures and releases GPU memory at every boundary (telemetry is real:
`torch.cuda.mem_*` + NVML + `nvidia-smi`).

## Release policy
- **Per wave** — temporary tensors dropped; CUDA cache cleared every N waves.
- **On cancel** — stop submitting new waves, drop the buffer, `force_gc()` +
  `clear_torch_cuda_cache()`; report VRAM after cleanup.
- **On completion** — final GC + cache clear; model stays warm, batch tensors freed.
- **On model switch** — `VisionModel.unload()` drops references then
  `empty_cache + ipc_collect + reset_peak_stats`.

`runtime.gpu_lifecycle` provides `get_gpu_memory_state`, `clear_torch_cuda_cache`,
`cleanup_gpu_after_model`, `assert_memory_returned_to_baseline`.

## Measured (RTX 5080, dfine-n) — `scripts/bench/memory_lifecycle.py`
| Scenario | Peak VRAM | After cleanup |
| -------- | --------: | ------------: |
| normal completion (48 frames) | 553 MB | **24 MB** |
| cancel mid-run | ~33 MB | **33 MB** |
| model switch (n→s) | per model | freed between |
| long video (300 frames streamed) | bounded | released |

VRAM does not grow with video length (streaming + per-wave release). After
`unload()`, allocated VRAM returns to ≈baseline (<200 MB).

## Caps
Target 85% / hard 92% / emergency 95% VRAM. OOM → halve batch + lower ceiling +
continue (`oom_recovered`), never crash the worker.

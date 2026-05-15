# GPU and MPS validation

VisionServeX automatically selects the fastest healthy accelerator. This
page explains how to verify that GPU (CUDA) or Apple MPS inference works
correctly.

## Quick start

```bash
# Check what devices are available and healthy
visionservex devices
visionservex devices --json           # machine-readable
visionservex devices --benchmark --quick  # tiny matmul benchmark

# Full GPU health check with fix suggestions
visionservex gpu doctor

# Smoke-test real inference on GPU
visionservex gpu smoke-test --models rfdetr-nano,dfine-n,swinv2-tiny \
    --device cuda

# Apple MPS smoke test (macOS only)
visionservex mps --models swinv2-tiny,sam2-hiera-tiny
```

## Device auto-selection logic

| Platform | Priority |
|----------|---------|
| Linux/Windows | CUDA GPU (highest free VRAM, sanity pass) → ROCm → CPU |
| macOS | MPS (sanity pass) → CPU |

A **sanity check** runs a tiny `3×3 @ 3×3` matmul on each CUDA device. If
this fails (e.g. `libnvrtc-builtins.so` missing), that GPU is marked
`sanity_ok=False` and skipped. The next candidate is tried.

## What was tested

All wired models are **CPU-verified** (fp32). GPU inference on this
repository's host was blocked by a missing CUDA runtime component
(`libnvrtc-builtins.so.13.0`) in the test environment. The device selection
code, sanity-check logic, and dtype-casting code are all implemented and
tested; we just cannot run a live GPU model benchmark on this host.

To verify GPU inference on your system:

```bash
visionservex gpu doctor          # shows if CUDA runtime is healthy
visionservex gpu smoke-test --models rfdetr-nano,dfine-n,swinv2-tiny --device cuda
```

## Fixing CUDA runtime issues

If `visionservex gpu doctor` reports CUDA is broken:

1. **Driver/toolkit mismatch** — reinstall the PyTorch wheel matching your
   CUDA version: https://pytorch.org/get-started/locally/
2. **Missing `libnvrtc-builtins.so`** — install the CUDA toolkit:
   `sudo apt install cuda-toolkit-12-x` or the appropriate version.
3. **`LD_LIBRARY_PATH`** — ensure `/usr/local/cuda/lib64` is on the path.
4. **Docker** — use the official NVIDIA CUDA image which includes the full
   runtime:
   ```bash
   docker run --gpus all nvidia/cuda:12.4-runtime-ubuntu22.04 ...
   ```

## CPU performance expectations

| Model | Cold load | Warm p50 | Notes |
|-------|-----------|----------|-------|
| `rfdetr-nano` | ~3000 ms | ~55 ms | rfdetr pkg handles AMP |
| `dfine-n` | ~800 ms | ~45 ms | HF Transformers, fp32 |
| `swinv2-tiny` | ~600 ms | ~28 ms | HF, small model |
| `sam2-hiera-tiny` | ~500 ms | ~505 ms | SAM2 HF, center point |
| `grounding-dino-tiny` | ~2000 ms | ~2300 ms | text encoding overhead |

GPU and MPS will be significantly faster, especially for repeated inference.

## MPS on Apple Silicon

MPS is preferred over CPU on macOS. Some PyTorch ops fall back to CPU
inside MPS; latency may be slightly lower than pure GPU. Run:

```bash
visionservex mps --models swinv2-tiny,sam2-hiera-tiny
```

If MPS sanity fails, `visionservex devices` shows the error and VisionServeX
falls back to CPU automatically.

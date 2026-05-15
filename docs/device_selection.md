# Device selection

VisionServeX automatically selects the fastest *healthy* device. A device
is considered healthy only if a quick sanity check (allocate + multiply a
tiny tensor) succeeds.

## Auto-selection order

**Linux / Windows:**

1. NVIDIA CUDA — among all detected GPUs, pick the one with the highest
   free VRAM that passes the sanity check.
2. ROCm — if detected via `torch.version.hip`.
3. DirectML — if `torch_directml` is installed on Windows.
4. CPU — always available.

**macOS:**

1. Apple MPS — if `torch.backends.mps.is_available()` and sanity check passes.
2. CPU.

## CUDA sanity check

The sanity check allocates a 3×3 float32 tensor on the device and performs a
matrix multiply. This catches:

- Missing shared libraries (`libnvrtc-builtins.so`, `libcuda.so`).
- Driver/toolkit version mismatches.
- Permission errors.

If the sanity check fails, that GPU is marked `sanity_ok=False` and
**not used** for inference. The next candidate is tried. This prevents
a silent crash at inference time.

## Multi-GPU selection

When multiple CUDA GPUs are present, VisionServeX selects the one with
the most **free** VRAM (at load time). To pin a specific GPU:

```bash
export VISIONSERVEX_RUNTIME__DEVICE_PREFERENCE=cuda:1
```

## Verifying your device

```bash
visionservex devices
visionservex devices --benchmark --quick
visionservex doctor
```

Output from `devices` includes:

- `sanity_ok`: whether the sanity check passed.
- `sanity_error`: the error message if it failed.
- `free_vram_gb`: current free VRAM.

## Overriding device

Python:

```python
m = VisionModel("rfdetr-nano", device="cuda")     # specific device
m = VisionModel("rfdetr-nano", device="cuda:1")   # second GPU
m = VisionModel("rfdetr-nano", device="cpu")      # force CPU
```

CLI:

```bash
visionservex predict rfdetr-nano image.jpg --device cpu
```

Config:

```bash
export VISIONSERVEX_RUNTIME__DEVICE_PREFERENCE=cuda
export VISIONSERVEX_RUNTIME__REQUIRE_GPU=false     # allow CPU fallback
export VISIONSERVEX_RUNTIME__MIN_FREE_VRAM_GB=4    # skip GPUs with <4 GB free
export VISIONSERVEX_RUNTIME__GPU_SANITY_CHECK=true # default
```

## Broken CUDA / fallback

If CUDA probing fails (e.g. `libnvrtc-builtins.so.13.0` missing):

```
GPU detected but CUDA runtime is broken: <error>. 
Run `visionservex doctor` for fix suggestions. Using CPU fallback.
```

Fix options:

1. Install the matching CUDA toolkit: `sudo apt install cuda-toolkit-12-x`.
2. Check `LD_LIBRARY_PATH` includes the CUDA lib directory.
3. Reinstall the correct PyTorch wheel for your CUDA version.
4. Use the Docker CUDA image which includes the full runtime.

## MPS on Apple Silicon

MPS is preferred over CPU on macOS when available and healthy. Some model
operations fall back to CPU within PyTorch; behavior is generally correct
but latency may not match expectations. If issues arise, pin `--device cpu`.

## CPU inference

All wired models work on CPU. Expect significantly higher latency compared
to GPU:

- Small models (`rfdetr-nano`, `dfine-n`, `swinv2-tiny`): 100ms – 500ms warm.
- Medium models (`grounding-dino-tiny`, `sam-vit-base`): 1–5 seconds warm.
- Composed pipelines (`grounded-sam`, `grounded-sam2`): 5–15 seconds warm.

CPU is the verified test environment for this repository. GPU is preferred
automatically when healthy — run `visionservex doctor` to confirm.

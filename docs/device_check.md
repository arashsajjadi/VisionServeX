# Device check

VisionServeX always tells you what device it is using and why. Three
commands cover all device-related diagnostics:

```bash
visionservex doctor       # holistic system + recommendation
visionservex devices      # device-only table
visionservex devices --json
```

The system shows for each device:

- **name**: `cpu`, `cuda`, `mps`, `rocm`, `directml`.
- **available**: whether VisionServeX could pick it.
- **detail**: human-readable label (e.g. `NVIDIA GeForce RTX 5080 (cap 12.0)`).
- **VRAM**: total and free GB when probable.
- **capability**: GPU compute capability (CUDA).

## Device selection rules

- `--device auto` (the default): pick CUDA > MPS > ROCm > DirectML > CPU,
  but only from devices the model supports.
- `--device cuda` (or `cuda:1`): honor the user's explicit pick; if CUDA
  is unavailable, fall back to CPU.
- `--device cpu`: always allowed.

Precision auto-selection:

| Device  | Auto precision        |
| ------- | --------------------- |
| CPU     | `fp32`                |
| CUDA    | `fp16` when supported |
| MPS     | `fp16` when supported |

To override, pass `--precision fp32|fp16|bf16|int8`.

## What to do when no GPU is detected

`visionservex doctor` will say so explicitly and switch its recommendation
to a CPU-friendly model. Mock and Grounding DINO (tiny) can run on CPU at
small image sizes.

## Apple Silicon (MPS)

We honestly report MPS as available on macOS when torch is installed and
`torch.backends.mps.is_available()` is true. Some ops fall back to CPU in
PyTorch; if you see odd behavior, set `--device cpu` explicitly.

## ROCm / DirectML

We do not auto-claim ROCm or DirectML support. They are detected
heuristically (`torch.version.hip`, `torch_directml`) and reported, but
not selected by `auto`. If you have a working setup, pin the device
explicitly: `--device rocm` (or `directml`).

## CUDA but no torch

`doctor` says so. It detects an NVIDIA GPU via `nvidia-smi` and then
shows: "GPU present but torch is not installed". Install:

```bash
pip install 'visionservex[torch]'
# or follow https://pytorch.org/get-started/locally/ for the right wheel.
```

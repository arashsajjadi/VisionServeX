# GPU Safety and VRAM Guard

VisionServeX includes a desktop-GPU safety guard to prevent VRAM exhaustion,
GUI freezes, and uncontrolled parallel workloads.

## Why this matters

An RTX 5080 has 16.3 GB of VRAM. If VisionServeX consumed all of it during a
benchmark or parallel test, the desktop compositor (gnome-shell, KWin, Xwayland)
would freeze. The VRAM guard prevents this by enforcing a safety buffer.

---

## Default safety policy

| Parameter | Default | Config env |
|-----------|---------|-----------|
| Max VRAM fraction | 80% | `VISIONSERVEX_RUNTIME__MAX_VRAM_FRACTION` |
| Min free VRAM | 3.0 GB | `VISIONSERVEX_RUNTIME__MIN_FREE_VRAM_GB` |
| GUI reservation | 3.0 GB | `VISIONSERVEX_RUNTIME__RESERVE_GUI_VRAM` |
| Desktop GPU mode | true | `VISIONSERVEX_RUNTIME__DESKTOP_GPU` |
| Allow high VRAM | false | `VISIONSERVEX_RUNTIME__ALLOW_HIGH_VRAM` |
| OOM guard strict | true | `VISIONSERVEX_RUNTIME__OOM_GUARD_STRICT` |

On a 16 GB desktop GPU with 1.5 GB already used, the effective budget for
VisionServeX is approximately:
```
min(16*0.80 - 1.5, 16 - 1.5 - 3.0) = min(11.3, 11.5) = 11.3 GB
```

---

## GPU status commands

```bash
# Live VRAM state, safety budget, and policy
visionservex gpu guard-status

# JSON output for scripting
visionservex gpu guard-status --json

# List all GPU compute processes with safety classification
visionservex gpu processes

# Safely terminate VisionServeX/pytest/benchmark GPU processes
# (GUI/system processes are NEVER touched)
visionservex gpu cleanup
visionservex gpu cleanup --yes          # skip confirmation
visionservex gpu cleanup --dry-run     # preview only

# Emergency recovery advice (does not auto-reset the GPU)
visionservex gpu reset-advice
```

---

## Protected vs safe-to-terminate processes

The cleanup command classifies GPU processes:

| Classification | Examples | Action |
|---------------|---------|--------|
| **Protected** | gnome-shell, Xwayland, Firefox, Brave, VS Code, PyCharm | **Never terminated** |
| **Safe to terminate** | visionservex, pytest, benchmark, parallel-test | Terminated with SIGTERM on confirm |
| **Unknown** | Other Python processes | Listed but not auto-terminated |

Never pass `--include-python` unless you are certain which processes are safe.

---

## GPU smoke test safety flags

```bash
# Serial smoke test with VRAM safety
visionservex gpu smoke-test \
  --serial \
  --max-vram-fraction 0.80 \
  --min-free-vram-gb 3.0 \
  --stop-on-vram-risk

# Add --allow-high-vram only on headless servers with no GUI
visionservex gpu smoke-test \
  --allow-high-vram \
  --max-vram-fraction 0.95
```

---

## Parallel test VRAM guard

```bash
visionservex parallel-test dfine-n examples/images/street.jpg \
  --device cuda \
  --concurrency 2 \
  --runs 3 \
  --stop-on-vram-risk \
  --max-vram-fraction 0.80 \
  --min-free-vram-gb 3.0
```

If VRAM is at risk, the command returns `GPU_MEMORY_GUARD` and stops.

---

## Server / gateway admission control

The local server queues or rejects inference requests when:
- The current VRAM usage exceeds the configured fraction cap.
- Free VRAM would drop below the minimum after loading the requested model.

Error codes:
- `GPU_MEMORY_GUARD` — VRAM budget exceeded; reduce concurrency or use CPU.
- `SERVER_BUSY` — queue full; retry after the `Retry-After` header value.

---

## Desktop GPU warning

When a display server is detected on the same GPU (common on Linux with
Wayland/X11), VisionServeX adds an extra GUI reservation:

```
Desktop GPU detected. Reserving 3.0GB VRAM for GUI/system responsiveness.
```

To run on a headless server without this reserve:
```bash
VISIONSERVEX_RUNTIME__DESKTOP_GPU=false \
VISIONSERVEX_RUNTIME__RESERVE_GUI_VRAM=false \
visionservex serve
```

---

## Emergency VRAM recovery

If VRAM is unexpectedly occupied after processes exit:

```bash
# 1. Inspect
nvidia-smi
nvidia-smi --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader

# 2. Terminate test processes
visionservex gpu cleanup
pkill -TERM -f 'pytest|visionservex|benchmark|parallel-test'

# 3. Wait for driver to release memory, then re-check
nvidia-smi

# 4. If still occupied, find which process holds it
fuser /dev/nvidia0 2>/dev/null
```

VisionServeX **never** runs `nvidia-smi --gpu-reset` automatically.
Manual GPU reset requires exclusive access and stops all GPU workloads.

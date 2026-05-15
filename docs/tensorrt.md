# TensorRT deployment

TensorRT optimisation is **not yet implemented** inside VisionServeX.
This page documents the planned path and available workarounds.

## Current status

| Step | Status |
|------|--------|
| ONNX export (SwinV2) | ✅ available (`visionservex export`) |
| ONNX export (other models) | ⚠ partial / not yet |
| TensorRT engine build | ❌ not implemented |
| TensorRT inference | ❌ not implemented |

## Workaround: manual TensorRT path

1. Export a model to ONNX (see [docs/export.md](export.md)).
2. Build a TensorRT engine using `trtexec` or the TensorRT Python API.
3. Load the engine and run inference with the `tensorrt` Python bindings.

```bash
# Build engine (FP16) from ONNX
trtexec \
  --onnx=exports/swinv2-tiny.onnx \
  --saveEngine=exports/swinv2-tiny.engine \
  --fp16 \
  --optShapes=pixel_values:1x3x256x256 \
  --minShapes=pixel_values:1x3x256x256 \
  --maxShapes=pixel_values:8x3x256x256
```

```python
# Inference with TensorRT (example, outside VisionServeX)
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# ... (standard TensorRT inference code)
```

## Prerequisites

- NVIDIA GPU with Tensor Cores (Volta+).
- CUDA toolkit and cuDNN matching your TensorRT version.
- `pip install tensorrt pycuda`

## Roadmap

VisionServeX plans to add:

1. `visionservex export MODEL_ID --format tensorrt`
2. `TensorRTEngine` class that wraps a `.engine` file.
3. Transparent device and precision selection for TRT.

Contributions are welcome. The `engines/onnx.py` stub shows where the
ONNX Runtime engine lives; a `TensorRTEngine` would follow the same pattern.

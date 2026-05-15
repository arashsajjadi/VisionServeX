# Model export

VisionServeX supports exporting wired models to ONNX format for downstream
deployment. Export quality varies by model architecture.

## Supported ONNX exports

| Model | ONNX | Opset | Dynamic batch | Notes |
|-------|------|-------|--------------|-------|
| `swinv2-tiny` | ✅ supported | 17 | yes | 2.7 MB, checker: pass |
| `swinv2-small` | ✅ supported | 17 | yes | similar size |
| `swinv2-base` | ✅ supported | 17 | yes | |
| `swinv2-large` | ✅ supported | 17 | yes | |
| `rfdetr-nano/small/base` | ⚠ via rfdetr pkg | — | see rfdetr docs | |
| `dfine-*` | ❌ not yet | — | complex arch | ONNX translation fails |
| `grounding-dino-*` | ❌ not yet | — | complex arch | |
| `sam*/sam2*` | ❌ not yet | — | heavy arch | |
| `oneformer-*` | ❌ not yet | — | complex arch | |

## CLI

```bash
# Export SwinV2 to ONNX
visionservex export swinv2-tiny --format onnx --out exports/swinv2-tiny.onnx

# Dry run (reports what would be exported without writing files)
visionservex export swinv2-tiny --format onnx --out /tmp/x.onnx --dry-run

# Check export capability
visionservex info swinv2-tiny --json | jq .export_support
```

## Python API

```python
from visionservex import VisionModel

m = VisionModel("swinv2-tiny", device="cpu")
m.warmup()
path = m.export(format="onnx", output_path="exports/swinv2-tiny.onnx")
print("Exported to", path)
```

## ONNX validation

After export, you can validate and benchmark with ONNX Runtime:

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("exports/swinv2-tiny.onnx")
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
out = sess.run(None, {"pixel_values": dummy})
print("Logits shape:", out[0].shape)
```

Install ONNX Runtime:

```bash
pip install 'visionservex[onnx]'
```

## RF-DETR export

The `rfdetr` package has its own export mechanism. Use the rfdetr package
directly:

```python
from rfdetr import RFDETRNano
model = RFDETRNano(device="cpu")
model.export(...)   # see rfdetr docs
```

## TensorRT

TensorRT export is **not implemented** in this version. It is a planned
roadmap item. For TensorRT deployment, the recommended path is:

1. Export to ONNX.
2. Use `trtexec` or the TensorRT Python API to convert the ONNX model.
3. Load the resulting `.engine` file with TensorRT directly.

Example (outside VisionServeX):

```bash
trtexec --onnx=swinv2-tiny.onnx --saveEngine=swinv2-tiny.engine \
        --fp16 --optShapes=pixel_values:1x3x256x256
```

See [docs/tensorrt.md](tensorrt.md) for details.

## Tracked files

Exported `.onnx` and `.engine` files are excluded from Git via `.gitignore`.
Never commit model exports to the repository.

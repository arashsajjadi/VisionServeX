# Installation

VisionServeX targets Python 3.10+. The basic install is light and adds the
CLI, registry, Python API, and `MockEngine`. Heavy backends are opt-in.

## Operating systems

We test against Linux, macOS, and Windows. CUDA is supported on Linux and
Windows. Apple Silicon GPUs use MPS through PyTorch. ROCm and DirectML are
acknowledged but not auto-detected unless you opt in.

## Basic install

```bash
pip install visionservex
visionservex doctor
```

`visionservex doctor` reports your devices, optional dependencies, and the
effective configuration.

## Server install

```bash
pip install 'visionservex[server]'
```

This adds FastAPI, uvicorn, python-multipart, and slowapi.

## Cloudflare Tunnel helpers

The Python extra is small; the heavy lifting is done by `cloudflared`.

```bash
pip install 'visionservex[cloudflare]'    # python helpers only
visionservex tunnel doctor                # tells you how to install cloudflared
```

**Linux** (Debian/Ubuntu):
```bash
curl -L https://pkg.cloudflare.com/cloudflared-stable-linux-amd64.deb -o /tmp/cf.deb
sudo dpkg -i /tmp/cf.deb
```

**macOS**:
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Windows PowerShell**:
```powershell
winget install --id Cloudflare.cloudflared
```

## Optional model backends

### PyTorch

For D-FINE, RF-DETR, Swin V2 (via Hugging Face), and most other vision models:

```bash
pip install 'visionservex[torch]'
```

### Hugging Face (Transformers + Hub)

For the real Grounding DINO backend and Hugging Face downloads:

```bash
pip install 'visionservex[hf]'
```

Pick the right torch wheel for your platform/CUDA per
<https://pytorch.org/get-started/locally/>. On Windows + CUDA you typically
need an extra index URL.

### ONNX Runtime

```bash
pip install 'visionservex[onnx]'
```

For GPU acceleration with ONNX, install `onnxruntime-gpu` separately.

### TensorRT

Install TensorRT per NVIDIA's docs; then add:

```bash
pip install 'visionservex[tensorrt]'
```

### OpenMMLab toolchain (RTMPose, RTMDet-R, Co-DINO, Swin V2 via MMPreTrain, InternImage)

```bash
pip install -U openmim
mim install mmengine mmcv mmdet mmpose mmpretrain mmrotate
```

Refer to the upstream MMxxx docs for version compatibility. Some custom CUDA
ops (e.g. InternImage) require building from source on the target platform.

### SAM 2 / SAM 2.1

```bash
pip install 'visionservex[sam2]'
pip install "git+https://github.com/facebookresearch/sam2"
```

Download SAM 2 checkpoints per upstream instructions and point VisionServeX
at the cache via `VISION_SERVEX_CACHE_DIR`.

### Grounding DINO / Grounded SAM 2

```bash
pip install 'visionservex[grounding]'
```

Grounding DINO weights come from Hugging Face or the upstream repo. Grounding
DINO 1.5 is API-gated by IDEA-Research; VisionServeX marks it `external` and
disables auto-loading.

## Verifying

```bash
visionservex doctor
visionservex list-models
visionservex predict mock-detect path/to/any/image.jpg --save out.jpg --json
```

The mock backend works without any heavy dependency installed.

## Development install

```bash
git clone <your-fork>
cd VisionServeX
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,server]"
pre-commit install
pytest
```

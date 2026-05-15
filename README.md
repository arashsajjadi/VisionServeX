<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

<h1 align="center">VisionServeX</h1>

<p align="center">
  <strong>Secure, beginner-friendly Python API serving for permissive computer vision models</strong><br>
  Local inference &middot; Cloudflare Tunnel &middot; Stable JSON API &middot; LLM-agent-friendly
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml">
    <img src="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/code%20style-ruff-orange.svg" alt="ruff">
  <img src="https://img.shields.io/badge/version-0.4.0-informational.svg" alt="v0.4.0">
</p>

> **Note on the CI badge:** it turns green after the workflow runs successfully on GitHub for the first time.
> Until then the badge shows "no status" — this is expected for a new repository.

---

VisionServeX is a permissive-license-aware Python framework for running modern
computer vision models locally, exposing them through a clean HTTP API, and
optionally sharing them securely over Cloudflare Tunnel.

- **No CUDA expertise required.** `visionservex doctor` tells you what your machine can run. GPU is preferred automatically when available and healthy; broken CUDA runtimes fall back to CPU with a clear warning.
- **One download command.** `visionservex pull rfdetr-nano` — weights are cached and verified.
- **Stable contracts.** Every prediction returns the same JSON envelope, whether from CLI, Python, or curl.
- **Honest.** Registry entries say `wired`, `partial`, or `stub`. Stubs never silently fake results.
- **Secure defaults.** Binds to `127.0.0.1`, requires auth for public mode, SSRF and bomb guards on.

---

## Quickstart (works on CPU, ~5 minutes)

```bash
pip install 'visionservex[server,hf,rfdetr]'

visionservex getting-started          # personalized guide for your machine

# RF-DETR detection — real, fast
visionservex pull rfdetr-nano
visionservex predict rfdetr-nano examples/images/street.jpg --save outputs/out.jpg

# Grounding DINO — text-prompted detection
visionservex pull grounding-dino-tiny
visionservex predict grounding-dino-tiny examples/images/street.jpg \
    --prompt "car,person" --save outputs/gd.jpg

# D-FINE — detection via HF Transformers
visionservex pull dfine-s
visionservex predict dfine-s examples/images/street.jpg --save outputs/dfine.jpg

# Start the API
visionservex serve
curl -F "image=@examples/images/street.jpg" \
     -F "model_id=rfdetr-nano" \
     http://127.0.0.1:8080/detect | jq
```

**Recommendation engine:**

```bash
visionservex recommend --task detect --simple
```

---

## What works today

| Family | Model IDs | Task | Status | Install |
|--------|-----------|------|--------|---------|
| **Mock (built-in)** | `mock-*` | All tasks | stable | base |
| **RF-DETR** | `rfdetr-nano/small/base/medium/large` | detect | beta | `[rfdetr]` |
| **RF-DETR-Seg** | `rfdetr-seg-nano/small/medium` | segment | beta | `[rfdetr]` |
| **D-FINE** | `dfine-n/s/m/l/x` | detect | beta | `[hf]` |
| **Grounding DINO** | `grounding-dino-tiny/swin-t/swin-b` | open-vocab detect | beta | `[hf]` |
| **SwinV2** | `swinv2-tiny/small/base/large` | classify | beta | `[hf]` |
| **SAM v1** | `sam-vit-base/large/huge` | foundation segment | beta | `[hf]` |
| **SAM 2** | `sam2-hiera-tiny/small/base-plus/large` | foundation segment | beta | `[hf]` |
| **Grounded SAM** | `grounded-sam` | grounded segment | beta | `[hf]` |
| **OneFormer** | `oneformer-swin-large/dinat-large/convnext-large` | segment (semantic/instance/panoptic) | beta | `[hf]` |

### Not yet wired

| Family | Why | Alternative |
|--------|-----|-------------|
| RTMPose | Requires OpenMMLab toolchain | `mock-pose` for schema |
| RTMDet-R/R2 (OBB) | Requires OpenMMLab + mmrotate | `mock-obb` for schema |
| Co-DINO-Inst | Requires heavy OpenMMLab | `rfdetr-seg-*` for instance seg |
| InternImage | Custom CUDA ops, build required | `swinv2-*` for classification |
| SEEM | Expert manual install | `oneformer-swin-large` |
| Grounded-SAM-2 | Needs upstream `sam2` package | `grounded-sam` (works today) |
| ONNX export | CLI exists; engine-quality varies | Use HF model repos for ONNX |
| TensorRT | Future roadmap | — |

> We make no benchmark claims. Pick by task, license, and hardware. See [docs/model_zoo.md](docs/model_zoo.md).

---

## Which model to start with?

| I want | Start with | CPU? |
|--------|-----------|------|
| Fast detection | `rfdetr-nano` | yes |
| More accurate detection | `dfine-s` | yes (slower) |
| Text-prompted detection | `grounding-dino-tiny` | yes (slower) |
| Instance segmentation | `rfdetr-seg-nano` | yes |
| SAM-style masking | `sam-vit-base` or `sam2-hiera-tiny` | yes (slow) |
| Text + mask together | `grounded-sam` | yes (slow) |
| Image classification | `swinv2-tiny` | yes |
| Semantic scene parsing | `oneformer-swin-large` | yes (slow) |
| Just testing/CI | `mock-detect` | yes (instant) |
| **I have no GPU** | Any `*-nano` or `*-tiny` model | yes |
| **I have NVIDIA GPU** | Run `visionservex doctor` first — GPU is used automatically when available | — |

---

## Installation

```bash
pip install visionservex                        # base: CLI, registry, mock
pip install 'visionservex[server]'              # + FastAPI HTTP server
pip install 'visionservex[hf]'                  # + D-FINE, GD, SwinV2, SAM, SAM2, OneFormer
pip install 'visionservex[rfdetr]'              # + RF-DETR and RF-DETR-Seg
pip install 'visionservex[server,hf,rfdetr]'    # full recommended install
```

For OpenMMLab models (RTMPose, RTMDet-R, Co-DINO):

```bash
pip install openmim
mim install mmengine mmcv mmpose mmdet mmrotate
```

See [docs/installation.md](docs/installation.md) for platform-specific notes.

---

## Python API

```python
from visionservex import VisionModel

# Object detection
m = VisionModel("rfdetr-nano")
result = m.predict("image.jpg")
for det in result.detections:
    print(det.label, f"{det.score:.2f}", det.box.to_xyxy())
result.save("annotated.jpg")

# D-FINE detection (HF Transformers)
m = VisionModel("dfine-s")
result = m.predict("image.jpg")

# SAM 2 (point prompt)
m = VisionModel("sam2-hiera-tiny")
result = m.predict("image.jpg", points=[[x, y]], point_labels=[1])

# SAM 2 (box prompt)
result = m.predict("image.jpg", boxes=[[x1, y1, x2, y2]])

# OneFormer (choose task)
m = VisionModel("oneformer-swin-large")
result = m.predict("image.jpg", task="semantic")       # or "instance", "panoptic"

# Grounding DINO
m = VisionModel("grounding-dino-tiny")
result = m.predict("image.jpg", prompts=["red car", "person walking"])

# Auto-pull on first use
m = VisionModel("dfine-s", auto_pull=True)
result = m.predict("image.jpg")
```

Stable result fields: `kind`, `model_id`, `task`, `device`, `precision`, `backend`,
`latency_ms`, `model_loaded_from`, `fallback_reason`, `warnings`.

---

## HTTP API

Stable response envelope:

```json
{
  "request_id": "...",
  "status": "completed",
  "model_id": "dfine-s",
  "task": "detect",
  "backend": "huggingface_dfine",
  "device": "cpu",
  "precision": "fp32",
  "latency_ms": 187.4,
  "results": [{"box": {...}, "score": 0.72, "label": "person", "class_id": 0}],
  "warnings": [],
  "metadata": {}
}
```

Error envelope:

```json
{
  "request_id": "...",
  "error": {
    "code": "MODEL_MISSING",
    "message": "Model weights for 'dfine-s' are not cached.",
    "hint": "Run: visionservex pull dfine-s",
    "details": {}
  }
}
```

Key endpoints: `GET /health`, `GET /devices`, `GET /models`,
`POST /detect`, `POST /segment`, `POST /classify`, `POST /open-vocab/detect`,
`POST /grounded-segment`, `GET /jobs/{id}`, `GET /metrics`.
Full reference: [docs/api_reference.md](docs/api_reference.md).

---

## Security defaults

| Setting | Default |
|---------|---------|
| Server bind | `127.0.0.1` only |
| Public mode | disabled (explicit opt-in) |
| Authentication | disabled — **enable before exposing** |
| Remote URL inputs | disabled (SSRF protection) |
| CORS | disabled |
| Upload limit | 20 MiB |
| Image pixel limit | ~33 MP (decompression-bomb guard) |
| Rate limit | 120 req/min per IP |
| Token redaction | enabled in all logs |

See [docs/security.md](docs/security.md) and [SECURITY.md](SECURITY.md).

---

## Safe Cloudflare Tunnel

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))")

visionservex tunnel doctor
visionservex tunnel create visionservex
visionservex tunnel route visionservex api.yourdomain.com
visionservex tunnel config api.yourdomain.com --out tunnel.yaml
visionservex serve &
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

The CLI refuses without auth enabled **and** the explicit confirmation flag.
The generated config always ends with a catch-all `http_status:404` rule.
See [docs/cloudflare_tunnel.md](docs/cloudflare_tunnel.md).

---

## Documentation

| | |
|-|-|
| [Beginner quickstart](docs/beginner_quickstart.md) | First prediction in 5 min |
| [Device check](docs/device_check.md) | GPU/CPU/MPS diagnostics |
| [Model zoo](docs/model_zoo.md) | All models, license table, "which model?" |
| [Model downloads](docs/model_downloads.md) | Download system, auto-pull |
| [Model licenses](docs/model_licenses.md) | Per-model license details |
| [Cloudflare Tunnel](docs/cloudflare_tunnel.md) | Safe public exposure |
| [Security](docs/security.md) | Threat model, all protections |
| [HTTP API reference](docs/api_reference.md) | Endpoints, error codes |
| [Python API](docs/python_api.md) | VisionModel, result types |
| [CLI reference](docs/cli.md) | Every command |
| [Troubleshooting](docs/troubleshooting.md) | Common errors |
| [LLM agent guide](docs/llm_agent_guide.md) | Stable CLI/JSON for agents |
| [About](docs/about.md) | Author, citation, acknowledgment |

---

## License and upstream models

VisionServeX is **Apache-2.0** (`SPDX-License-Identifier: Apache-2.0`).
See [LICENSE](LICENSE) and [NOTICE](NOTICE).

> **Each integrated model retains its own upstream license.**
> Review the model, checkpoint, and training-data licenses before commercial use.
> VisionServeX does not provide legal advice. See [docs/model_licenses.md](docs/model_licenses.md).

---

## Citation

```bibtex
@software{sajjadi2026visionservex,
  author = {Arash Sajjadi},
  title  = {{VisionServeX: A permissive-license-aware framework for
             local computer vision model serving}},
  year   = {2026},
  url    = {https://github.com/arashsajjadi/VisionServeX},
  note   = {Developed under the supervision of Prof. Mark Eramian,
            Department of Computer Science, University of Saskatchewan.}
}
```

**Author:** Arash Sajjadi — PhD Candidate, Department of Computer Science,
University of Saskatchewan  
**Supervision:** Prof. Mark Eramian, Computer Vision Lab, University of Saskatchewan  
*(This project is not an official product of the University of Saskatchewan.)*

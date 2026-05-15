<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

# VisionServeX

**Secure, beginner-friendly Python API serving for permissive computer vision models.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Code Style: ruff](https://img.shields.io/badge/code%20style-ruff-orange.svg)](https://docs.astral.sh/ruff/)

VisionServeX is a permissive-license-aware Python framework for serving
modern computer vision models locally and over Cloudflare Tunnel. It is
designed to be easy enough for a complete beginner — no CUDA expertise, no
HuggingFace Hub internals, no Cloudflare DNS knowledge required.

**Author:** Arash Sajjadi · PhD Candidate, Department of Computer Science,
University of Saskatchewan  
**Supervision:** Developed under the supervision of Prof. Mark Eramian,
Department of Computer Science, University of Saskatchewan, Computer Vision Lab.

> This project is not an official product of the University of Saskatchewan.

---

## Beginner quickstart (< 5 minutes)

```bash
pip install 'visionservex[server,hf]'

# 1. Diagnose your system
visionservex doctor

# 2. Get personalized recommendations
visionservex recommend --task detect --simple

# 3. Download a model
visionservex pull grounding-dino-tiny        # real, wired, text-prompted
# or
visionservex pull rfdetr-nano                # real, wired, COCO detection (needs `visionservex[rfdetr]`)
# or
visionservex pull mock-detect               # always works, no deps

# 4. Predict
visionservex predict grounding-dino-tiny examples/images/street.jpg \
    --prompt "car,person" --save outputs/out.jpg

# 5. Start the API
visionservex serve
```

In another terminal:

```bash
curl -F "image=@examples/images/street.jpg" \
     -F "model_id=grounding-dino-tiny" \
     -F "prompts=car,person" \
     http://127.0.0.1:8080/predict | jq
```

---

## "I have…" quickstart paths

| I have…               | Recommended model           | Install command                                   |
| --------------------- | --------------------------- | ------------------------------------------------- |
| No GPU                | `grounding-dino-tiny` (CPU) | `pip install 'visionservex[hf]'`                  |
| NVIDIA GPU            | `rfdetr-nano` or GD-tiny    | `pip install 'visionservex[rfdetr,hf]'`           |
| Just Python           | `mock-detect`               | `pip install visionservex` (no extras)            |
| Need detection        | `rfdetr-nano`, `rfdetr-small` | `pip install 'visionservex[rfdetr]'`             |
| Need segmentation     | `rfdetr-seg-nano`           | `pip install 'visionservex[rfdetr]'`              |
| Need text-prompt detection | `grounding-dino-tiny`  | `pip install 'visionservex[hf]'`                  |
| Need text-prompt masks | `grounded-sam`             | `pip install 'visionservex[hf]'`                  |
| Need classification   | `swinv2-tiny`               | `pip install 'visionservex[hf]'`                  |
| Need SAM segmentation | `sam-vit-base`              | `pip install 'visionservex[hf]'`                  |
| Need an API server    | any model                   | `pip install 'visionservex[server]'`              |
| Need Cloudflare Tunnel | any model + tunnel docs    | `pip install 'visionservex[server]'`              |

---

## Real model backends (Pass 3)

| Family                   | Model IDs                             | Task             | Status   | Backend          | CPU works? |
| ------------------------ | ------------------------------------- | ---------------- | -------- | ---------------- | ---------- |
| Mock (built-in)          | `mock-*`                              | All tasks        | stable   | built-in         | yes        |
| Grounding DINO           | `grounding-dino-tiny / swin-t / swin-b` | text-prompt detect | beta  | HF Transformers  | yes        |
| RF-DETR detection        | `rfdetr-nano / small / base / medium / large` | detect      | beta     | rfdetr package   | yes        |
| RF-DETR segmentation     | `rfdetr-seg-nano / small / medium`    | segment          | beta     | rfdetr package   | yes        |
| SwinV2 classification    | `swinv2-tiny / small / base / large`  | classify         | beta     | HF Transformers  | yes        |
| SAM v1                   | `sam-vit-base / large / huge`         | foundation_seg   | beta     | HF Transformers  | yes        |
| Grounded SAM (composed)  | `grounded-sam`                        | grounded_seg     | beta     | HF (GD + SAM)    | yes        |

Models still as stubs (registry-only, no real inference yet):
D-FINE, SAM 2 / 2.1 (needs `sam2` pip package), RTMPose, RTMDet-R/R2, InternImage, Co-DINO, SEEM, OneFormer, Grounded-SAM-2.

---

## Python API

```python
from visionservex import VisionModel

# Detection with RF-DETR (real, wired)
m = VisionModel("rfdetr-nano")
result = m.predict("examples/images/street.jpg")
print(result.summary())
for det in result.detections:
    print(det.label, det.score, det.box.to_xyxy())

# Text-prompted detection with Grounding DINO (real, wired)
m = VisionModel("grounding-dino-tiny")
result = m.predict("image.jpg", prompts=["red car", "person walking"])

# Classification with SwinV2 (real, wired)
m = VisionModel("swinv2-tiny")
result = m.predict("dog.jpg")
for label, score in result.top_k[:5]:
    print(label, score)

# Foundation segmentation with SAM v1 (real, wired)
m = VisionModel("sam-vit-base")
result = m.predict("image.jpg", points=[[100, 150]], point_labels=[1])
result.save("mask.jpg")

# Grounded segmentation (composed GD + SAM, real, wired)
m = VisionModel("grounded-sam")
result = m.predict("image.jpg", prompts=["dog", "leash"])
result.save("grounded.jpg")
```

Auto-pull (download weights on first use):

```python
m = VisionModel("rfdetr-small", auto_pull=True)
result = m.predict("image.jpg")
```

---

## Stable API response

```json
{
  "request_id": "abc123",
  "status": "completed",
  "model_id": "rfdetr-nano",
  "task": "detect",
  "backend": "rfdetr_package",
  "device": "cpu",
  "precision": "fp32",
  "latency_ms": 61.4,
  "model_loaded_from": "cache",
  "results": [
    {"box": {"x1": 59.0, "y1": 300.0, "x2": 201.5, "y2": 383.0},
     "score": 0.723, "label": "fire hydrant", "class_id": 10}
  ],
  "warnings": [],
  "metadata": {}
}
```

---

## CLI reference

```bash
# Diagnostics
visionservex getting-started              # beginner guide with exact next commands
visionservex doctor                       # full system + device + dependency report
visionservex doctor --fix-suggestions     # actionable fix commands
visionservex status                       # quick package/cache/device status
visionservex devices

# Models
visionservex list-models --friendly       # human-readable table
visionservex list-models --easy --can-run # only easy auto-downloadable models
visionservex recommend --task detect --simple
visionservex info grounding-dino-tiny

# Downloads
visionservex pull rfdetr-nano
visionservex pull-easy                    # all beginner auto-downloadable
visionservex pull-all --task detect --yes-i-understand-large-downloads

# Inference
visionservex predict rfdetr-nano examples/images/street.jpg --save outputs/out.jpg
visionservex predict grounding-dino-tiny examples/images/street.jpg \
    --prompt "car,person" --save outputs/gd.jpg
visionservex benchmark mock-detect examples/images/simple_shapes.jpg --n 20

# Server
visionservex serve --host 127.0.0.1 --port 8080

# Cloudflare Tunnel
visionservex tunnel doctor
visionservex tunnel config api.example.com --out tunnel.yaml
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

---

## Auto-pull during server requests

```bash
export VISIONSERVEX_MODELS__AUTO_PULL=true
export VISIONSERVEX_MODELS__AUTO_PULL_POLICY=easy_only
visionservex serve
```

Client-side: pass `?wait_for_download=false` to get a job id when the model is missing.  
Track progress at `GET /jobs/{id}`.

---

## Security defaults

| Default                     | Value              |
| ----------------------------| -------------------|
| Server bind address         | `127.0.0.1`        |
| Public mode                 | disabled           |
| API key authentication      | disabled           |
| Remote URL image input      | disabled           |
| Local file path input       | disabled           |
| CORS                        | disabled           |
| Max upload size             | 20 MiB             |
| Max image pixels            | ~33 MP             |
| Rate limit                  | 120 / minute       |
| Decompression-bomb guard    | enabled            |
| SSRF protection             | enabled            |
| Path traversal protection   | enabled            |
| Secret redaction in logs    | enabled            |

## Secure Cloudflare Tunnel

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(48))")

visionservex tunnel doctor
visionservex tunnel create visionservex
visionservex tunnel route visionservex api.example.com
visionservex tunnel config api.example.com --out tunnel.yaml
visionservex serve &
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

- The CLI refuses to start the tunnel unless auth is enabled AND the
  `--i-understand-this-is-public` flag is passed.
- The generated `tunnel.yaml` always ends with a catch-all `http_status:404`.
- We recommend adding a Cloudflare Access policy (service tokens for automation,
  mTLS for high-value clients). See [`docs/cloudflare_tunnel.md`](docs/cloudflare_tunnel.md).

---

## Documentation

| Document                                                           | Topic                           |
| ------------------------------------------------------------------ | ------------------------------- |
| [`docs/beginner_quickstart.md`](docs/beginner_quickstart.md)       | 5-minute walkthrough            |
| [`docs/installation.md`](docs/installation.md)                     | All install options             |
| [`docs/device_check.md`](docs/device_check.md)                     | GPU/CPU/MPS/ROCm guide          |
| [`docs/model_downloads.md`](docs/model_downloads.md)               | Download system, auto-pull      |
| [`docs/model_zoo.md`](docs/model_zoo.md)                           | Model table + "Which model?"    |
| [`docs/model_licenses.md`](docs/model_licenses.md)                 | Per-model license details       |
| [`docs/cloudflare_tunnel.md`](docs/cloudflare_tunnel.md)           | Secure public exposure          |
| [`docs/security.md`](docs/security.md)                             | Security model, threat model    |
| [`docs/api_reference.md`](docs/api_reference.md)                   | Full HTTP API spec              |
| [`docs/python_api.md`](docs/python_api.md)                         | Python API reference            |
| [`docs/cli.md`](docs/cli.md)                                       | CLI reference                   |
| [`docs/troubleshooting.md`](docs/troubleshooting.md)               | Common errors + fixes           |
| [`docs/llm_agent_guide.md`](docs/llm_agent_guide.md)               | For LLM agents / automation     |
| [`docs/about.md`](docs/about.md)                                   | Author, citation, acknowledgment |

---

## Honest status

We do not claim benchmark superiority for any model. Each registry entry
carries `implementation_status`:

- `wired` — real inference runs when the optional extra is installed.
- `partial` — code path exists but has rough edges.
- `stub` — registry entry only; no real inference in this build.

Models with uncertain licensing are flagged `license_uncertain=true` and
disabled/labelled accordingly.

---

## License

Apache-2.0 (`SPDX-License-Identifier: Apache-2.0`). See [`LICENSE`](LICENSE),
[`NOTICE`](NOTICE), and [`docs/model_licenses.md`](docs/model_licenses.md).

## Citation

```bibtex
@software{sajjadi2026visionservex,
  author = {Arash Sajjadi},
  title  = {{VisionServeX: A permissive-license-aware framework for local computer vision model serving}},
  year   = {2026},
  url    = {https://github.com/example/visionservex},
  note   = {Developed under the supervision of Prof. Mark Eramian,
            Department of Computer Science, University of Saskatchewan.}
}
```

See also [`CITATION.cff`](CITATION.cff).

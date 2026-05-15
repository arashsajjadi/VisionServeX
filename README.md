<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

<h1 align="center">VisionServeX</h1>

<p align="center">
  <strong>Local-first computer vision API gateway — secure, private, and honest.</strong><br>
  Serve modern CV models on your machine. Local-only by default. No data retained.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml">
    <img src="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/version-1.0.0-informational.svg" alt="v1.0.0">
  <img src="https://img.shields.io/badge/code%20style-ruff-orange.svg" alt="ruff">
</p>

---

## What is VisionServeX?

VisionServeX is an open-source, permissive-license-aware Python framework for running modern computer vision models locally and exposing them through a stable HTTP API. It works as a **local model gateway**: start it once, call any supported model through one clean API.

**Privacy-first design:**
- Binds to `127.0.0.1` by default — nothing leaves your machine.
- Images are decoded in memory for inference and never written to disk by default.
- No data is retained between requests by default.
- Log redaction removes tokens, base64, and API keys from all output.

> ⚠️ **No end-to-end encryption claimed.** VisionServeX cannot provide E2E encryption in the cryptographic sense — the inference server must see plaintext image tensors to run models. We provide local-first processing, no-retention defaults, optional encryption-at-rest for job metadata, and auth for public mode. See [docs/privacy.md](docs/privacy.md).

---

## Quickstart (CPU, 5 minutes)

```bash
pip install 'visionservex[server,hf,rfdetr]'

visionservex getting-started      # personalized guide
visionservex pull rfdetr-nano     # fast COCO detection, CPU-capable
visionservex serve                 # http://127.0.0.1:8080
```

```bash
curl -F "image=@image.jpg" -F "model_id=rfdetr-nano" \
     http://127.0.0.1:8080/detect | jq
```

---

## Python Client

```python
from visionservex import Client, VisionModel

# Direct inference (local, no server needed)
result = VisionModel("dfine-s").predict("image.jpg")

# Via local gateway
client = Client("http://127.0.0.1:8080")
result = client.detect("rfdetr-nano", "image.jpg")
result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
result = client.classify("swinv2-tiny", "image.jpg")
```

---

## What works today

| Family | Models | Task | Status | Install |
|--------|--------|------|--------|---------|
| **Mock** | `mock-*` | All | stable | base |
| **RF-DETR** | `rfdetr-nano/small/…` | detect | beta | `[rfdetr]` |
| **RF-DETR-Seg** | `rfdetr-seg-nano/small/…` | segment | beta | `[rfdetr]` |
| **D-FINE** | `dfine-n/s/m/l/x` | detect | beta | `[hf]` |
| **Grounding DINO** | `grounding-dino-tiny/…` | open-vocab detect | beta | `[hf]` |
| **SwinV2** | `swinv2-tiny/small/base/large` | classify | beta | `[hf]` |
| **SAM v1** | `sam-vit-base/large/huge` | foundation segment | beta | `[hf]` |
| **SAM 2** | `sam2-hiera-tiny/small/…` | foundation segment | beta | `[hf]` |
| **Grounded SAM** | `grounded-sam` | grounded segment | beta | `[hf]` |
| **Grounded-SAM2** | `grounded-sam2` | grounded segment | beta | `[hf]` |
| **OneFormer** | `oneformer-swin-large/…` | semantic/panoptic | beta | `[hf]` |
| **RTMPose** | `rtmpose-s/m/…` | pose | docker_checkpoint_required | `openmmlab` |
| **RTMDet-R/R2** | `rtmdet-r*/r2*` | OBB | docker_checkpoint_required | `openmmlab` |
| **ONNX export** | SwinV2 | — | working | `[onnx]` |
| **TensorRT** | — | — | experimental/dry-run | — |

**GPU:** CUDA verified on RTX 5080 for 6+ model families. Run `visionservex gpu smoke-test` on your hardware.  
**MPS (Apple Silicon):** Implemented, not maintainer-verified (no test hardware). See [docs/gpu_validation.md](docs/gpu_validation.md).  
**VRAM safety:** Desktop GPU guard reserves 3 GB for GUI/system. GPU tests run serially by default. See [docs/gpu_safety.md](docs/gpu_safety.md).

> **"beta" means:** CPU-verified, CUDA-verified when applicable, CLI/Python/gateway tested. No known regressions. May have edge cases.

---

## Known limitations

- **OpenMMLab** (RTMPose, RTMDet-R/R2, Co-DINO, InternImage): Requires the OpenMMLab toolchain and manually-obtained checkpoints. Returns `CHECKPOINT_REQUIRED` structured error — no fake output. See [docs/openmmlab_expert_models.md](docs/openmmlab_expert_models.md).
- **TensorRT**: ONNX export works for SwinV2. TensorRT engine build requires `trtexec` and is not implemented inside VisionServeX. See [docs/tensorrt.md](docs/tensorrt.md).
- **Apple MPS**: Implemented but not maintainer-verified. Users are encouraged to run `visionservex mps smoke-test --models swinv2-tiny,sam2-hiera-tiny` on Apple Silicon and report results.
- **In-flight cancellation**: Queued jobs can be cancelled. In-flight inference may complete before cancellation takes effect.

---

## Security and Privacy

```bash
# Check your current security posture
visionservex security audit --json

# Switch to public-mode configuration
visionservex security mode cloudflare_private --apply

# Generate an API key for development
visionservex gateway token

# Verify log redaction works
visionservex security test-redaction

# See what temp files exist
visionservex privacy inspect-cache
visionservex privacy cleanup --dry-run
```

**Security modes:**
| Mode | Binding | Auth | Notes |
|------|---------|------|-------|
| `local_private` | 127.0.0.1 | Optional | Default, safest |
| `lan_private` | LAN | Required | TLS recommended |
| `cloudflare_private` | 127.0.0.1 + tunnel | Required | Cloudflare Access recommended |
| `production_multi_user` | 127.0.0.1 + proxy | Required | Encrypted job store, audit logs |

---

## Safe Cloudflare Tunnel

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(visionservex gateway token 2>&1 | grep "API key:" | awk '{print $NF}')

visionservex tunnel config --domain api.yourdomain.com --out tunnel.yaml
visionservex serve &
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

---

## GPU Safety

```bash
# Check VRAM state and safety budget
visionservex gpu guard-status

# List GPU compute processes (GUI processes are protected)
visionservex gpu processes

# Safely clean up VisionServeX/pytest GPU processes
visionservex gpu cleanup --dry-run
visionservex gpu cleanup --yes
```

See [docs/gpu_safety.md](docs/gpu_safety.md) and [docs/parallel_safety.md](docs/parallel_safety.md).

---

## Installation

```bash
pip install visionservex                   # base (no heavy deps)
pip install 'visionservex[server]'         # + HTTP API server
pip install 'visionservex[hf]'             # + HF Transformers (D-FINE, GD, SwinV2, SAM, SAM2, OneFormer)
pip install 'visionservex[rfdetr]'         # + RF-DETR and RF-DETR-Seg
pip install 'visionservex[server,hf,rfdetr]'  # full recommended
```

OpenMMLab (RTMPose, RTMDet-R): Docker sidecar or `pip install openmim && mim install mmengine mmcv mmpose`. See [docs/openmmlab_expert_models.md](docs/openmmlab_expert_models.md).

---

## Syntax Contract

All 222 documented CLI/Python/API examples are covered and verified. No example is allowed to silently fail or return a raw traceback.

```bash
visionservex syntax audit             # verify 222 examples, failing must be 0
visionservex validation run release   # run full CI test suite
```

---

## Documentation

| | |
|-|-|
| [Beginner quickstart](docs/beginner_quickstart.md) | 5-minute guide |
| [Local gateway](docs/local_gateway.md) | Gateway commands and Python client |
| [Security](docs/security.md) | Threat model, modes, configuration |
| [Privacy](docs/privacy.md) | No E2E claim, retention policy, encryption |
| [Threat model](docs/threat_model.md) | What we protect and what we don't |
| [Model zoo](docs/model_zoo.md) | All 68 models with current status |
| [Model downloads](docs/model_downloads.md) | Download system, auto-pull |
| [GPU safety](docs/gpu_safety.md) | VRAM guard, cleanup, emergency recovery |
| [Parallel safety](docs/parallel_safety.md) | Model concurrency policies, benchmarks |
| [OpenMMLab expert](docs/openmmlab_expert_models.md) | RTMPose, RTMDet-R, Co-DINO, InternImage |
| [Cloudflare Tunnel](docs/cloudflare_tunnel.md) | Public mode safely |
| [GPU validation](docs/gpu_validation.md) | CPU/CUDA/MPS status |
| [TensorRT](docs/tensorrt.md) | ONNX export and TensorRT roadmap |
| [Benchmarks](docs/benchmarks.md) | Latency numbers |
| [Syntax contract](docs/syntax_contract.md) | 222 verified examples |
| [Troubleshooting](docs/troubleshooting.md) | Common errors |
| [About](docs/about.md) | Author, citation |

---

## License and model licenses

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

> Each integrated model retains its own upstream license. Review model, checkpoint, and dataset licenses before commercial use. See [docs/model_licenses.md](docs/model_licenses.md).

---

## Citation

```bibtex
@software{sajjadi2026visionservex,
  author = {Arash Sajjadi},
  title  = {{VisionServeX: A permissive-license-aware framework for local CV model serving}},
  year   = {2026},
  url    = {https://github.com/arashsajjadi/VisionServeX},
  note   = {Developed under the supervision of Prof. Mark Eramian, University of Saskatchewan.}
}
```

**Author:** Arash Sajjadi — PhD Candidate, Department of Computer Science, University of Saskatchewan  
**Supervision:** Prof. Mark Eramian, Computer Vision Lab  
*(This project is not an official product of the University of Saskatchewan.)*

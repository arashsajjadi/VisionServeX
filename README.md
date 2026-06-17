<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

<h1 align="center">VisionServeX</h1>

<p align="center">
  <strong>License-aware local computer-vision gateway — curated models, honest blockers, no fake claims.</strong><br>
  Serve modern CV models on your machine. Local-only by default. No data retained.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml">
    <img src="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/version-3.17.0-informational.svg" alt="v3.17.0">
  <img src="https://img.shields.io/badge/code%20style-ruff-orange.svg" alt="ruff">
</p>

---

## What is VisionServeX?

VisionServeX is an open-source Python framework for running modern computer vision models locally
and exposing them through a stable HTTP API. It works as a **local model gateway**: start it once,
call any supported model through one clean API, on your own hardware, with no data leaving your
machine.

Every model in the registry carries an explicit license classification, an honest availability
status, and a clear commercial posture. Blockers are documented — not hidden.

---

## Why VisionServeX?

- **Local-first inference** — binds to `127.0.0.1` by default; images never leave your machine
- **License-aware registry** — every model is classified: commercial-safe core, BYOT gated, non-commercial, or external-API-only
- **Stable Python + CLI + HTTP API** — one interface across detection, segmentation, embedding, depth, classification, and open-vocabulary tasks
- **Commercial-safe core** — SAM v1/2/2.1, DINOv2, RF-DETR, Florence-2, CLIP, OWLv2, Grounding DINO, and more; all Apache-2.0 or MIT, no token required
- **Permissive detector training** *(v3.13/v3.14)* — fine-tune LibreYOLO (YOLOX / YOLOv9 / RT-DETR / D-FINE) on your own YOLO datasets, reload trained checkpoints, and export to ONNX — no Ultralytics/AGPL. See [docs/libreyolo_training.md](docs/libreyolo_training.md). Standalone HF D-FINE stays inference-only; YOLO-NAS (non-commercial) is excluded.
- **Classic torchvision classifiers** *(new in v3.15.0)* — AlexNet / ResNet / ResNeXt / Wide-ResNet / DenseNet / MobileNet / EfficientNet / ConvNeXt: pretrained ImageNet inference, ImageFolder fine-tune, checkpoint reload, and ONNX export. BSD-3-Clause, commercial-safe. See [docs/torchvision_classifiers.md](docs/torchvision_classifiers.md).
- **Capability truth contract** *(new in v3.15.0)* — `VisionModel(id).capabilities()` returns an honest per-model object (legal status, inference/train readiness, reload+predict, export); no fake-ready states. Full inventory in [docs/qa/v315_model_coverage/](docs/qa/v315_model_coverage/).
- **BYOT support for gated models** — SAM3/SAM3.1, DINOv3; your token, your cache, your accepted license
- **No bundled gated weights** — VisionServeX never puts gated model weights into PyPI, GitHub, Docker, or any release artifact
- **Honest blockers** — unavailable models explain exactly why and what to do next

---

## Quickstart

```bash
pip install 'visionservex[hf,rfdetr]'

visionservex --version
visionservex getting-started        # personalized guide

# Detection (RF-DETR, no token needed)
visionservex detect rfdetr-small image.jpg

# Segmentation (SAM2.1, no token needed)
visionservex sam-family smoke-test sam2.1-hiera-small image.jpg

# HTTP gateway
visionservex serve                  # http://127.0.0.1:8080
curl -F "image=@image.jpg" -F "model_id=rfdetr-small" http://127.0.0.1:8080/detect | jq
```

```python
from visionservex import VisionModel, VSX

# Direct inference (no server)
result = VisionModel("rfdetr-small").predict("image.jpg")
result.to_json()

# SAM2 segmentation
VSX.sam("sam2.1-hiera-small").segment("image.jpg", box=[10, 20, 200, 220])

# DINOv2 embedding (Apache-2.0, no token)
VSX.dino("dinov2-base").embed("image.jpg")
```

---

## What works today

| Capability | Runnable models | Install | Token needed? |
|---|---|---|---|
| Object detection | D-FINE (n/s/m/l/x), RF-DETR (nano/small/medium/large) | `[hf]`, `[rfdetr]` | no |
| Instance segmentation | RF-DETR-Seg (nano/small/medium) | `[rfdetr]` | no |
| Promptable segmentation | SAM v1, SAM 2, SAM 2.1 | `[hf]` | no |
| SAM3/SAM3.1 BYOT masks | real mask artifacts (62 K–307 K px verified) | `[hf]` + HF token | yes, BYOT |
| Open-vocabulary detection | Grounding DINO (tiny, swin-b), OWL-ViT, OWLv2 | `[hf]` | no |
| Multi-task VLM | Florence-2 (base, large) | `[hf]` + isolated env | no |
| Classification | SwinV2, ConvNeXtV2, MaxViT | `[hf]` | no |
| Dense embedding | DINOv2 (s/b/l/g), CLIP, SigLIP2 | `[hf]` | no |
| DINOv3 BYOT embedding | dinov3-vits16 through dinov3-vit7b16 | `[dino]` + HF token | yes, BYOT |
| DINOv3 depth head | CHMv2 DPT depth estimation (`transformers>=5.10`) | `[dino]` + HF token | yes, BYOT |
| SAM2.1 ONNX encoder | image-encoder export + ONNX Runtime smoke | `[hf]` + onnx | yes, BYOT |
| INSID3 segmentation | training-free in-context segmentation (DINOv3 backbone) | `[hf]` + HF token | yes, BYOT |
| Medical segmentation | MedSAM (research only) | `[hf]` | no |
| Anomaly detection | PatchCore, PaDiM (via anomalib) | `[anomaly]` | no |
| Surveillance search | Index + text query (SigLIP2 + ByteTrack) | `[hf]` | no |
| HTTP API server | Full REST gateway | `[server]` | no |

---

## Hugging Face BYOT for gated models

SAM3, SAM3.1, and DINOv3 are gated on Hugging Face. VisionServeX provides a clean BYOT path:
you supply your own token, accept the upstream license once, and the weights stay in your local
HF cache.

```bash
# Step 1 — connect your token
huggingface-cli login                        # or:
visionservex hf connect --token-env HF_TOKEN

# Step 2 — check status (no download)
visionservex hf status
visionservex hf check-model facebook/sam3

# Step 3 — accept the upstream license on the model page, then pull
visionservex model pull sam3-base --accept-upstream-license
visionservex model doctor sam3-base
```

```python
from visionservex import VSX

VSX.hf.status()
VSX.model("sam3-base").pull(accept_upstream_license=True)
VSX.sam("sam3-base").segment("image.jpg", text="person")   # BYOT inference
VSX.dino("dinov3-vitb16").embed("image.jpg")               # BYOT embedding
```

### SAM3/SAM3.1 real mask results (v3.10.0)

Real mask artifacts are produced after accepting the upstream license. Both models have been
benchmarked locally and produce confirmed non-zero masks:

| Model | Mask area (px) | State |
|---|---|---|
| SAM3 (`facebook/sam3`) | 62,423 | `benchmark_passed_byot_mask` |
| SAM3.1 Base-Plus (`facebook/sam3.1`) | 306,808 | `benchmark_passed_byot_mask` |

> **VisionServeX does not redistribute gated model weights.** Weights remain in your HF cache.
> They are never included in PyPI, GitHub, Docker, or any VisionServeX release artifact.

---

## License and commercial posture

| Model group | Runs locally? | Token required? | Commercial-safe by default? | Can VisionServeX ship weights? | Default behavior |
|---|---|---|---|---|---|
| **Commercial-safe core** | yes | no | yes (Apache-2.0 / MIT) | download-on-demand, no bundling | enabled |
| **BYOT gated** | yes, after access | yes (HF token) | depends on upstream license you accepted | no | disabled until you accept |
| **External API only** | no local weights | provider key | depends on provider terms | no | connector only |
| **Research / non-commercial** | local maybe | maybe | no | no | disabled |
| **Legal review** | maybe | maybe | no until reviewed | no | disabled |

**Commercial-safe core** (39 models) includes: SAM v1/2/2.1, DINOv2, RF-DETR family, Grounding
DINO (open variants), Florence-2, CLIP, OWLv2, SigLIP2, SwinV2, ConvNeXtV2, depth-anything-small,
and more. No token needed; weights download from official upstream sources on demand.

**BYOT models** (SAM3, SAM3.1, DINOv3) are not automatically commercial-safe. Commercial use
depends on the upstream license *you* accepted. VisionServeX provides the infrastructure; the
license decision is yours.

Detailed counts and bucket breakdowns: [docs/global_model_count.md](docs/global_model_count.md)

---

## Model Families

| Family | Best model | Status | Install |
|---|---|---|---|
| D-FINE | `dfine-s-o365-coco` | runnable | `[hf]` |
| RF-DETR | `rfdetr-large` | runnable | `[rfdetr]` |
| RF-DETR-Seg | `rfdetr-seg-medium` | runnable | `[rfdetr]` |
| SAM v1 | `sam-vit-base` | runnable | `[hf]` |
| SAM 2 / 2.1 | `sam2.1-hiera-large` | runnable | `[hf]` |
| SAM 3 / 3.1 | `sam3-base`, `sam3.1-base-plus` | BYOT (gated) | `[hf]` + token |
| Florence-2 | `florence-2-large` | runnable (isolated env) | `[hf]` |
| OWLv2 | `owlv2-large-patch14` | runnable | `[hf]` |
| Grounding DINO | `grounding-dino-swin-b` | runnable | `[hf]` |
| SwinV2 | `swinv2-base` | runnable | `[hf]` |
| DINOv2 | `dinov2-large` | runnable | `[hf]` |
| DINOv3 | `dinov3-vitb16` | BYOT (gated) | `[dino]` + token |
| DINOv3 CHMv2 depth | `dinov3-vitl16-chmv2-dpt-head` | BYOT, `transformers>=5.10` | `[dino]` + token |
| CLIP / SigLIP2 | `clip-vit-large-patch14` | runnable | `[hf]` |
| MedSAM | `medsam` | runnable (research) | `[hf]` |
| PatchCore | `anomalib-patchcore` | optional_extra | `[anomaly]` |
| RTMPose | `rtmpose-m` | expert_sidecar | OpenMMLab conda |
| ByteTrack / OC-SORT | `bytetrack` | optional, pip install | `pip install bytetracker` |
| MaskDINO | `maskdino-swinl-coco` | expert_sidecar | Detectron2 sidecar |

Models marked **expert_sidecar** require an isolated environment (OpenMMLab, Detectron2). Use
`visionservex openmmlab create-env` for the exact conda recipe.

Models marked **BYOT (gated)** require accepting the upstream license on Hugging Face.

---

## Python API

```python
from visionservex import VisionModel, VSX, Client

# Direct inference (no server needed)
result = VisionModel("dfine-s-o365-coco").predict("image.jpg")
result.to_json()
result.save("outputs/")
result.plot()                          # PIL Image

# Context manager with GPU cleanup
with VisionModel("rfdetr-large", device="cuda") as model:
    result = model.predict("image.jpg")

# HTTP client
client = Client("http://127.0.0.1:8080")
result = client.detect("rfdetr-small", "image.jpg")
result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
```

Output normalization handles all common detection serialization formats:

```python
from visionservex import normalize_detections

dets = normalize_detections([
    {"xyxy": [10, 20, 100, 200], "score": 0.9, "label": "cat"},
    {"box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}, "confidence": 0.8, "category": "dog"},
])
```

---

## HTTP Gateway

```bash
pip install 'visionservex[server,hf,rfdetr]'
visionservex serve                     # http://127.0.0.1:8080

# Detect
curl -F "image=@image.jpg" -F "model_id=rfdetr-small" http://127.0.0.1:8080/detect | jq

# Segment
curl -F "image=@image.jpg" -F "model_id=sam2.1-hiera-small" \
     -F 'box=[10,20,200,220]' http://127.0.0.1:8080/segment | jq
```

Public mode (Cloudflare Tunnel):

```bash
export VISIONSERVEX_AUTH__ENABLED=true
export VISIONSERVEX_AUTH__API_KEY=$(visionservex gateway token 2>&1 | grep "API key:" | awk '{print $NF}')
visionservex tunnel config --domain api.yourdomain.com --out tunnel.yaml
visionservex serve &
visionservex tunnel run tunnel.yaml --i-understand-this-is-public
```

---

## CLI Reference (selected)

```bash
# Model lifecycle
visionservex detect dfine-s-o365-coco image.jpg --conf 0.25
visionservex segment rfdetr-seg-medium image.jpg
visionservex classify swinv2-base image.jpg --top-k 5
visionservex open-vocab grounding-dino-swin-b image.jpg --prompt "car,person"
visionservex embed dinov2-base image.jpg --out embedding.npy
visionservex similarity siglip2-base-patch16-224 a.jpg b.jpg

# Model info
visionservex model pull rfdetr-small
visionservex model info rfdetr-small
visionservex model license sam3-base
visionservex model doctor sam3-base
visionservex model-card show dfine-s-o365-coco

# Capabilities and health
visionservex capabilities report
visionservex models health --runnable-only
visionservex recommend --task detect --goal accuracy

# HF BYOT
visionservex hf status
visionservex hf whoami
visionservex hf check-model facebook/sam3

# Benchmark (requires annotated dataset)
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --max-images 20 --device auto

# Debug
visionservex debug-output rfdetr-small image.jpg --threshold 0.01
visionservex dev resources
visionservex dev test quick
```

---

## Feature Intelligence

```bash
# Build a similarity index
visionservex embed dinov2-base folder/ --out embeddings/
visionservex index dinov2-base folder/ --out indexes/dinov2_base
visionservex search dinov2-base query.jpg --index indexes/dinov2_base --top-k 10
visionservex deduplicate dinov2-base folder/ --threshold 0.98 --out dups.csv

# Surveillance search (index + text query)
visionservex video-search index ./frames/ \
  --detector owlv2-base-patch16 --embedder siglip2-base-patch16-224 \
  --prompt "person" --out indexes/camera01
visionservex video-search query indexes/camera01 --text "red shirt" --top-k 20
```

---

## Privacy and Security

- Binds to `127.0.0.1` by default — nothing leaves your machine
- Images decoded in memory; never written to disk by default
- Log redaction removes tokens and API keys from all output
- No data retained between requests by default

> VisionServeX cannot provide E2E encryption — the inference server must see
> plaintext image tensors. It provides local-first processing, no-retention
> defaults, and auth for public mode. See [docs/privacy.md](docs/privacy.md).

```bash
visionservex security audit --json
visionservex privacy inspect-cache
visionservex privacy cleanup --dry-run
```

---

## Resource Safety

VisionServeX includes a resource guard (RAM / VRAM / disk) that prevents exhaustion during
testing and development. Production CLI commands are unaffected.

```bash
visionservex dev resources
visionservex dev gpu-profile --format json
visionservex gpu guard-status
visionservex gpu cleanup --dry-run
visionservex dev kill-tests          # kill stray pytest processes
```

Default budgets: 8 GB free RAM, 2 GB free VRAM, 10 GB free disk.
See [docs/agent_safety.md](docs/agent_safety.md) and [AGENT_RULES.md](AGENT_RULES.md).

---

## Installation

```bash
pip install visionservex                        # base (no heavy deps)
pip install 'visionservex[server]'              # + HTTP API server
pip install 'visionservex[hf]'                  # + HF Transformers (D-FINE, SAM, GD, SwinV2, …)
pip install 'visionservex[rfdetr]'              # + RF-DETR and RF-DETR-Seg
pip install 'visionservex[dino]'                # + DINOv3 depth head (transformers>=5.10)
pip install 'visionservex[server,hf,rfdetr]'    # full recommended
```

OpenMMLab (RTMPose, RTMDet-R, Co-DINO): Docker sidecar or
`pip install openmim && mim install mmengine mmcv mmpose`.
See [docs/openmmlab_expert_models.md](docs/openmmlab_expert_models.md).

---

## Known Limitations

- **SAM3/SAM3.1**: Gated on Hugging Face; requires accepting the upstream license. BYOT only.
- **DINOv3 CHMv2 depth head**: Requires `transformers>=5.10`; may conflict with Florence-2 (<5.0) — install in a separate env.
- **Florence-2**: Requires isolated env (`transformers==4.46.3 + einops + timm`). Use `visionservex florence2 create-env` for the validated recipe.
- **DEIMv2**: Registered but not wired — no HF Transformers support yet; custom loader required.
- **MedSAM**: Research only; non-commercial restricted.
- **SAM2.1 ONNX**: Image-encoder export works via Module shim. Full interactive decoder ONNX not yet verified.
- **OpenMMLab** (RTMPose, RTMDet-R/R2, Co-DINO, InternImage): Expert sidecar; use `visionservex openmmlab create-env`.
- **Apple MPS**: Implemented but not maintainer-verified.
- **GPU**: CUDA verified on RTX 5080 for 6+ model families. Run `visionservex gpu smoke-test`.

---

## Documentation

| | |
|-|-|
| [Beginner quickstart](docs/beginner_quickstart.md) | 5-minute guide |
| [Global model count](docs/global_model_count.md) | Policy rows, manifest entries, runnable count |
| [BYOT models](docs/byot_models.md) | SAM3, SAM3.1, DINOv3 — gated model usage |
| [Commercial-safe core](docs/commercial_safe_core.md) | Apache-2.0/MIT models enabled by default |
| [Model license policy](docs/model_license_policy.md) | Full policy bucket reference |
| [SAM3 mask benchmark](docs/sam3_mask_benchmark.md) | Real mask evidence (v3.10.0) |
| [DINOv3 depth head](docs/dinov3_depth.md) | CHMv2 DPT depth estimation |
| [SAM2.1 ONNX](docs/sam21_onnx.md) | Image-encoder export and ONNX Runtime |
| [INSID3](docs/insid3.md) | Training-free in-context segmentation (CVPR 2026 Oral) |
| [Local gateway](docs/local_gateway.md) | Gateway commands and Python client |
| [Security](docs/security.md) | Threat model, modes, configuration |
| [Privacy](docs/privacy.md) | Retention policy, encryption |
| [Model zoo](docs/model_zoo.md) | Full model list with status |
| [Benchmark competitiveness](docs/benchmark_competitiveness.md) | AP/mAP evaluation |
| [GPU safety](docs/gpu_safety.md) | VRAM guard, cleanup |
| [OpenMMLab expert](docs/openmmlab_expert_models.md) | RTMPose, RTMDet-R, Co-DINO |
| [Colab GPU worker](docs/colab_gpu_worker.md) | Temporary GPU demo worker |
| [Troubleshooting](docs/troubleshooting.md) | Common errors |
| [Reports and audits](docs/reports.md) | Detailed evidence, ledgers |

---

## Reports and Audits

Detailed audit evidence — policy matrices, benchmark ledgers, test run reports,
execution logs — lives in `docs/reports.md` and
`notebook/99_final_report/reports/`. It is internal audit data, not public
marketing. The counts there (policy rows, manifest entries, test passes) are
measurement evidence, not product-health scores.

See [docs/reports.md](docs/reports.md) and [docs/global_model_count.md](docs/global_model_count.md).

---

## License and Model Licenses

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

> Each integrated model retains its own upstream license. Review model, checkpoint, and
> dataset licenses before commercial use. See [docs/model_licenses.md](docs/model_licenses.md).

---

## Citation

```bibtex
@software{sajjadi2026visionservex,
  author = {Arash Sajjadi},
  title  = {{VisionServeX: A license-aware framework for local CV model serving}},
  year   = {2026},
  url    = {https://github.com/arashsajjadi/VisionServeX},
  note   = {Developed under the supervision of Prof. Mark Eramian, University of Saskatchewan.}
}
```

**Author:** Arash Sajjadi — PhD Candidate, Department of Computer Science, University of Saskatchewan  
**Supervision:** Prof. Mark Eramian, Computer Vision Lab  
*(This project is not an official product of the University of Saskatchewan.)*

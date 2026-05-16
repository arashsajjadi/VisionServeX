<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Arash Sajjadi -->

<h1 align="center">VisionServeX</h1>

<p align="center">
  <strong>Accuracy-aware computer vision model gateway — honest, local-first, and privacy-respecting.</strong><br>
  Serve modern CV models on your machine. Local-only by default. No data retained.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <a href="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml">
    <img src="https://github.com/arashsajjadi/VisionServeX/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI">
  </a>
  <img src="https://img.shields.io/badge/version-1.2.0-informational.svg" alt="v1.2.0">
  <img src="https://img.shields.io/badge/code%20style-ruff-orange.svg" alt="ruff">
</p>

---

## What is VisionServeX?

VisionServeX is an open-source, permissive-license-aware Python framework for running modern computer vision models locally and exposing them through a stable HTTP API. It works as a **local model gateway**: start it once, call any supported model through one clean API.

**Accuracy-aware design (v1.2.0):**  
Every model carries an explicit accuracy taxonomy label: `demo_fast`, `production_recommended`, `accuracy_grade`, `experimental_sota`, `expert_sidecar`, `external_api`, or `unavailable_with_reason`. The recommender, benchmark tools, and registry are aligned to these labels so you always know what tier you are running.

**Honesty policy:**  
VisionServeX does not claim to beat Ultralytics globally. The `benchmark-competitiveness` tool is designed to reveal the honest truth. If YOLO wins, it will say so.

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
visionservex pull dfine-s-o365-coco   # accuracy-grade detection, CPU-capable
visionservex serve                     # http://127.0.0.1:8080
```

```bash
curl -F "image=@image.jpg" -F "model_id=dfine-s-o365-coco" \
     http://127.0.0.1:8080/detect | jq
```

For a quick demo (smallest model):
```bash
visionservex pull rfdetr-nano          # demo_fast, CPU-capable
visionservex predict rfdetr-nano image.jpg
```

---

## Python Client

```python
from visionservex import Client, VisionModel

# Direct inference (local, no server needed)
result = VisionModel("dfine-s-o365-coco").predict("image.jpg")   # accuracy_grade
result = VisionModel("rfdetr-nano").predict("image.jpg")          # demo_fast

# Via local gateway
client = Client("http://127.0.0.1:8080")
result = client.detect("dfine-s-o365-coco", "image.jpg")
result = client.grounded_segment("grounded-sam2", "image.jpg", prompt="car, person")
result = client.classify("swinv2-tiny", "image.jpg")
```

---

## Model Taxonomy

Every model in the registry now carries an explicit `model_category` label.

| Category | Meaning | Examples |
|----------|---------|---------|
| `demo_fast` | Quick demo, small, not for accuracy benchmarks | `dfine-n`, `rfdetr-nano`, `rfdetr-seg-nano`, `grounding-dino-tiny` |
| `production_recommended` | Solid accuracy, ready for real use | `rfdetr-small`, `rfdetr-seg-small`, `swinv2-tiny`, `sam-vit-base` |
| `accuracy_grade` | Tracked for AP benchmarks; explicitly wired | `dfine-s-o365-coco`, `dfine-m/l/x-o365-coco`, `rfdetr-medium/large`, `grounding-dino-swin-b` |
| `experimental_sota` | Claims SOTA but not fully verified in this build | `deim-s/m`, `deimv2-s/m`, `rtdetrv4-s/m/l/x`, `maskdino-r50-coco` |
| `expert_sidecar` | Requires expert setup (OpenMMLab, custom ops) | `rtmpose-*`, `internimage-*`, `co-dino-*` |
| `external_api` | API-gated upstream; not self-hostable | `grounding-dino-1.5/1.6` |
| `unavailable_with_reason` | Blocked; honest reason documented | `rfdetr-seg-large/xlarge/2xlarge` |
| `utility` | Mock / built-in / test helpers | `mock-detect`, `mock-classify`, … |

**Key rule:** `demo_fast` models are not used to claim competitiveness with YOLO. Use `accuracy_grade` variants for AP benchmarks.

---

## What works today

### Detection (wired, runnable)

| Model ID | Category | Checkpoint | Install |
|----------|----------|-----------|---------|
| `dfine-n` / `dfine-n-coco` | demo_fast | ustc-community/dfine-nano-coco | `[hf]` |
| `dfine-s-o365-coco` ★ | **accuracy_grade** | ustc-community/dfine-small-obj2coco | `[hf]` |
| `dfine-m-o365-coco` | **accuracy_grade** | ustc-community/dfine-medium-obj2coco | `[hf]` |
| `dfine-l-o365-coco` | **accuracy_grade** | ustc-community/dfine-large-obj2coco-e25 | `[hf]` |
| `dfine-x-o365-coco` | **accuracy_grade** | ustc-community/dfine-xlarge-obj2coco | `[hf]` |
| `rfdetr-nano` | demo_fast | rfdetr pkg | `[rfdetr]` |
| `rfdetr-small` ★ | production_recommended | rfdetr pkg | `[rfdetr]` |
| `rfdetr-medium` | **accuracy_grade** | rfdetr pkg | `[rfdetr]` |
| `rfdetr-large` | **accuracy_grade** | rfdetr pkg | `[rfdetr]` |

★ Recommended accuracy entry points: `dfine-s-o365-coco` (CPU-capable) and `rfdetr-small` (GPU-preferred).

### Segmentation

| Family | Models | Category | Install |
|--------|--------|----------|---------|
| RF-DETR-Seg | `rfdetr-seg-nano/small/medium` | demo_fast / production_recommended / accuracy_grade | `[rfdetr]` |
| SAM v1 | `sam-vit-base/large/huge` | production_recommended / accuracy_grade | `[hf]` |
| SAM 2 | `sam2-hiera-tiny/small/base-plus/large` | production_recommended / accuracy_grade | `[hf]` |
| Grounded SAM | `grounded-sam`, `grounded-sam2` | production_recommended | `[hf]` |
| OneFormer | `oneformer-swin-large/dinat-large/convnext-large` | accuracy_grade | `[hf]` |

### Classification

| Family | Models | Category | Install |
|--------|--------|----------|---------|
| SwinV2 | `swinv2-tiny/small` | production_recommended | `[hf]` |
| SwinV2 | `swinv2-base/large` | accuracy_grade | `[hf]` |
| InternImage | `internimage-t/s/b/l/h` | expert_sidecar | OpenMMLab |

### Open-Vocabulary Detection

| Model | Category | Install |
|-------|----------|---------|
| `grounding-dino-tiny` | demo_fast | `[hf]` |
| `grounding-dino-swin-b` | accuracy_grade | `[hf]` |
| `grounding-dino-1.5/1.6` | external_api | API token required |

### Experimental SOTA (stub — not runnable yet)

| Family | Models | Blocker |
|--------|--------|---------|
| DEIM | `deim-s/m`, `deimv2-s/m` | No HF/pip path; custom loader + license verification needed |
| RT-DETRv4 | `rtdetrv4-s/m/l/x` | No official release numbering; checkpoint source unclear |
| MaskDINO | `maskdino-r50-coco/panoptic` | detectron2 environment required |

---

## Competitiveness Benchmark

```bash
# Compare models head-to-head (latency + detection health)
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --max-images 20 \
  --device auto

# Add YOLO baseline (requires ultralytics)
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small,ultralytics:yolo11n \
  --max-images 50

# Export results as JSON
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --max-images 20 --json
```

**Note:** This tool reports latency and output health diagnostics. AP50/mAP computation requires ground-truth COCO annotations (not included). The tool is designed to be honest — if YOLO wins on latency, it will say so.

---

## Debug Output

Before declaring a checkpoint weak, run the postprocessing audit:

```bash
visionservex debug-output dfine-s-o365-coco image.jpg
visionservex debug-output dfine-s-o365-coco image.jpg --threshold 0.01 --json
```

Reports: score histogram, label histogram, first 10 boxes, invalid boxes, unmapped labels, preprocessing notes.

---

## Model Recommender

```bash
# By goal (v1.2.0)
visionservex recommend --task detect --goal accuracy
visionservex recommend --task detect --goal fastest_demo
visionservex recommend --goal best_segmentation
visionservex recommend --goal best_open_vocab

# By task and hardware
visionservex recommend --task detect --device cpu
visionservex recommend --task detect --device cuda --vram 8
```

For `--goal accuracy --task detect`, the recommender surfaces `dfine-s/m-o365-coco` and `rfdetr-small/medium`, not nano variants.

---

## Security and Privacy

```bash
visionservex security audit --json
visionservex security mode cloudflare_private --apply
visionservex gateway token
visionservex security test-redaction
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
visionservex gpu guard-status
visionservex gpu processes
visionservex gpu cleanup --dry-run
visionservex gpu cleanup --yes
```

See [docs/gpu_safety.md](docs/gpu_safety.md) and [docs/parallel_safety.md](docs/parallel_safety.md).

---

## Temporary Colab GPU Worker (optional)

Run VisionServeX on a Google Colab GPU as a short-lived remote worker. Good for demos and benchmarks, **not** for production — Colab sessions can disconnect at any time.

```bash
# Inside a Colab notebook:
!pip install -U 'visionservex[server,hf,rfdetr]'
!visionservex colab doctor
!visionservex gateway start --profile colab-gpu-worker
```

A copy-paste notebook lives at [`examples/colab/VisionServeX_Colab_GPU_Worker.ipynb`](examples/colab/VisionServeX_Colab_GPU_Worker.ipynb). Full guide: [docs/colab_gpu_worker.md](docs/colab_gpu_worker.md).

---

## Installation

```bash
pip install visionservex                        # base (no heavy deps)
pip install 'visionservex[server]'              # + HTTP API server
pip install 'visionservex[hf]'                  # + HF Transformers (D-FINE, GD, SwinV2, SAM, SAM2, OneFormer)
pip install 'visionservex[rfdetr]'              # + RF-DETR and RF-DETR-Seg
pip install 'visionservex[server,hf,rfdetr]'    # full recommended
```

OpenMMLab (RTMPose, RTMDet-R, Co-DINO, InternImage): Docker sidecar or `pip install openmim && mim install mmengine mmcv mmpose`. See [docs/openmmlab_expert_models.md](docs/openmmlab_expert_models.md).

---

## Known Limitations

- **D-FINE COCO-only variants** (`dfine-s-coco` etc.): Point to HF repos that may not exist yet. Use `dfine-s-o365-coco` (Objects365+COCO) for guaranteed availability.
- **DEIM / RT-DETRv4**: Registered as `experimental_sota` but not wired. Blockers documented per-model in the registry.
- **AP50/mAP benchmark**: The `benchmark-competitiveness` tool reports latency and detection health only. Full AP evaluation requires ground-truth COCO annotations not bundled with VisionServeX.
- **OpenMMLab** (RTMPose, RTMDet-R/R2, Co-DINO, InternImage): Requires the OpenMMLab toolchain and manually-obtained checkpoints. Returns `CHECKPOINT_REQUIRED` structured error — no fake output.
- **TensorRT**: ONNX export works for SwinV2. TensorRT engine build requires `trtexec`.
- **Apple MPS**: Implemented but not maintainer-verified.

**GPU:** CUDA verified on RTX 5080 for 6+ model families. Run `visionservex gpu smoke-test` on your hardware.  
**MPS (Apple Silicon):** Implemented, not maintainer-verified. See [docs/gpu_validation.md](docs/gpu_validation.md).  
**VRAM safety:** Desktop GPU guard reserves 3 GB for GUI/system. GPU tests run serially by default. See [docs/gpu_safety.md](docs/gpu_safety.md).

---

## Syntax Contract

All documented CLI/Python/API examples are covered and verified. No example is allowed to silently fail or return a raw traceback.

```bash
visionservex syntax audit             # verify examples, failing must be 0
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
| [Model zoo](docs/model_zoo.md) | All 87 models with current status and taxonomy |
| [Model downloads](docs/model_downloads.md) | Download system, auto-pull |
| [GPU safety](docs/gpu_safety.md) | VRAM guard, cleanup, emergency recovery |
| [Parallel safety](docs/parallel_safety.md) | Model concurrency policies, benchmarks |
| [Colab GPU worker](docs/colab_gpu_worker.md) | Run VisionServeX on a Colab GPU for demos |
| [OpenMMLab expert](docs/openmmlab_expert_models.md) | RTMPose, RTMDet-R, Co-DINO, InternImage |
| [Cloudflare Tunnel](docs/cloudflare_tunnel.md) | Public mode safely |
| [GPU validation](docs/gpu_validation.md) | CPU/CUDA/MPS status |
| [TensorRT](docs/tensorrt.md) | ONNX export and TensorRT roadmap |
| [Benchmarks](docs/benchmarks.md) | Latency numbers |
| [Troubleshooting](docs/troubleshooting.md) | Common errors |
| [About](docs/about.md) | Author, citation |

---

## License and Model Licenses

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

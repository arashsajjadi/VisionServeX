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
  <img src="https://img.shields.io/badge/version-2.13.1-informational.svg" alt="v2.13.1">
  <img src="https://img.shields.io/badge/code%20style-ruff-orange.svg" alt="ruff">
</p>

---

## What is VisionServeX?

VisionServeX is an open-source, permissive-license-aware Python framework for running modern computer vision models locally and exposing them through a stable HTTP API. It works as a **local model gateway**: start it once, call any supported model through one clean API.

**Accuracy-aware and scientifically usable:**  
Every model carries an explicit accuracy taxonomy label: `demo_fast`, `production_recommended`, `accuracy_grade`, `experimental_sota`, `expert_sidecar`, `external_api`, or `unavailable_with_reason`. The recommender, benchmark tools, and registry are aligned to these labels so you always know what tier you are running. Real AP50/mAP50:95 is computed when you provide an annotated dataset.

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

## Ultralytics-Like Workflow

Same mental model — different backends, all permissive-license.

```python
from visionservex import VisionModel

model = VisionModel("dfine-x-o365-coco")
model.pull()                          # download checkpoint
model.info()                          # show registry metadata
results = model.predict("image.jpg", conf=0.25)
results.save("outputs/")             # save annotated image
results.plot()                        # returns PIL Image
results.to_json()                     # JSON string
results.to_csv()                      # CSV string
results.debug()                       # detailed debug string

# Check what operations are supported
model.supports("val")                 # {"supported": True, ...}
model.supports("train")               # {"supported": False, "reason": "..."}
model.training_info()                 # per-family training capabilities
model.export_info()                   # per-family export capabilities
model.val(dataset="yolo:/data/coco128", max_images=100)  # AP50/mAP50:95
```

> **Note:** Not all operations exist for all models. Use `model.supports("operation")`
> and `visionservex model-card show MODEL` to check capabilities.
> Unlike Ultralytics, VisionServeX does not depend on Ultralytics as a package.

```bash
# CLI task aliases
visionservex detect dfine-x-o365-coco image.jpg --conf 0.25 --device cuda
visionservex segment rfdetr-seg-medium image.jpg --save-image out.jpg
visionservex classify swinv2-base image.jpg --top-k 5
visionservex open-vocab grounding-dino-swin-b image.jpg --prompt "car,person"
visionservex val dfine-x-o365-coco --dataset yolo:/path/to/coco128 --max-images 128

# Model lifecycle
visionservex model pull dfine-x-o365-coco --dry-run
visionservex model info dfine-x-o365-coco
visionservex model checkpoint-info dfine-x-o365-coco
visionservex training capabilities --model rfdetr-large
visionservex export-cmd capabilities --model dfine-x-o365-coco
```

---

## Output Normalization

The built-in normalizer handles all common detection serialization formats:

```python
from visionservex import normalize_detections, parse_api_response

# Accepts all these formats:
dets = normalize_detections([
    {"xyxy": [10, 20, 100, 200], "score": 0.9, "label": "cat"},
    {"box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}, "confidence": 0.8, "category": "dog"},
    {"bbox": [10, 20, 90, 180], "bbox_format": "xywh", "conf": 0.7, "class_id": 0},
])

# Parse VisionServeX HTTP API responses directly:
import requests
resp = requests.get("http://127.0.0.1:8080/detect", ...)
dets = parse_api_response(resp.json())
```

Never silently drops all predictions — emits `AllPredictionsDroppedWarning` if normalization fails.

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

## Model Families

Full detail: [docs/model_zoo_matrix.md](docs/model_zoo_matrix.md) | [docs/model_zoo_gap_report.md](docs/model_zoo_gap_report.md)

Release readiness telemetry (functional/operational/certainty per family)
lives in [docs/release_readiness/latest.md](docs/release_readiness/latest.md);
run `visionservex readiness verdict --json` to re-check it locally.

| Family | Best Model | Status | Install | Example |
|--------|-----------|--------|---------|---------|
| D-FINE | `dfine-s-o365-coco` | runnable | `[hf]` | `visionservex detect dfine-s-o365-coco image.jpg` |
| RF-DETR | `rfdetr-large` | runnable | `[rfdetr]` | `visionservex detect rfdetr-large image.jpg` |
| RF-DETR-Seg | `rfdetr-seg-medium` | runnable | `[rfdetr]` | `visionservex segment rfdetr-seg-medium image.jpg` |
| SAM v1 | `sam-vit-base` | runnable | `[hf]` | `visionservex sam-family smoke-test sam-vit-base img.jpg` |
| SAM 2 | `sam2-hiera-tiny` | runnable | `[hf]` | `visionservex sam-family smoke-test sam2-hiera-tiny img.jpg` |
| SAM 2.1 | `sam2.1-hiera-large` | runnable | `[hf]` | `visionservex sam-family smoke-test sam2.1-hiera-large img.jpg` |
| Florence-2 | `florence-2-large` | runnable | `[hf]` | `visionservex florence2 predict florence-2-large img.jpg --task '<OD>'` |
| OWLv2 | `owlv2-large-patch14` | runnable | `[hf]` | `visionservex open-vocab owlv2-large-patch14 img.jpg --prompt "cat"` |
| OWL-ViT | `owlvit-large-patch14` | runnable | `[hf]` | `visionservex open-vocab owlvit-large-patch14 img.jpg --prompt "dog"` |
| Grounding DINO | `grounding-dino-swin-b` | runnable | `[hf]` | `visionservex open-vocab grounding-dino-swin-b img.jpg --prompt "car"` |
| SwinV2 | `swinv2-base` | runnable | `[hf]` | `visionservex classify swinv2-base image.jpg --top-k 5` |
| ConvNeXtV2 | `convnextv2-large` | runnable | `[hf]` | `visionservex classify convnextv2-large image.jpg` |
| DINOv2 | `dinov2-large` | runnable | `[hf]` | `visionservex feature embed dinov2-large image.jpg` |
| CLIP | `clip-vit-large-patch14` | runnable | `[hf]` | `visionservex feature embed clip-vit-large-patch14 image.jpg` |
| SigLIP2 | `siglip2-base-patch16-224` | runnable | `[hf]` | `visionservex feature embed siglip2-base-patch16-224 image.jpg` |
| MedSAM | `medsam` | runnable | `[hf]` | `visionservex medical segment medsam ct.png --box 10,20,100,200 --out /tmp` |
| PatchCore | `anomalib-patchcore` | optional_extra | `[anomaly]` | `visionservex anomaly train patchcore --data /data/normal --out /tmp` |
| RTMDet-R | `rtmdet-r2-s` | expert_sidecar | OpenMMLab | `visionservex aerial detect aerial.jpg --model rtmdet-r2-s` |
| ByteTrack | `bytetrack` | real_smoke_verified | `pip install bytetracker` | `visionservex video-search tracker-smoke --tracker bytetrack` |
| OC-SORT | `ocsort` | real_smoke_verified | `pip install ocsort` | `visionservex video-search tracker-smoke --tracker ocsort` |
| RTMPose-m | `rtmpose-m` | real_smoke_verified | conda Python 3.10 sidecar | `bash scripts/run_openmmlab_rtmpose_smoke.sh` |
| RTMDet-tiny | `rtmdet-tiny-coco` | real_smoke_verified | conda Python 3.10 sidecar | `visionservex openmmlab smoke-test rtmdet-tiny-coco --device cpu` |
| Torchreid / OSNet | `osnet` | optional_extra | `pip install torchreid` | `visionservex video-search reid-smoke --reid osnet --image crop.jpg` |
| MaskDINO | `maskdino-swinl-coco` | expert_sidecar | Detectron2 sidecar | `visionservex maskdino create-env` |
| DEIMv2 | `deimv2-s/m/l/x` | unavailable | — | Blocked: native loader / no HF Transformers support |
| FastSAM | `fastsam-s/x` | do_not_add | — | AGPL-3.0 license; use SAM v1/2 instead |
| DeepSORT | — | do_not_add | — | GPL-3.0; not routed through permissive core |
| RF-DETR Plus/XL/2XL | — | non_core_license_optional | `pip install rfdetr[plus]` | PML 1.0 license — manual install only |
| SAM 3 / SAM 3.1 | `sam3.1` | external_api / gated | HF auth | `visionservex sam-family login-help sam3.1` |

### Status Legend

| Status | Meaning |
|--------|---------|
| `runnable` | Works now with the listed install command |
| `real_smoke_verified` | Runnable + smoke-tested on real hardware |
| `optional_extra` | Needs an extra pip package; clean install path exists |
| `expert_sidecar` | Needs isolated env (OpenMMLab, Detectron2, etc.) |
| `external_api` | API-gated; not self-hostable |
| `gated` | License/auth required; not auto-pulled |
| `non_core_license_optional` | Permissive core excludes it; manual opt-in only |
| `do_not_add` | Excluded (GPL/AGPL or non-commercial) |
| `unavailable_with_reason` | Blocked by a known technical/source issue |

See [docs/license_risk_table.md](docs/license_risk_table.md) for the
authoritative license-tier map.

### What works today (runnable models)

**Detection:** D-FINE (n/s/m/l/x), RF-DETR (nano/small/medium/large), Grounding DINO (tiny, swin-b)

**Segmentation:** RF-DETR-Seg (nano/small/medium), SAM v1 (vit-base), SAM 2 (hiera-tiny/small/base-plus/large), SAM 2.1 (tiny/small/base-plus/large), MedSAM

**Classification:** SwinV2 (tiny/small/base/large), ConvNeXtV2 (tiny/base/large), MaxViT (tiny)

**Open-vocab / VLM:** Florence-2 (base, large), OWLv2 (base, large), OWL-ViT (base, large), Grounding DINO (tiny, swin-b)

**Embedding:** DINOv2 (small/base/large/giant), CLIP (base, large), SigLIP (base), SigLIP2 (base)

**Experimental SOTA (not runnable yet):**

| Family | Models | Blocker |
|--------|--------|---------|
| DEIMv2 | `deimv2-s/m/l/x` | No HF Transformers support; custom loader required |
| RT-DETRv4 | `rtdetrv4-s` | No official checkpoint URLs |
| MaskDINO | `maskdino-swinl-coco` | Detectron2 environment required |

---

## Competitiveness Benchmark

```bash
# Synthetic mode (latency + detection health, no ground truth needed)
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --max-images 20 --device auto

# Real AP mode (AP50/mAP50:95 with YOLO-format annotated dataset)
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small,ultralytics:yolo11n \
  --dataset yolo:/path/to/coco128 \
  --max-images 100 \
  --out reports/ap_benchmark

# COCO JSON format
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --dataset coco-json:/data/coco/images:/data/coco/annotations/instances_val2017.json \
  --max-images 500
```

**Real AP/mAP** is computed with COCO-style 101-point interpolated PR curves when `--dataset` is provided. Results are exported as JSON + CSV. The tool is honest — if YOLO wins, it will say so.

**Note:** Accuracy-grade models are separate from demo models. Do not judge VisionServeX by `dfine-n` or `rfdetr-nano` — use `dfine-s-o365-coco` or `rfdetr-small` for AP comparison.

Detection, segmentation, classification, pose, OBB, and open-vocabulary tasks need different metrics.

**Task-specific benchmark commands:**

```bash
# Classification benchmark (top-k accuracy, per-class, latency)
visionservex benchmark-classification \
  --dataset folder:/path/to/dataset \
  --models convnextv2-tiny,swinv2-base,maxvit-tiny-tf-224 \
  --top-k 5 --max-images 100 --out /tmp/cls_bench.json

# Anomaly detection benchmark (PatchCore; requires [anomaly])
visionservex benchmark-anomaly \
  --dataset mvtec:/path/to/mvtec_like \
  --model patchcore --max-images 50 --out /tmp/anom_bench.json

# Surveillance-search retrieval benchmark (MAP@k, cosine similarity)
visionservex benchmark-surveillance-search \
  --index /path/to/index \
  --queries /path/to/queries.json \
  --top-k 5 --out /tmp/surv_bench.json
```

If `anomalib` is not installed, `benchmark-anomaly` returns `ANOMALIB_REQUIRED` with the exact install command rather than crashing.

---

## Capabilities Report

```bash
# What can VisionServeX do on this machine right now?
visionservex capabilities report
visionservex capabilities report --format markdown --out docs/capabilities.md
visionservex capabilities report --json
```

Covers: devices, installed extras, model counts by task/category, runnable models, unavailable blockers, goal-based recommendations, security status, and known limitations.

---

## Model Cards

```bash
# Structured per-model documentation
visionservex model-card show dfine-s-o365-coco
visionservex model-card show dfine-s-o365-coco --format markdown
visionservex model-card list --task detect
visionservex model-card export --out docs/model_cards.md
```

Every card includes: recommended_for, not_recommended_for, competes_with, hardware requirements, official benchmark note, and VisionServeX benchmark status.

---

## Replacement Map

```bash
# Which VisionServeX models replace each Ultralytics/YOLO task?
visionservex replacement-map map --task detect
visionservex replacement-map map --task segment
visionservex replacement-map map --task classify
visionservex replacement-map map --task pose
visionservex replacement-map map --format markdown
```

Honest and task-specific. Does not claim "better" unless AP evidence exists.

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

## Classification, Embedding & Open-Vocabulary Detection

```bash
# Classification (SwinV2 — real-smoke verified from local cache)
visionservex classify swinv2-tiny image.jpg --top-k 5

# Embeddings (DINOv2 — real-smoke verified; SigLIP2 — self-similarity verified)
visionservex embed dinov2-base image.jpg --out /tmp/dinov2.npy
visionservex similarity siglip2-base-patch16-224 a.jpg b.jpg

# Florence-2 (requires isolated env — REAL SMOKE PASSED: transformers==4.46.3 + einops + timm)
visionservex florence2 create-env --name vsx-florence --python 3.11  # generates validated recipe
visionservex florence2 doctor        # check compatibility
visionservex florence2 smoke-test florence-2-base image.jpg --task caption
```

Florence-2 real smoke result: `"a red truck with a light on top of it"` (street.jpg, transformers==4.46.3, CPU).

## Open-Vocabulary Detection & Multi-Task VLM

```python
# OWLv2 — zero-shot detection with free-form text queries
from visionservex import VisionModel
model = VisionModel("owlv2-base-patch16")
result = model.predict("image.jpg", prompt="person, red shirt, car")
result.to_json()

# Florence-2 — captioning, detection, OCR, phrase grounding (one model, many tasks)
model = VisionModel("florence-2-base")
model.predict("image.jpg", task="caption")
model.predict("image.jpg", task="object_detection")
model.predict("image.jpg", task="phrase_grounding", prompt="person wearing red shirt")
model.predict("image.jpg", task="ocr")
```

```bash
visionservex model pull owlv2-base-patch16
visionservex open-vocab owlv2-base-patch16 image.jpg --prompt "person, car"

visionservex model pull florence-2-base
visionservex predict florence-2-base image.jpg --task caption
```

## Surveillance Video-Search (local-only)

Index a folder of frames (or a video file) with a detector + tracker + embedder, then search by free-form text. **Appearance-based retrieval only — no face recognition, no biometric identity.**

```bash
# Check tracker and ReID backend availability
visionservex video-search trackers          # lists simple-iou (built-in), bytetrack, bot-sort, ocsort
visionservex video-search reid-models       # lists cosine-siglip2 (built-in), osnet, fastreid
visionservex video-search doctor --tracker bytetrack  # BYTETRACK_REQUIRED + exact install
visionservex video-search doctor --reid osnet         # TORCHREID_REQUIRED + exact install

visionservex video-search index ./frames/ \
  --detector owlv2-base-patch16 \
  --embedder siglip2-base-patch16-224 \
  --prompt "person" \
  --sample-fps 1 \
  --out indexes/camera01

visionservex video-search query indexes/camera01 \
  --text "person wearing a red shirt" \
  --top-k 20 \
  --out reports/red_shirt.html

visionservex video-search inspect indexes/camera01
visionservex video-search cleanup indexes/camera01 --yes
```

## Industrial Anomaly Detection

```bash
pip install 'visionservex[anomaly]'        # pulls anomalib
visionservex anomaly list
visionservex anomaly doctor
visionservex anomaly train patchcore --data normal_images/ --out runs/patchcore --dry-run
visionservex anomaly predict runs/patchcore test.jpg
```

PatchCore / PaDiM / FastFlow / EfficientAD / WinCLIP / DRAEM / Reverse-Distillation supported; missing dep returns `ANOMALIB_REQUIRED` with the exact install command and a fallback tip: use `--model mock-anomaly` to benchmark dataset statistics without anomalib.

## Medical Imaging (research only)

```bash
visionservex medical list
visionservex medical validate totalsegmentator
visionservex medical recommend --goal ct-segmentation

# MedSAM — real mask output (produces mask_000.png + medsam_metadata.json)
visionservex medical segment medsam image.png --box 10,20,200,200 --out output/
```

No diagnostic claims. Optional extras: `pip install 'visionservex[medical]'` for nibabel/NIfTI I/O.

MedSAM produces binary mask PNGs with IoU scores via SAM HF engine (`wanglab/medsam-vit-base`). Returns `CHECKPOINT_REQUIRED` if model not cached.

## Agriculture and Aerial Domain Commands

```bash
# Agriculture
visionservex agriculture recommend --goal weed-detection
visionservex agriculture prompt-detect field.jpg --prompt "weed" --detector owlv2-base-patch16
visionservex agriculture export-training-template --model rfdetr-small --out data_template/

# Aerial / drone
visionservex aerial recommend --goal oriented-detection
visionservex aerial dataset validate-dota --path /path/to/DOTA
visionservex aerial dataset validate-visdrone --path /path/to/VisDrone
```

OBB models (RTMDet-R/R2 via MMRotate) report **rotated IoU mAP50**, not axis-aligned AP. Do not compare them using box detection metrics.

---

## Gated Models & Expert Sidecars

```bash
# SAM3 / SAM3.1 — gated, auth-aware status check
visionservex sam3 status --model sam3.1-base-plus
visionservex sam3 login-help

# Heavy frameworks (OpenMMLab, Detectron2, MaskDINO, Co-DETR) — conda env recipe
visionservex expert list
visionservex openmmlab create-env --name visionservex-openmmlab --python 3.10  # conda recipe
visionservex openmmlab install-help    # native/conda/docker install options
visionservex openmmlab doctor          # check which packages are installed
visionservex openmmlab validate rtmpose-s  # OPENMMLAB_REQUIRED if deps missing
```

VisionServeX never auto-installs expert frameworks. `openmmlab create-env` generates the exact conda recipe; `openmmlab validate` returns structured errors with exact checkpoint/config expectations.

---

## Resource Safety & Developer Commands

VisionServeX includes a resource guard that prevents RAM/VRAM/disk exhaustion during testing and development. **Production CLI commands (`predict`, `embed`, `similarity`) are unaffected** — the guard only runs when you explicitly invoke a `dev` subcommand or pytest.

```bash
# Show current resources (RAM, VRAM, CPU, disk, processes)
visionservex dev resources

# Run quick tests (no real model, no GPU, no download) — < 60 s
visionservex dev test quick

# Targeted test on a single file
visionservex dev test targeted tests/test_my_feature.py

# Real model smoke tests (opt-in, uses smallest models, resource-checked)
visionservex dev test real-smoke

# GPU smoke tests (opt-in, VRAM-checked first)
visionservex dev test gpu-smoke --allow-gpu

# Benchmark smoke (process-isolated, max 3 images)
visionservex dev test benchmark-smoke

# Kill stray pytest processes (repo-scoped only)
visionservex dev kill-tests

# Clean test artifacts
visionservex dev clean-temp
visionservex dev clean-reports

# Model health: which models can run, checkpoint status, smoke results
visionservex models health --runnable-only
```

A pytest lockfile at `/tmp/visionservex_pytest.lock` prevents concurrent test runs. Default budgets: 8 GB free RAM, 2 GB free VRAM, 10 GB free disk. See [AGENT_RULES.md](AGENT_RULES.md) and [docs/agent_safety.md](docs/agent_safety.md).

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

## Feature Intelligence (DINOv2)

```bash
# Single-image embedding
visionservex embed dinov2-base image.jpg --out embedding.npy

# Folder embedding
visionservex embed dinov2-base folder/ --out embeddings_dir/

# Pairwise similarity
visionservex similarity dinov2-base a.jpg b.jpg

# Build search index + query
visionservex index dinov2-base folder/ --out indexes/dinov2_base
visionservex search dinov2-base query.jpg --index indexes/dinov2_base --top-k 10

# Deduplication
visionservex deduplicate dinov2-base folder/ --threshold 0.98 --out dups.csv

# Dataset intelligence
visionservex dataset-report dinov2-base folder/ --out report.md
visionservex active-select dinov2-base folder/ --budget 100 --out selected.csv
visionservex domain-shift dinov2-base train/ test/

# kNN benchmark (with labels.csv)
visionservex benchmark-embeddings --model dinov2-base --dataset folder:test_set/
```

Powered by `facebook/dinov2-{small,base,large,giant}` and `google/siglip2-base-patch16-224`. L2-normalized embeddings. **Do not** mix with detection AP — embeddings serve retrieval, deduplication, dataset audits, active learning.

---

## Specialized Model Zoo

```bash
# Source-grounded manifest (every model cites its upstream)
visionservex model-zoo sources
visionservex model-zoo show dinov2-base
visionservex model-zoo verify-links
visionservex model-zoo export --format markdown --out docs/model_zoo_manifest.md

# Domain recommendations
visionservex domain-zoo list
visionservex domain-zoo yolo26-competitors
visionservex domain-zoo sam-family
visionservex domain-zoo feature-intelligence
visionservex domain-zoo surveillance
visionservex domain-zoo medical
visionservex domain-zoo industrial

# Goal-driven recipe
visionservex domain-zoo recommend --domain surveillance --goal "red shirt person search"
```

Every model entry carries `recommended_action`: `add_now`, `expert_sidecar`, `external_api`, `non_core_license_optional`, `audit_only`, or `do_not_add` (e.g. YOLO-World — GPL/AGPL).

---

## VRAM Lifecycle Safety

VisionServeX manages GPU memory to prevent stepwise VRAM accumulation during repeated model loads.

```python
# Context manager — GPU cleanup on exit
with VisionModel("dfine-x-o365-coco", device="cuda") as model:
    result = model.predict("image.jpg")
# GPU memory flushed automatically after context exit

# Explicit cleanup
model = VisionModel("rfdetr-large", device="cuda")
result = model.predict("image.jpg")
model.unload()  # full cleanup: engine.unload + GC + CUDA empty_cache + ipc_collect

# One-shot predict with immediate unload
result = model.predict("image.jpg", unload_after=True)
```

```bash
# VRAM diagnostics
visionservex gpu explain-memory      # allocated vs reserved breakdown
visionservex gpu cleanup-cache       # flush CUDA allocator cache
visionservex gpu memory-test dfine-s-o365-coco --runs 5   # check VRAM growth
visionservex gpu memory-test-suite --models dfine-s-o365-coco,rfdetr-small

# Process-isolated benchmark (full CUDA context released after each model)
visionservex benchmark benchmark-competitiveness \
  --models dfine-x-o365-coco,rfdetr-large \
  --dataset yolo:/path/to/coco128 \
  --isolate-process \
  --out reports/ap_benchmark
```

---

## Segmentation Evaluation

```bash
# Latency-only (no ground truth needed)
visionservex benchmark benchmark-segmentation \
  --models rfdetr-seg-medium --max-images 20

# Real mask AP with COCO JSON annotations
visionservex benchmark benchmark-segmentation \
  --models rfdetr-seg-medium \
  --dataset coco-json:/data/coco/images:/data/coco/annotations/instances_val2017.json \
  --max-images 200 --out reports/seg_ap
```

**Note:** Mask AP uses binary mask IoU — NOT the same as detection box AP50. Do not mix these metrics.

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
- **benchmark-anomaly**: Functional — `--model mock-anomaly` computes pixel-stats proxy scores without anomalib. PatchCore training requires `pip install 'visionservex[anomaly]'`; version-dispatch adapter handles anomalib 1.x/2.x API differences.
- **Florence-2**: Real smoke **PASSED** in isolated env (transformers==4.46.3 + einops + timm). Use `visionservex florence2 create-env` for the exact validated recipe. The current environment (transformers 5.x) is incompatible.
- **MedSAM**: `medical segment medsam image.png --box 10,20,100,200 --out /tmp/out` now produces `mask_000.png` + `medsam_metadata.json` (real SAM HF engine). No longer delegates.
- **ByteTrack**: `video-search index --tracker bytetrack` is now a real selectable option. Returns `BYTETRACK_REQUIRED` if package missing, uses `_ByteTrackAdapter` if installed.
- **SAM2.1**: Registry wired (`facebook/sam2.1-hiera-*`); inference requires `[hf]` and a GPU. MobileSAM/EfficientSAM/HQ-SAM/EdgeSAM: Apache-2.0 expert sidecars. FastSAM: excluded (AGPL-3.0).
- **OpenMMLab** (RTMPose, RTMDet-R/R2, Co-DINO, InternImage): Use `visionservex openmmlab create-env` for the conda recipe. Returns `OPENMMLAB_REQUIRED` if deps missing; `CHECKPOINT_REQUIRED` if checkpoint missing.
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
| [Model cards](docs/model_cards.md) | Structured per-model cards with honest benchmark notes |
| [Replacement map](docs/replacement_map.md) | Ultralytics/YOLO → VisionServeX replacement guide |
| [Benchmark competitiveness](docs/benchmark_competitiveness.md) | AP/mAP evaluation guide |
| [Evaluation metrics](docs/evaluation_metrics.md) | AP50, mAP50:95, and metric definitions |
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

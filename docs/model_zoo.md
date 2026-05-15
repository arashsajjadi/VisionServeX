# Model zoo

The bundled registry (`visionservex list-models`) is always authoritative. This page is a curated, human-readable overview. Run `visionservex list-models --json` for machine-readable data.

## Status legend

| Status | Meaning |
|--------|---------|
| `stable` | CI-verified, full CLI/Python/API integration, regression tests pass |
| `beta` | Real backend wired; CLI/Python/gateway tested; CUDA and/or CPU verified; may have edge cases |
| `experimental` | Backend partially wired or sidecar-only; expect rough edges |
| `manual` | Backend not wired; user must install toolchain and download checkpoint manually |
| `external` | Upstream is API-gated; not self-hostable without an API token |

**`wired`** (Impl column) — this build runs the real model when its dependencies are present.  
**`partial`** — some code path is wired; expect rough edges.  
**`stub`** — registry entry exists but no real inference; raises a structured error.

---

## Auto-downloadable models (recommended first)

| Model ID | Task | Difficulty | VRAM (rec) | Status | Impl |
|----------|------|-----------|-----------|--------|------|
| `mock-detect` | detect | very_easy | n/a | stable | wired |
| `mock-classify` | classify | very_easy | n/a | stable | wired |
| `mock-segment` | segment | very_easy | n/a | stable | wired |
| `mock-grounded-segment` | grounded_segment | very_easy | n/a | stable | wired |
| `mock-foundation-segment` | foundation_segment | very_easy | n/a | stable | wired |
| `mock-pose` | pose | very_easy | n/a | stable | wired |
| `mock-obb` | obb | very_easy | n/a | stable | wired |
| `mock-open-vocab` | open_vocab_detect | very_easy | n/a | stable | wired |
| `dfine-n` | detect | very_easy | 1 GB | beta | wired |
| `dfine-s` | detect | easy | 2 GB | beta | wired |
| `dfine-m` | detect | medium | 4 GB | beta | wired |
| `swinv2-tiny` | classify | very_easy | 1 GB | beta | wired |
| `swinv2-small` | classify | easy | 2 GB | beta | wired |
| `rfdetr-nano` | detect | very_easy | 1.5 GB | beta | wired |
| `rfdetr-small` | detect | easy | 3 GB | beta | wired |
| `rfdetr-base` | detect | easy | 4 GB | beta | wired |
| `rfdetr-medium` | detect | medium | 6 GB | beta | wired |
| `rfdetr-large` | detect | medium | 12 GB | beta | wired |
| `rfdetr-seg-nano` | segment | easy | 2 GB | beta | wired |
| `rfdetr-seg-small` | segment | easy | 3 GB | beta | wired |
| `rfdetr-seg-medium` | segment | medium | 8 GB | beta | wired |
| `grounding-dino-tiny` | open-vocab detect | easy | 4 GB | beta | wired |
| `grounding-dino-swin-t` | open-vocab detect | easy | 6 GB | beta | wired |
| `grounding-dino-swin-b` | open-vocab detect | medium | 12 GB | beta | wired |
| `sam-vit-base` | foundation segment | easy | 2 GB | beta | wired |
| `sam-vit-large` | foundation segment | medium | 6 GB | beta | wired |
| `sam2-hiera-tiny` | foundation segment | medium | 4 GB | beta | wired |
| `sam2-hiera-small` | foundation segment | medium | 6 GB | beta | wired |
| `grounded-sam` | grounded segment | medium | 6 GB | beta | wired |
| `grounded-sam2` | grounded segment | medium | 8 GB | beta | wired |

---

## Requires manual download (not auto-downloadable)

| Model ID | Task | VRAM (rec) | Status | Impl | Notes |
|----------|------|-----------|--------|------|-------|
| `dfine-l` | detect | 8 GB | beta | wired | HF download |
| `dfine-x` | detect | 12 GB | beta | wired | HF download |
| `swinv2-base` | classify | 3 GB | beta | wired | HF download |
| `swinv2-large` | classify | 6 GB | beta | wired | HF download |
| `sam-vit-huge` | foundation segment | 12 GB | beta | wired | HF download |
| `sam2-hiera-base-plus` | foundation segment | 8 GB | beta | wired | HF download |
| `sam2-hiera-large` | foundation segment | 12 GB | beta | wired | HF download |
| `oneformer-swin-large` | semantic/panoptic | 12 GB | beta | wired | HF download |
| `oneformer-dinat-large` | semantic/panoptic | 14 GB | beta | wired | HF download |
| `oneformer-convnext-large` | semantic/panoptic | 12 GB | experimental | wired | HF download |

---

## OpenMMLab — docker/manual only

These models require the OpenMMLab toolchain. They are **not part of the stable v1.0.0 core**.
Use `visionservex openmmlab pull <model_id>` for instructions.

| Model ID | Task | Status | Impl | Path |
|----------|------|--------|------|------|
| `rtmpose-t/s/m/l` | pose | manual / experimental | stub / partial | Docker or native mmpose |
| `rtmdet-r-t/s/m/l` | OBB | manual | stub | Docker or native mmrotate |
| `rtmdet-r2-t/s/m/l` | OBB | manual / experimental | stub / partial | Docker or native mmrotate |
| `co-dino-inst-vit-l-*` | instance segment | manual | stub | Docker + heavy VRAM |
| `internimage-t/s/b/l/h` | classify | manual | stub | Docker + custom CUDA ops |
| `seem-focal-t`, `seem-davit-d3` | grounded segment | manual | stub | Docker only |

---

## External / API-gated

| Model ID | Task | Notes |
|----------|------|-------|
| `grounding-dino-1.5` | open-vocab detect | IDEA-Research API; weights not openly downloadable |
| `grounding-dino-1.6` | open-vocab detect | IDEA-Research API; weights not openly downloadable |

---

## RF-DETR-Seg large variants (experimental stubs)

| Model ID | VRAM | Notes |
|----------|------|-------|
| `rfdetr-seg-large` | 10 GB | Not yet wired; use rfdetr-seg-nano/small/medium |
| `rfdetr-seg-xlarge` | 16 GB | Not yet wired |
| `rfdetr-seg-2xlarge` | 24 GB | Not yet wired, expert hardware required |

---

## Which model should I start with?

- **Detection (CPU)**: `dfine-n` — fast, HF auto-download, CPU-capable.
- **Detection (GPU)**: `rfdetr-nano` or `dfine-n`.
- **Open-vocab detection**: `grounding-dino-tiny` — text prompt, auto-download.
- **Classification**: `swinv2-tiny` — lightweight, CPU-capable.
- **Foundation segmentation**: `sam2-hiera-tiny` — HF auto-download.
- **Prompt-driven segmentation**: `grounded-sam2` — Grounding DINO + SAM 2.
- **Semantic/panoptic**: `oneformer-swin-large` — manual HF download needed.
- **Pose (expert)**: `rtmpose-s` — requires OpenMMLab toolchain.
- **OBB (expert)**: `rtmdet-r2-s` — requires OpenMMLab toolchain.
- **No GPU, no extras installed**: `mock-*` — built-in, always works.

## License-uncertain models

The following entries set `license_uncertain=true`. Verify upstream before commercial use:

- `co-dino-inst-vit-l-coco`, `co-dino-inst-vit-l-lvis`
- `rtmdet-r-*`, `rtmdet-r2-*`
- `internimage-*`
- `seem-*`
- `grounding-dino-1.5`, `grounding-dino-1.6` (also `external`)

See [`model_licenses.md`](model_licenses.md) for the per-model table.

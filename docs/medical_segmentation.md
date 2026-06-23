<!-- SPDX-License-Identifier: Apache-2.0 -->
# Medical Segmentation in VisionServeX

> **NOT A MEDICAL DEVICE.** VisionServeX medical models are for **research and
> education only**. They make **no diagnostic, treatment, or clinical claim**. Do
> not use any output for patient care. VisionServeX **never ships model weights**
> and **never auto-installs** medical dependencies.

This page is the single source of truth for what is actually runnable today, what
is a research-only sidecar, and what each model's license really allows.

## TL;DR

| You want… | Use | Commercial-safe? |
|---|---|---|
| Promptable mask on a 2D medical image (box/point) | **`medsam`** (wired) | **No** — research-only weights |
| General-purpose promptable seg you *can* use commercially on medical images (annotation assist, non-diagnostic) | **`sam2.1-hiera-*`** / `sam-vit-*` (Apache-2.0) | **Yes** (general model, with non-diagnostic disclaimer) |
| 3D volume / video promptable medical seg | **`medsam2`** — research-only **expert sidecar**, not runnable in core | **No** — non-commercial weights |
| CT/MR multi-organ, nnU-Net, MONAI | External sidecars (run upstream) | Varies — see table |

There is currently **no commercial-safe, medical-specialized** segmentation model
in VisionServeX. The only commercial-safe promptable segmenters are the **general**
SAM / SAM2 / SAM2.1 family.

## Supported models

| Model ID | Upstream | Runtime status | Task | Input | Output | Fine-tune |
|---|---|---|---|---|---|---|
| `medsam` | [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) | **wired** (engine `sam_hf`) | 2D promptable seg | RGB image (PNG/JPEG) | mask (PNG + COCO RLE + box) | `NOT_TRAINABLE_BY_DESIGN` (inference only) |
| `medsam2` | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2) | **research-only sidecar** (engine `medsam2_sidecar`, dependency-gated, not runnable in core) | (3D/video promptable — claimed upstream) | — | — | `EXTERNAL_TRAINING_ONLY` |
| `totalsegmentator` | [wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | external sidecar (run upstream) | organ seg (CT/MR) | NIfTI | volume label map | external |
| `totalsegmentator-tissue` | same | external sidecar, **license-key gated** | tissue/body-stats | NIfTI | label map | external |
| `nnunet-v2` | [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) | external framework | 3D seg | NIfTI | label map | external (your data) |
| `monai-bundles` / `auto3dseg` | [Project-MONAI](https://github.com/Project-MONAI/MONAI) | external framework | various | NIfTI/DICOM | bundle-defined | external |

## Commercial-safety (authoritative: `visionservex.licensing.policy`)

| Model | Code license | Weights license | Policy status | Commercial-safe default? |
|---|---|---|---|---|
| `medsam` | Apache-2.0 | Apache-2.0 weights / **medical provenance** | `legal_review_required` | **No** |
| `medsam2` | Apache-2.0 | **non-commercial** (medical dataset provenance) | `noncommercial_restricted` | **No** |
| `totalsegmentator` (core) | Apache-2.0 | Apache-2.0 (core total) | optional extra | Re-verify before public install path |
| `totalsegmentator-tissue` | Apache-2.0 | **proprietary, key-gated** | `non_core_license_optional` | **No** |
| `sam2.1-hiera-*`, `sam-vit-*` | Apache-2.0 | Apache-2.0 | `commercial_safe_core` | **Yes** (general, non-diagnostic) |

The license firewall is enforced by tests
(`test_v323_medical_segmentation_pack.py`, `test_v310_sidecar_triage.py`): the
model-zoo manifest and gap report can never present a non-commercial / legal-review
medical model as a bare permissive license.

## MedSAM v1 — exact syntax

**Python**
```python
from visionservex import VisionModel

m = VisionModel("medsam")                       # requires: pip install 'visionservex[hf]'
result = m.predict("scan.png", boxes=[[10, 20, 100, 200]])
for seg in result.segments:                      # one segment per box prompt
    mask = seg.mask                               # HxW uint8 numpy array (0/1)
    print(seg.score, seg.box)
```

Multi-box (each box yields its own mask — fixed in v3.23):
```python
result = m.predict("scan.png", boxes=[[10, 20, 100, 200], [120, 40, 200, 180]])
assert len(result.segments) == 2
```

Point prompts:
```python
result = m.predict("scan.png", points=[[64, 64]], point_labels=[1])
```

**CLI**
```bash
visionservex medical segment medsam scan.png --box 10,20,100,200 --out out/ --json
# multi-box: repeat --box
visionservex medical segment medsam scan.png --box 10,20,100,200 --box 120,40,200,180 --out out/
# generic predict path also works:
visionservex predict medsam scan.png --box 10,20,100,200 --json
```

Outputs in `out/`: `mask_000.png`, … and `medsam_metadata.json`
(`{model_id, masks_saved[], n_masks, device, status}`).

**HTTP**
```bash
curl -X POST localhost:8000/segment/b64 -H 'content-type: application/json' -d '{
  "model_id": "medsam",
  "image_b64": "<base64 PNG>",
  "options": {"boxes": [[10,20,100,200]], "points": [[64,64]], "point_labels": [1]}
}'
```
Response is a `PredictionResponse`; each segment carries a COCO **RLE** mask
(`results[].rle`), `box`, `mask_shape`, `mask_pixels_on`.

**Batch / parallel** — MedSAM v1 uses the standard VisionServeX batch path
(`model.batch_predict([img1, img2, ...])`), which preserves input order and
returns one `BaseResult` per input (each result's `metadata['batch_mode']` reports
whether a true tensor batch or an honest loop was used). Keep GPU workers
conservative (one model per device); do not fan out duplicate large models per
thread.

## MedSAM2 — research-only expert sidecar (not runnable in core)

MedSAM2 is **not** a VisionServeX runtime model and is **not** reachable via
`VisionModel("medsam2")`, `visionservex predict medsam2`, or HTTP (`/segment/b64`
returns a clean `404 MODEL_NOT_FOUND`). This is deliberate:

* **Non-commercial weights.** The published checkpoints carry medical-dataset
  provenance restrictions (`noncommercial_restricted`). They must never be
  labelled commercial-safe or offered as a default.
* **Non-HF checkpoints.** Upstream ships raw SAM 2 `.pt` files, not HF
  `transformers` format, so `sam_hf`/`sam2_hf` cannot load them — a native
  `build_sam2` predictor from the upstream repo is required.

What VisionServeX provides is an **honest, dependency-gated sidecar skeleton**
(`engines/medsam2_sidecar.py`): it fails cleanly with a structured error and an
actionable install hint, and it **never** returns mock output as if it were real.

```bash
# Probe / install guidance (no download, no crash):
visionservex medical validate medsam2 --json
visionservex medical install-help medsam2 --json
# Research-only setup, isolated env:
git clone https://github.com/bowang-lab/MedSAM2 && cd MedSAM2 && pip install -e .
# then obtain a MedSAM2 checkpoint from upstream (non-commercial).
pip install 'visionservex[medsam2]'   # marker extra; never auto-installed
```

## Fine-tuning status (no false claims)

| Model | Status |
|---|---|
| `medsam` | `NOT_TRAINABLE_BY_DESIGN` — foundation segmenter, inference only |
| `medsam2` | `EXTERNAL_TRAINING_ONLY` — train via upstream repo, not wrapped in VSX |
| `nnunet-v2`, `monai`, `auto3dseg` | `EXTERNAL_TRAINING_ONLY` — frameworks; train on your own data upstream |

VisionServeX does not implement or claim medical fine-tuning. SAM-family decoder
fine-tuning, where offered elsewhere in the package, is a frozen-encoder /
decoder-only path and is never advertised as full training.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MEDICAL_EXTRA_REQUIRED` / missing `transformers` | `[hf]` not installed | `pip install 'visionservex[hf]'` |
| `CHECKPOINT_REQUIRED` (medsam) | weights not cached | `visionservex model pull medsam` |
| `MEDSAM2_REQUIRED` / `MEDSAM2_CHECKPOINT_UNVERIFIED` | MedSAM2 sidecar deps/checkpoint absent | research-only setup above; not runnable in core |
| `404 MODEL_NOT_FOUND` for `medsam2` over HTTP | MedSAM2 is not a runtime model | expected — use the CLI sidecar path |
| `NIFTI_IO_REQUIRED` | NIfTI input without `nibabel` | `pip install 'visionservex[medical]'` |
| CUDA unavailable | no GPU | MedSAM v1 runs on CPU (slower); set `--device cpu` |
| DICOM input | not supported in core | convert to NIfTI/PNG first |

## See also

- License policy: `src/visionservex/licensing/policy.py`
- Gap report: `docs/model_zoo_gap_report.md`
- Sidecar engine: `src/visionservex/engines/medsam2_sidecar.py`
- Tests: `tests/test_v323_medical_segmentation_pack.py`

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
| Real 2D MedSAM2 promptable seg (experimental) | **`medsam2`** via `visionservex medical medsam2 …` in an isolated env | **No** — research-only weights |
| 3D volume / video promptable medical seg | **`medsam2`** — NOT wired in VisionServeX (only `2d_slice`); use upstream | **No** — non-commercial weights |
| CT/MR multi-organ, nnU-Net, MONAI | External sidecars (run upstream) | Varies — see table |

There is currently **no commercial-safe, medical-specialized** segmentation model
in VisionServeX. The only commercial-safe promptable segmenters are the **general**
SAM / SAM2 / SAM2.1 family.

## Supported models

| Model ID | Upstream | Runtime status | Task | Input | Output | Fine-tune |
|---|---|---|---|---|---|---|
| `medsam` | [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) | **wired** (engine `sam_hf`) | 2D promptable seg | RGB image (PNG/JPEG) | mask (PNG + COCO RLE + box) | `NOT_TRAINABLE_BY_DESIGN` (inference only) |
| `medsam2` | [bowang-lab/MedSAM2](https://github.com/bowang-lab/MedSAM2) | **experimental 2D runtime** (`medical/medsam2_runtime.py`, isolated env only; not in registry) | 2D PNG/JPEG, NIfTI middle slice | mask (PNG + box + RLE) | `EXTERNAL_TRAINING_ONLY` |
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

## MedSAM2 — research-only runtime (experimental, isolated env)

MedSAM2 is **research/education only** (the HF model card states: *"The model
weights can only be used for research and education purposes"*). It is therefore
**never commercial-safe**, is **not** in the runtime registry, and is **not**
reachable via `VisionModel("medsam2")`, `visionservex predict medsam2`, or HTTP
(`/segment/b64` returns a clean `404 MODEL_NOT_FOUND`).

A **real 2D (slice/frame) runtime** is available through an experimental,
dependency-gated CLI — `visionservex medical medsam2 …` — backed by the in-process
adapter `visionservex.medical.medsam2_runtime`. It runs the genuine upstream
`build_sam2` + `SAM2ImagePredictor` and returns a real mask normalized to the
VisionServeX segmentation schema. **3D-volume and video are NOT wired** (only
`2d_slice`); unsupported inputs/modes raise structured errors, never fake output.

### Isolated setup (CPU; reproducible)

Upstream needs **Python 3.12 + torch 2.5.1**; do not install it into a normal
VisionServeX env. The host GPU here is Blackwell (sm_120), which torch cu124
wheels don't support, so CPU is the clean path:

```bash
bash scripts/medsam2/setup_isolated_env.sh
# creates conda env vsx-medsam2 (py3.12), installs torch 2.5.1 CPU + the MedSAM2
# fork (provides `sam2`), downloads MedSAM2_latest.pt (156 MB, research-only),
# and runs a 2D CPU smoke. No checkpoints are written into the repo.
```

### Exact syntax (run inside the isolated env)

```bash
visionservex medical medsam2 doctor --json
visionservex medical medsam2 load --checkpoint MedSAM2_latest.pt --device cpu --json
visionservex medical medsam2 segment slice.png --checkpoint MedSAM2_latest.pt \
    --box 64,64,192,192 --out out/ --json
visionservex medical medsam2 batch inputs.txt --checkpoint MedSAM2_latest.pt \
    --out out/ --workers 1 --json          # order-preserving; deterministic filenames
visionservex medical medsam2 benchmark-smoke --checkpoint MedSAM2_latest.pt --json
```

```python
from visionservex.medical.medsam2_runtime import load_medsam2_runtime, load_2d_input, segment_2d

rt = load_medsam2_runtime("MedSAM2_latest.pt", device="cpu")
img = load_2d_input("slice.png")
result = segment_2d(rt, img, boxes=[[64, 64, 192, 192]])   # real SegmentationResult
assert result.metadata["commercial_safe"] is False
```

**Structured error codes:** `MEDSAM2_REQUIRED`, `MEDSAM2_CHECKPOINT_REQUIRED`,
`MEDSAM2_CHECKPOINT_INVALID`, `MEDSAM2_CONFIG_REQUIRED`, `MEDSAM2_RUNTIME_UNAVAILABLE`,
`MEDSAM2_OOM`, `MEDSAM2_UNSUPPORTED_INPUT`, `MEDSAM2_LICENSE_RESTRICTED`.

**Batch / parallel:** one model is loaded once and reused; a shared SAM2 predictor
is not thread-safe, so a shared model runs sequentially (`effective_workers=1`) and
GPU never duplicates the model. Outputs are deterministic
(`{index:05d}_{stem}_medsam2_mask_{m:03d}.png` + a `medsam2_batch_manifest.json`)
and never overwritten without `--overwrite`.

## Medical training truth (no fake fine-tuning)

VisionServeX does **not** train or fine-tune any medical model in-process. The
machine-readable matrix lives in `visionservex.medical.training.TRAINING_MATRIX`;
`visionservex medical train …` exposes it and offers dataset validation + **exact
upstream command generation** (dry-run) only.

| Model | Status |
|---|---|
| `medsam` | `NOT_TRAINABLE_BY_DESIGN` — foundation segmenter, inference only |
| `medsam2` | `EXTERNAL_TRAINING_ONLY` — train via upstream repo; VSX offers dataset-validate + command dry-run |
| `nnunet-v2`, `monai`, `swinunetr`, `auto3dseg` | `TRAINING_FRAMEWORK_EXTERNAL` — run their own trainer; VSX dry-run generates the exact command |
| `sam2` (general) | `NOT_TRAINABLE_IN_VSX` |

```bash
visionservex medical train doctor --json
visionservex medical train validate-dataset DATASET_DIR --task segmentation --json   # images/ + masks/
visionservex medical train dry-run --framework nnunet --dataset DATASET_DIR --out OUT --json
```

No model is ever reported `trainable_in_vsx` / `finetunable_in_vsx`. Dataset
contract: `DATASET_DIR/images/*.png` + `DATASET_DIR/masks/*.png` paired by stem.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `MEDICAL_EXTRA_REQUIRED` / missing `transformers` | `[hf]` not installed | `pip install 'visionservex[hf]'` |
| `CHECKPOINT_REQUIRED` (medsam) | weights not cached | `visionservex model pull medsam` |
| `MEDSAM2_REQUIRED` | MedSAM2 runtime deps (`sam2`) absent | run in the isolated env (`scripts/medsam2/setup_isolated_env.sh`) |
| `MEDSAM2_CHECKPOINT_REQUIRED` / `MEDSAM2_CHECKPOINT_INVALID` | checkpoint missing or incompatible | download `MedSAM2_latest.pt` from HF `wanglab/MedSAM2` (research-only) |
| `MEDSAM2_UNSUPPORTED_INPUT` (DICOM / 3D / video) | unsupported input or mode | convert DICOM→PNG/NIfTI; only `2d_slice` is wired |
| `MEDSAM2_OOM` | out of memory | use CPU or a smaller image |
| `404 MODEL_NOT_FOUND` for `medsam2` over HTTP | MedSAM2 is not a runtime registry model | expected — use the `medical medsam2` CLI |
| `NIFTI_IO_REQUIRED` | NIfTI input without `nibabel` | `pip install 'visionservex[medical]'` |
| CUDA unavailable / Blackwell sm_120 | torch cu124 lacks sm_120 kernels | use `--device cpu` |

## What NOT to claim

- ❌ MedSAM2 is commercial-safe (weights are research/education only).
- ❌ MedSAM2 does diagnosis / has clinical accuracy.
- ❌ VisionServeX fine-tunes / trains MedSAM, MedSAM2, nnU-Net, or MONAI.
- ❌ MedSAM2 3D-volume or video is supported in VisionServeX (only `2d_slice`).
- ❌ MedSAM2 runs in the normal core env (it needs an isolated py3.12 + torch 2.5.1 env).

## See also

- License policy: `src/visionservex/licensing/policy.py`
- Runtime adapter: `src/visionservex/medical/medsam2_runtime.py`
- Isolated setup + smoke: `scripts/medsam2/setup_isolated_env.sh`, `scripts/medsam2/real_runtime_smoke.py`
- Tests: `tests/test_v324_medsam2_runtime.py`, `tests/test_v323_medical_segmentation_pack.py`

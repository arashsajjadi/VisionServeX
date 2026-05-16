# VisionServeX Optional Extras

Optional extras are pip installable groups that unlock additional models and capabilities. None of them are required to run the core set of models.

```bash
pip install 'visionservex[EXTRA]'
# Multiple extras:
pip install 'visionservex[server,hf,rfdetr]'
```

---

## `[hf]` — Hugging Face Transformers models

**Install:** `pip install 'visionservex[hf]'`

Unlocks all models loaded via `transformers` AutoModel classes:

- **Detection:** D-FINE (all sizes), Grounding DINO (tiny/swin-b)
- **Segmentation:** SAM v1, SAM 2, SAM 2.1, MedSAM, RF-DETR-Seg
- **Classification:** SwinV2 (all sizes), ConvNeXtV2 (all sizes), MaxViT
- **Embedding:** DINOv2 (all sizes), SigLIP, SigLIP2, CLIP, Florence-2
- **Open-vocab:** OWLv2, OWL-ViT v1, Florence-2
- **VLM:** Florence-2 (base, large)

**Required pip packages:** `transformers>=4.40`, `pillow`, `torch`

---

## `[rfdetr]` — RF-DETR detection and segmentation

**Install:** `pip install 'visionservex[rfdetr]'`

Unlocks RF-DETR models from Roboflow:

- **Detection:** `rfdetr-nano`, `rfdetr-small`, `rfdetr-medium`, `rfdetr-large`
- **Segmentation:** `rfdetr-seg-nano`, `rfdetr-seg-small`, `rfdetr-seg-medium`

**Required pip packages:** `rfdetr` (Roboflow package)

**License:** Apache-2.0 (all RF-DETR models)

---

## `[anomaly]` — Anomalib industrial anomaly detection

**Install:** `pip install 'visionservex[anomaly]'`

Or use the dedicated conda environment:
```bash
visionservex anomaly create-env --name vsx-anomaly --python 3.11
```

Unlocks:
- `visionservex anomaly train patchcore|padim|fastflow|efficientad|winclip|draem`
- `visionservex anomaly predict`
- `visionservex anomaly benchmark`

**Validated versions:** `anomalib>=1.0,<3.0`

**Required pip packages:** `anomalib`, `torch`, `torchvision`

**Use case:** Unsupervised industrial defect detection (MVTec AD, custom datasets)

---

## `[florence2]` — Florence-2 isolated environment

**Install:** `pip install 'visionservex[hf]'` (basic)

For conflict-free use, create an isolated env:
```bash
visionservex florence2 create-env --name vsx-florence2
```

Florence-2 optionally requires `flash-attn` for faster inference:
```bash
pip install flash-attn --no-build-isolation
```

**Unlocks:** `visionservex florence2 predict` with all task tokens

---

## `[medical]` — Medical imaging dependencies

NIfTI volume input (`.nii`, `.nii.gz`) requires `nibabel`:

**Install:** `pip install nibabel`

This is not a formal VisionServeX extra but a runtime dependency for NIfTI support in `visionservex medical segment`.

Full medical model dependencies:
- `medsam`: included in `[hf]` (uses `transformers`)
- `totalsegmentator`: `pip install TotalSegmentator` (brings nibabel, nnunetv2, SimpleITK)
- `nnunet-v2`: `pip install nnunetv2`
- `monai-bundles`: `pip install monai`

---

## `[server]` — HTTP gateway server

**Install:** `pip install 'visionservex[server]'`

Unlocks the local HTTP API gateway:

```bash
visionservex serve
# Listens at http://127.0.0.1:8080 by default

visionservex gateway start --host 0.0.0.0 --port 8080
```

**Required pip packages:** `fastapi`, `uvicorn[standard]`

---

## Future extras (planned)

### `[tracking]` — tracker backends

When `bytetrack`, `bot-sort`, and `ocsort` are packaged:
```bash
pip install 'visionservex[tracking]'  # planned
```

Currently: use `visionservex video-search install-help --tracker bytetrack` for manual install instructions.

### `[reid]` — ReID backends

When `torchreid` / `fastreid` are packaged:
```bash
pip install 'visionservex[reid]'  # planned
```

Currently: use `visionservex video-search install-help --reid osnet` for manual install instructions.

---

## Check what is installed

```bash
visionservex capabilities report
visionservex model-zoo verify-links
visionservex anomaly doctor
visionservex medical doctor
visionservex sam-family doctor
```

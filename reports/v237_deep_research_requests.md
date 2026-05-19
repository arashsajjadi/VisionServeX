# v2.37 Deep Research Requests

Models that still need external information before they can be fully resolved.
All have precise blocker codes; this document specifies the exact missing piece for each.

---

## Deep Research Request: oneformer-convnext-large

### Goal
Find whether SHI-Labs has published an official COCO-trained OneFormer-ConvNeXt-Large checkpoint, or confirm that no such checkpoint exists.

### Current blocker
`UPSTREAM_UNAVAILABLE` (registry mapping returns 404; only swin_large and dinat_large exist for COCO)

### What we already tried
- `visionservex predict oneformer-convnext-large IMAGE` → 404 at HF
- Searched https://huggingface.co/shi-labs — only `oneformer_coco_swin_large` and `oneformer_coco_dinat_large` are present
- Official repo https://github.com/SHI-Labs/OneFormer doesn't list a COCO ConvNeXt checkpoint

### Missing information needed
1. Whether ConvNeXt OneFormer for COCO is officially released
2. If yes, exact HF model ID
3. If no, should the registry row be removed?

### Required output
- official URL if it exists
- otherwise: confirmation that this row should be `wrong_registry_entry` and removed

### Priority
P2 (alternative OneFormer variants exist and SwinLarge already benchmarked at 0.1649)

---

## Deep Research Request: rtdetrv4-{s,m,l,x}

### Goal
Find the exact Google Drive IDs (or other download URLs) for RT-DETRv4 official checkpoints.

### Current blocker
`MANUAL_CHECKPOINT_REQUIRED` (Google Drive IDs not yet documented in our system)

### What we already tried
- Visited https://github.com/RT-DETRs/RT-DETRv4 — confirmed checkpoint distribution is via Google Drive
- No `--auto-pull` capability

### Missing information needed
1. Exact Google Drive file IDs for rtdetrv4-s, rtdetrv4-m, rtdetrv4-l, rtdetrv4-x
2. Expected SHA256 checksums (if officially published)
3. Required config file names for each variant

### Required output
For each of {s,m,l,x}:
- gdown ID
- expected weight filename
- expected config filename
- known accuracy on COCO (paper-reported)

### Priority
P1 — RT-DETR family is detection-relevant. Already trailing yolo26x, but worth checking.

---

## Deep Research Request: seem-davit-d3 and seem-focal-t

### Goal
Find the minimal Python/torch version compatibility matrix to run SEEM (Segment Everything Everywhere Model).

### Current blocker
`SIDECAR_REQUIRED` (X-Decoder/SEEM uses old dependency stack)

### What we already tried
- Identified official source: https://github.com/microsoft/x-decoder + https://github.com/MqLeet/SEEM
- Did not attempt sidecar creation (would require Python 3.9/3.10 + Detectron2 + custom packages)

### Missing information needed
1. Latest-supported Python version (3.10 or 3.11?)
2. Required torch version
3. Required Detectron2 version
4. Whether HuggingFace integration exists (eliminating sidecar need)
5. Exact pip install sequence

### Priority
P2 — SAM2/SAM2.1 already provide promptable segmentation at 0.8060 IoU

---

## Deep Research Request: maskdino-r50-coco, maskdino-r50-panoptic

### Goal
Find the official MaskDINO HuggingFace IDs or checkpoint URLs for COCO instance and panoptic.

### Current blocker
`SIDECAR_REQUIRED` (OpenMMLab/MaskDINO stack)

### What we already tried
- Identified MaskDINO official sources but did not attempt download
- Did not yet create OpenMMLab sidecars

### Missing information needed
1. Official MaskDINO checkpoint URLs for COCO instance + panoptic
2. Required mmcv/mmdet versions for MaskDINO
3. Whether MaskDINO has a maintained HF integration

### Priority
P0 — current best VisionServeX auto-segmentation is OneFormer-SwinLarge at 0.1649. MaskDINO is state-of-the-art and would close the gap to yolo26x-seg (0.2728).

---

## Deep Research Request: CO-DETR (co-dino-inst-vit-l-coco/lvis)

### Goal
Find the official zongzhuofan/co-detr-vit-large-coco or HF integration for CO-DETR ViT-L.

### Current blocker
`SIDECAR_REQUIRED`

### What we already tried
- Identified: https://github.com/sense-x/co-detr and https://huggingface.co/zongzhuofan/co-detr-vit-large-lvis
- Did not yet create OpenMMLab sidecar

### Missing information needed
1. Whether CO-DETR can run via HuggingFace Transformers (avoiding mmdet)
2. Required versions of mmcv/mmdet if sidecar needed
3. Exact COCO checkpoint URL (LVIS one is on HF but COCO variant unclear)

### Priority
P0 — CO-DETR achieves state-of-the-art detection (~67 mAP on COCO). Would significantly beat yolo26x (0.4894) and libreyolo-dfine-x (0.5030) if usable.

---

## Deep Research Request: RTMDet-R / RTMDet-R2 (OBB family)

### Goal
Find the easiest way to run RTMDet-Rotated for aerial OBB detection on a Python 3.11 sidecar.

### Current blocker
`SIDECAR_REQUIRED`

### What we already tried
- Identified MMRotate as the codebase, but not yet installed

### Missing information needed
1. mmrotate version compatible with Python 3.11 + torch 2.5+
2. RTMDet-R config filenames for each size variant
3. Whether DOTA-trained checkpoints are publicly distributable

### Priority
P2 — OBB detection is a domain-specific task; no current VisionServeX OBB benchmark exists.

---

## Deep Research Request: RTMPose

### Goal
Find the simplest way to run RTMPose for pose estimation via MMPose, or whether HF integration exists.

### Current blocker
`SIDECAR_REQUIRED`

### Missing information needed
1. mmpose version compatible with Python 3.11
2. RTMPose config filenames for s/m/l with 256x192 and 384x288 input sizes
3. Whether pose smoke asset (person_pose.jpg) exists in repo

### Priority
P2 — pose estimation is domain-specific; not in current benchmark suite

---

## Deep Research Request: InternImage classification

### Goal
Find the easiest way to load InternImage classification weights (OpenGVLab) — directly via timm/HF or only via mmpretrain?

### Current blocker
`SIDECAR_REQUIRED`

### What we already tried
- Identified HF org: https://huggingface.co/OpenGVLab
- Found: internimage_t_1k_224, internimage_s_1k_224, internimage_b_1k_224

### Missing information needed
1. Whether InternImage has HF Transformers integration
2. If only timm: which timm version exposes InternImage?
3. Custom CUDA ops requirement (deformable attention)

### Priority
P3 — classification is a baseline task; SwinV2 already provides classification

---

## Models NOT worth solving before v3

The following are documented and have known blockers but are low priority:
- `grounding-dino-1.5/1.6` — API-gated, only 2 models, requires external API key
- `sam3-base` — HF-gated, requires user license acceptance
- `seem-*` — superseded by SAM2/SAM2.1 promptable benchmark
- `internimage-*` — classification is already covered

## Models HIGH priority for v3

- MaskDINO (instance/panoptic) — could close auto-seg gap
- CO-DETR — could establish new detection SOTA in VisionServeX


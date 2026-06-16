# v3.16 Plan — remaining model-coverage work (honest backlog)

v3.15.0 delivered the capability-truth contract + a full-lifecycle torchvision
classifier family. The items below are **not** done in v3.15.0; their capability
truth is already correct (inference-ready / catalog-only / blocked with an exact
blocker), and this is the planned path to close them.

## P1 — pretrained inference (easy, permissive)

- **torchvision detectors** — `RetinaNet`, `Faster R-CNN`, `Mask R-CNN`
  (COCO-pretrained, BSD-3). New `torchvision_detection` engine + COCO label map +
  box/mask normalization. Inference first; training is heavier (defer).
- **torchvision segmentation** — `DeepLabV3`, `FCN` (COCO/VOC, BSD-3) as
  semantic-segmentation inference.

## P2 — training / fine-tune

- **Generic classifier fine-tune for HF classifiers** (convnextv2 / swinv2 /
  timm) via HF `Trainer` — currently inference-only; add reload+predict lifecycle.
- **Detector lifecycle for torchvision detectors** (COCO-format) once the engine
  exists.

## P3 — heavy / gated (capability truth already correct)

- **SAM / SAM2 fine-tune** — pretrained prompting is inference-ready today;
  training is **blocked** (`SAM_TRAINING_NOT_IMPLEMENTED`). SAM-style fine-tuning
  (mask-decoder / LoRA on point/box prompts) is a separate effort; only expose
  after a real train→reload→predict lifecycle proof. Do not fake.
- **Standalone HF D-FINE training** — stays inference-only
  (`TRAINING_NOT_SUPPORTED_IN_HF_BACKEND`); use `libreyolo-dfine-n` for a trainable
  permissive D-FINE.
- **OpenMMLab / MMDetection detectors** (`_stub`/sidecar) — catalog-only until the
  mmdet sidecar runtime builds; keep `CATALOG_ONLY` blocker.
- **DINOv3 / SAM3 (gated BYOT)** — inference via BYOT token; not training targets.

## Legal firewall (permanent)

- Ultralytics / AGPL / GPL never enters a runtime or default-safe path.
- YOLO-NAS / Deci non-commercial never trainable or default-safe.
- FastSAM only if a permissive (non-AGPL) implementation/weights are confirmed;
  otherwise remains excluded.

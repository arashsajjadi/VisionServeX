# VisionServeX v34 — Final audit report

VisionServeX `2.30.0` / notebook `v34`

## Package smoke matrix
- total: 65
- smoke_passed: 59
- benchmark_passed: 0
- expected_blocker: 0
- dependency_required: 3
- download_failed_retryable: 3
- manual_checkpoint_required: 0
- license_blocked: 0
- failed_runtime: 0
- unclassified: 0
- package_bug_remaining: 0

## Detection
- Dataset: COCO val2017, 400 images, object-rich balanced subset
- Models benchmarked: 13
- Best mAP50:95: `yolo26x.pt` = `0.4894`

## Automatic segmentation
- Dataset: COCO val2017, 400 images, mask GT
- Models benchmarked: 4
- Best mask mAP50:95: `yolo26x-seg.pt` = `0.2728`
- Scope: Ultralytics yolo*-seg only; RF-DETR-Seg AP pending pycocotools

## Promptable segmentation
- Models attempted: `['sam2-hiera-tiny', 'sam2.1-hiera-tiny']`
- smoke/eval set; full COCO 400 promptable benchmark deferred

## v3 readiness
- v3_ready: **False**
- Blockers:
  - RT-DETRv4 manual checkpoints
  - RF-DETR-Seg pycocotools mask-AP integration
  - Full COCO 400 promptable segmentation benchmark
  - Labelled ImageNet-style classification dataset
  - Labelled retrieval dataset for embedding benchmark
  - Permissive aerial/OBB dataset
  - anomalib install
  - natten install
  - transformers<5.0 sidecar for Florence-2
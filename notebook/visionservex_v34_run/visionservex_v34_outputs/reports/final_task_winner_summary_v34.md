# VisionServeX v34 — Final task-specific winner summary

VisionServeX version : `2.30.0`
Notebook version     : `v34`

## Detection (real COCO val2017 400 benchmark)
- **Best mAP50:95**: `yolo26x.pt` (ultralytics) = `0.4894` (AP50=`0.6612` if available)
- **Fastest p50**: `yolov10b.pt` at `6.05 ms` (mAP=`0.4393`)

## Automatic instance segmentation (COCO val2017 400)
- **Best mask mAP50:95**: `yolo26x-seg.pt` = `0.2728`
- Scope: Ultralytics yolo*-seg only — RF-DETR-Seg AP pending pycocotools RLE.

## Promptable segmentation
- Models attempted: `['sam2-hiera-tiny', 'sam2.1-hiera-tiny']`
- Smoke/eval with GT box prompts. Full COCO 400 promptable benchmark deferred.

## Other tasks
- **classification_status**: smoke_only (no ImageNet labels supplied)
- **embedding_status**: smoke_only (no retrieval GT supplied)
- **medical_status**: smoke_only (no NIfTI GT)
- **agriculture_status**: smoke_only (no crop/weed GT)
- **aerial_obb_status**: dataset_required
- **anomaly_status**: expected_blocker (ANOMALIB_REQUIRED)
- **surveillance_status**: expected_blocker (BYTETRACK/OCSORT/TORCHREID required)
# VisionServeX vs Ultralytics Comparison Plan

## Eligible VisionServeX Models

- `dfine-l` (family: dfine)
- `dfine-l-coco` (family: dfine)
- `dfine-l-o365-coco` (family: dfine)
- `dfine-m` (family: dfine)
- `dfine-m-coco` (family: dfine)
- `dfine-m-o365-coco` (family: dfine)
- `dfine-n` (family: dfine)
- `dfine-n-coco` (family: dfine)
- `dfine-s` (family: dfine)
- `dfine-s-coco` (family: dfine)
- `dfine-s-o365-coco` (family: dfine)
- `dfine-x` (family: dfine)
- `dfine-x-coco` (family: dfine)
- `dfine-x-o365-coco` (family: dfine)
- `rfdetr-base` (family: rfdetr)
- `rfdetr-large` (family: rfdetr)
- `rfdetr-medium` (family: rfdetr)
- `rfdetr-nano` (family: rfdetr)
- `rfdetr-small` (family: rfdetr)

## Ultralytics Baselines

- yolo11n
- yolo11s

## Metrics

- AP50
- mAP50_95
- precision
- recall
- latency_ms_median
- latency_ms_p95
- imgs_per_sec
- peak_ram_mb

## Caveats

- 100-image COCO128 subset is not SOTA proof
- Prompt-based/open-vocab models are not directly comparable without closed-set config
- VisionServeX honesty policy: if YOLO wins, the report says so
- VisionServeX value is multi-family platform breadth, not single-task COCO leaderboard

## Not Eligible

- Not closed-set detection: ['dinov2', 'clip', 'siglip', 'siglip2']
- Promptable segmentation: ['sam', 'sam2', 'sam2.1', 'medsam']
- Medical: ['medsam2', 'totalsegmentator', 'nnunet']
- Anomaly detection: ['anomalib']
- Pose estimation: ['rtmpose']
- OBB: ['rtmdet', 'oriented-rcnn']
- AGPL excluded: ['fastsam-s', 'fastsam-x']

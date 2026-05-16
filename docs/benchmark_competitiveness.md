# Benchmark Competitiveness Guide

VisionServeX provides two modes for the `benchmark-competitiveness` command:

## Synthetic mode (default)

No annotated dataset required. Tests latency and detection output health on synthetic images.

```bash
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small \
  --max-images 20
```

Reports: P50/P95 latency, average detections per image, zero-detection rate, invalid-box rate.

**Important:** Synthetic results do not indicate accuracy. Do not use to claim YOLO comparison.

## Real AP/mAP mode (with annotated dataset)

Provide a YOLO-format or COCO JSON annotated dataset.

```bash
# YOLO format
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco,rfdetr-small,ultralytics:yolo11n \
  --dataset yolo:/path/to/coco128 \
  --max-images 100 \
  --out reports/ap_results

# COCO JSON format
visionservex benchmark benchmark-competitiveness \
  --models dfine-s-o365-coco \
  --dataset coco-json:/data/coco/images/val2017:/data/coco/annotations/instances_val2017.json \
  --max-images 500
```

### YOLO format structure

```
dataset/
  images/
    img1.jpg
  labels/
    img1.txt   (class_id cx cy w h — normalized 0-1)
  data.yaml    (names: [person, bicycle, ...])
```

### Metrics computed

| Metric | Description |
|--------|-------------|
| AP50 | Average Precision at IoU=0.50 |
| mAP50:95 | AP averaged over IoU 0.50→0.95 (0.05 step) |
| Precision | At best-F1 threshold |
| Recall | At best-F1 threshold |
| F1 | Harmonic mean of precision and recall |
| Latency P50/P95 | Inference time percentiles |

AP is computed with COCO-style 101-point interpolated PR curves.

### Getting COCO128

```bash
# Using wget:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip
unzip coco128.zip
# Then use: --dataset yolo:coco128/
```

## Output format

With `--out reports/results`, two files are written:
- `reports/results.json` — full results with per-class AP
- `reports/results.csv` — summary table (model, AP50, mAP50:95, latency, ...)

## Honest conclusion

The tool always reports which model has the best AP50 on the given dataset. If Ultralytics YOLO wins, the conclusion will say so. Small datasets (<100 images) produce high-variance AP estimates — a warning is included.

## Non-detection tasks

```bash
# These return BENCHMARK_NOT_IMPLEMENTED (exit code 2)
visionservex benchmark benchmark-segmentation --json
visionservex benchmark benchmark-classification --json
visionservex benchmark benchmark-pose --json
visionservex benchmark benchmark-obb --json
visionservex benchmark benchmark-open-vocab --json
```

Each returns the correct annotation format, recommended dataset, and proper metrics for that task. Roadmap: v1.4.

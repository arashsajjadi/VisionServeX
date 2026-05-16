# Evaluation Metrics Reference

## Detection metrics

### AP50 (Average Precision at IoU=0.50)

The most common detection metric. A predicted box matches a GT box when IoU ≥ 0.50.

**Algorithm (COCO-style):**
1. For each class, collect all predictions across all images sorted by confidence.
2. For each prediction, check if it matches any unmatched GT box in the same image (IoU ≥ 0.50).
3. Compute cumulative precision-recall curve.
4. Interpolate at 101 recall thresholds from 0 to 1.
5. AP = mean of interpolated precision values.

### mAP50:95

Average over IoU thresholds 0.50, 0.55, 0.60, ..., 0.95 (10 thresholds). Then average over classes.

This is the primary COCO metric. Harder than AP50 because it requires tight boxes.

### Precision, Recall, F1

At the best-F1 confidence threshold:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = 2 × Precision × Recall / (Precision + Recall)

### Per-class AP

AP computed independently for each class. mAP = mean over classes with ground truth.

## Segmentation metrics (roadmap v1.4)

- **Mask AP50**: IoU of binary masks instead of boxes
- **Boundary IoU**: Measures boundary precision
- **Panoptic Quality (PQ)**: For panoptic segmentation

## Classification metrics (roadmap v1.4)

- **Top-1 accuracy**: Fraction where predicted class = ground truth
- **Top-5 accuracy**: GT class in top-5 predictions

## Pose estimation metrics (roadmap v1.4)

- **OKS AP**: Object Keypoint Similarity at IoU-equivalent thresholds

## OBB metrics (roadmap v1.4)

- **Rotated IoU AP**: AP using rotated box IoU instead of axis-aligned IoU

## Critical warnings

- Do NOT mix detection AP with segmentation AP, classification accuracy, OKS, or PQ.
- Do NOT compare zero-shot AP (open-vocab) with closed-set AP (COCO detection).
- Do NOT use nano/demo models to judge AP competitiveness.
- AP from <100 images has high variance — run on full val set for reliable numbers.

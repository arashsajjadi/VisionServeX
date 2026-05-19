# Model Checkpoints

Model weights are organized by family under `checkpoints/`.
Model files are **gitignored** — see `.gitignore`.

See `model_inventory.csv` for the full list of moved checkpoints.

## Layout

```
checkpoints/
  rfdetr/       — RF-DETR detection (Apache-2.0)
  rfdetr_seg/   — RF-DETR-Seg (Apache-2.0)
  sam/          — SAM vit-b/l/h (Apache-2.0)
  sam2/         — SAM2 / SAM2.1 (Apache-2.0)
  mobile_sam/   — MobileSAM (Apache-2.0)
  rtdetr/       — RT-DETR (Apache-2.0)
  ultralytics_yolo_new/ — YOLO11, YOLO12, YOLO26 (AGPL)
  ultralytics_yolov8/   — YOLOv8 (AGPL)
  ultralytics_seg/      — YOLO-seg (AGPL)
  libreyolo/    — LibreYOLO weights (per-family license)
```

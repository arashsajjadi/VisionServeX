# Ultralytics → VisionServeX Replacement Map

## Usage

```bash
visionservex replacement-map map
visionservex replacement-map map --task detect
visionservex replacement-map map --task segment
visionservex replacement-map map --task classify
visionservex replacement-map map --task pose
visionservex replacement-map map --task obb
visionservex replacement-map map --task open-vocab
visionservex replacement-map map --task sam
visionservex replacement-map map --format markdown
```

## Detection (YOLO detect → VisionServeX)

| Tier | Models | Notes |
|------|--------|-------|
| fastest_demo | `dfine-n`, `rfdetr-nano` | CPU-capable, not for AP claims |
| production | `dfine-s-o365-coco`, `rfdetr-small` | AP entry points, Objects365 pretrained |
| accuracy | `dfine-m/l/x-o365-coco`, `rfdetr-medium/large` | GPU required, upstream AP 52–56 |

**Caveats:** Run `benchmark-competitiveness --dataset yolo:<path>` to verify AP. Do not use nano variants for AP comparison.

## Segmentation (YOLO-seg → VisionServeX)

| Tier | Models | Notes |
|------|--------|-------|
| fastest_demo | `rfdetr-seg-nano` | CPU-capable demo |
| production | `rfdetr-seg-small`, `rfdetr-seg-medium` | Wired instance segmentation |
| prompt_based | `grounded-sam`, `grounded-sam2` | Text-prompted, not for closed-set AP |
| expert | `co-dino-inst-vit-l-coco` | Not wired — requires OpenMMLab |

## Pose (YOLO-pose → VisionServeX)

Expert only: `rtmpose-s/m/l` via OpenMMLab sidecar. Not natively wired. No verified AP winner vs YOLO-pose.

## OBB (YOLO-OBB → VisionServeX)

Expert only: `rtmdet-r-s/m/l` via MMRotate. Not natively wired. No verified AP winner vs YOLO-OBB.

## Classification (YOLO-cls → VisionServeX)

| Tier | Models | Notes |
|------|--------|-------|
| production | `swinv2-tiny`, `swinv2-small` | CPU-capable, MIT license |
| accuracy | `swinv2-base`, `swinv2-large` | 86%+ ImageNet top-1 |
| expert | `internimage-t/b/l` | DCNv3 custom ops required |

## Open-vocabulary (YOLO-World/YOLOE → VisionServeX)

| Tier | Models | Notes |
|------|--------|-------|
| demo | `grounding-dino-tiny` | CPU-capable, demo_fast |
| accuracy | `grounding-dino-swin-b` | COCO zero-shot ~56.7 AP |
| API-gated | `grounding-dino-1.5/1.6` | external_api, not self-hostable |

## SAM wrappers (Ultralytics SAM → VisionServeX)

`sam-vit-base/large/huge` and `sam2-hiera-tiny/small/base-plus/large` via HF Transformers — same weights as Ultralytics SAM wrapper, no overhead.

## Honesty policy

The replacement map does not claim "better" without AP evidence. It tells you which tier to use for fair comparison. Run `visionservex benchmark benchmark-competitiveness --dataset yolo:<path>` to get actual AP numbers.

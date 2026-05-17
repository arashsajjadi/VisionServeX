# VisionServeX Expected Blockers

These structured errors are expected and tested. Treat them as PASS in notebook.


## GATED_HF_AUTH_REQUIRED
- **models**: ['sam3-base']
- **action**: visionservex sam-family login-help sam3.1

## DETECTRON2_REQUIRED
- **family**: maskdino
- **action**: bash scripts/run_maskdino_smoke.sh

## OPENMMLAB_REQUIRED
- **family**: rtmpose
- **action**: bash scripts/run_openmmlab_rtmpose_smoke.sh

## OBB_INFERENCER_UNAVAILABLE
- **family**: rtmdet
- **action**: bash scripts/run_mmrotate_oriented_rcnn_smoke.sh

## ANOMALIB_REQUIRED
- **family**: anomalib
- **action**: bash scripts/run_anomaly_smoke.sh

## DO_NOT_ADD
- **notes**: FastSAM (AGPL-3.0), DeepSORT (GPL-3.0) — excluded from permissive core

## NON_CORE_LICENSE_OPTIONAL
- **notes**: RF-DETR Plus/XL/2XL (PML 1.0), TotalSegmentator tissue subtasks (proprietary)

# VisionServeX Benchmark Plan

## Group 0: Package audit (quick, default=True)
- Objective: Verify package installs correctly, CLI works, load matrix is clean.
- Models: mock-detect, mock-classify
- Commands:
  - `visionservex version`
  - `visionservex dev cli-audit --json`
  - `visionservex models load-matrix-run --mode all --ci-safe`
  - `visionservex readiness verdict --json`
- Expected output: 23/23 CLI PASS, 0 core_failures, RELEASE_OK
- Caveats: None

## Group 1: Model load matrix (quick, default=True)
- Objective: Confirm every registry model loads or returns structured blocker.
- Command: `visionservex models load-matrix-run --mode all --ci-safe --out /tmp/lmr.json`
- Expected output: n_rows=113, core_failures=0, v3_gate_pass=True
- Caveats: Some rows produce SKIP_EXPECTED due to placeholder commands

## Group 2: Detection vs Ultralytics (balanced, default=False)
- Objective: Compare VisionServeX detectors to YOLO11n/s on COCO128.
- VisionServeX models: dfine-s-o365-coco, dfine-m-o365-coco, rfdetr-small, rfdetr-large
- Ultralytics baselines: yolo11n, yolo11s
- Dataset: COCO128 (100 images) — balanced
- Metrics: AP50, mAP50:95, latency_ms_median, peak_ram_mb
- Commands:
  - `visionservex val dfine-s-o365-coco --dataset yolo:<coco_dir> --max-images 100`
  - `visionservex val rfdetr-small --dataset yolo:<coco_dir> --max-images 100`
- NOT eligible: DINOv2, CLIP, SigLIP, SAM, MedSAM, pose, OBB, VLM
- Caveats: Small subset; honesty policy in effect

## Group 3: Open-vocab detection (quick, default=True)
- Models: grounding-dino-tiny, owlv2-base-patch16
- Command: `visionservex open-vocab MODEL IMAGE --prompt "person, car"`
- Metrics: visual demo only; no COCO AP without closed-set config
- Caveats: Not comparable to COCO-trained closed-set detectors

## Group 4: Classification (quick, default=True)
- Models: swinv2-tiny, convnextv2-tiny, maxvit-tiny-tf-224
- Command: `visionservex benchmark-classification --model MODEL --dataset <dir>`
- Metrics: top-1, top-5, latency

## Group 5: Embedding / Retrieval (quick, default=True)
- Models: dinov2-base, siglip2-base-patch16-224, clip-vit-base-patch32
- Command: `visionservex embed MODEL IMAGE --out /tmp/out.npy`
- Metrics: embedding dim, L2-norm, cosine similarity demo

## Group 6: SAM / Promptable Segmentation (quick, default=True)
- Models: sam-vit-base, sam2-hiera-tiny, sam2.1-hiera-tiny, medsam
- Command: `visionservex sam-family smoke-test MODEL IMAGE --box 10,20,100,200`
- Metrics: IoU (visual), mask output shape

## Group 7: Medical (optional, default=False)
- Models: medsam (runnable), totalsegmentator (sidecar, user NIfTI required)
- Command: `visionservex medical segment medsam IMAGE --box ... --out /tmp`
- Input: RGB image for MedSAM; user-supplied NIfTI for TotalSegmentator
- Disclaimer: NOT for clinical diagnosis

## Group 8: Industrial Anomaly (balanced, default=False)
- Models: anomalib-patchcore (via Anomalib sidecar)
- Command: `bash scripts/run_anomaly_smoke.sh`
- Requires: pip install anomalib
- Metrics: pred_score (anomaly score) on synthetic fixture
- No MVTec AD bundled (CC BY-NC-SA 4.0 not commercial-safe)

## Group 9: Surveillance / Video Search (balanced, default=False)
- Trackers: simple-iou (builtin), bytetrack (optional), oc-sort (optional)
- ReID: osnet (optional, checkpoint required)
- Command: `visionservex video-search tracker-smoke --tracker bytetrack`
- Requires: pip install bytetracker ocsort filterpy

## Group 10: OpenMMLab sidecar (sidecar, default=False)
- Models: rtmpose-m, rtmdet-tiny-coco
- Requires: conda Python 3.10, setuptools<72, mmcv 2.1.0
- Command: `bash scripts/run_openmmlab_rtmpose_smoke.sh`
- Output: 17 keypoints (RTMPose), 300 boxes (RTMDet)

## Group 11: Aerial / OBB (sidecar, default=False)
- Models: rtmdet-r2-s, oriented-rcnn (legacy mmrotate sidecar)
- Requires: Legacy sidecar (torch 1.13+cu117, mmcv-full 1.7, mmrotate 0.3.4)
- Command: `bash scripts/run_mmrotate_oriented_rcnn_smoke.sh IMAGE`
- OBB schema: [x_center, y_center, width, height, theta, score, label]

## Group 12: Non-core / Gated / Sidecar (quick audit, default=True)
- Models: sam3-base (gated), fastsam-s (AGPL excluded), rfdetr-seg-large (PML)
- Command: `visionservex model-zoo blockers --family maskdino --refresh`
- Expected: all return structured blocker codes, no crashes

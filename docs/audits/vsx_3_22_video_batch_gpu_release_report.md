# VisionServeX v3.22.0 — Video / True-Batch / GPU / Memory Release Report

**Branch:** `fix/video-true-batch-gpu-memory-vsx` · **GPU:** RTX 5080 (sm_120, 16 GB)
**Toolchain:** torch 2.11.0+cu130, ffmpeg 8.0.1 (NVENC+NVDEC), Python 3.13.12

This report summarizes what changed, what was proven with live evidence, and what
remains an honest blocker. Evidence JSON is under `docs/audits/evidence/`.

## Headline outcomes (all live-measured on RTX 5080)
1. **True tensor batch — implemented & proven for D-FINE.** 8 images → 1 forward;
   per-image logits bitwise-identical regardless of batch-mates; 77→231 fps. A
   forward-call verifier + a failing test catch any engine that fakes batch.
2. **Adaptive GPU scheduler — implemented & proven.** Climbed batch 4→12
   (71→210 fps) then held at the *measured* CPU-preprocess bottleneck — the honest
   answer to "why isn't the GPU at 100%".
3. **Worker video pipeline — implemented & proven on the owner's video.** ffprobe
   diagnostics; lossless faststart remux (74 ms); browser H.264 transcode via NVENC
   (4.46 s); streamed frame extraction (61–245 fps); HTTP endpoints.
4. **Segmentation masks — fixed.** RF-DETR-Seg masks were produced but dropped at
   JSON time; now serialized as COCO RLE by default (round-trip IoU 1.0) + polygons
   on request + quality checks + honest masks-unavailable warning.
5. **Real cancellation + lifecycle.** Cancel now interrupts a running video batch
   (stopped at 8/916 frames), releases VRAM (→33 MB), marks the job cancelled.
6. **Tiled inference for small objects.** 7→22 detections on a late traffic frame,
   smallest box 498→148 px², via overlapping tiles + cross-tile class-aware NMS.

## What was honest-corrected
- The owner's belief "D-FINE is the only true tensor-batch path" was **half-true**:
  D-FINE *can* true-batch but the adapter always ran batch=1 (now fixed). RF-DETR's
  registry `batch_support=True` was **unproven** (adapter is single-image) → the
  registry was corrected so `batch_support=true` iff the engine is `dfine`.
- "GPU stays 0–10%" is **real and explained**: for small detectors the CPU
  preprocess + frame decode dominate; the GPU forward is a small fraction. Measured,
  not guessed.
- "Segmentation returns boxes" was a **serialization** bug, not a model bug.

## New modules
- `runtime/batch_infer.py` — telemetry + true-forward-batch verifier.
- `runtime/adaptive_batch.py` — adaptive scheduler.
- `runtime/ffmpeg_tools.py` — probe/remux/transcode/hwaccel (preset-only).
- `runtime/video_pipeline.py` — streamed, cancellable, adaptive video inference.
- `runtime/video_export.py` — overlay MP4 (NVENC) + annotation exporters.
- `runtime/mask_encoding.py` — RLE/polygon/quality.
- `runtime/tiling.py` — sliced inference + per-frame diagnostics.
- `server/video_routes.py` — `/infer-batch`, `/video/*`, `/jobs/*` endpoints.

## Tests & benchmarks
- Unit (CI-safe, no GPU): `tests/test_v322_true_batch.py`,
  `test_v322_adaptive_scheduler.py`, `test_v322_segmentation_output.py`,
  `test_v322_video_pipeline.py` — 31 passing.
- Live GPU (gated `VSX_LIVE_GPU=1`): `tests/live/test_v322_live_gpu.py` — 4 passing.
- Benchmarks (JSON+MD): `scripts/bench/{model_variant_matrix,video_batch_matrix,
  memory_lifecycle,video_decode_matrix}.py`.

## Remaining honest blockers
- **True batch beyond D-FINE.** RF-DETR's package accepts a list but its forward-
  batch is unverified; LibreYOLO/SAM are single-image. Not upgraded without proof.
- **Export formats.** Overlay MP4 + JSON/CSV/COCO implemented; YOLO/VOC/ZIP not yet.
- **Utilization on small models** is CPU/decode-bound by nature, not a bug.

See the per-phase evidence files and `docs/anastig_integration_contract.md`.

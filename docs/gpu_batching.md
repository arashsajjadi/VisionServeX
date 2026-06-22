# GPU Batching (v3.22.0)

## True tensor batch vs internal loop
A **true tensor batch** runs `model.forward` ONCE over a stacked batch of N images.
An **internal loop** calls single-image `predict` N times. VisionServeX makes this
falsifiable: `runtime.batch_infer.verify_true_forward_batch` counts forward calls,
and `tests/test_v322_true_batch.py` fails any engine that claims true batch while
looping.

- `BaseEngine.predict_batch` default = honest loop, tags `batch_mode="internal_loop"`.
- `DFINEEngine.predict_batch` = real batched forward, `supports_true_batch=True`.
- **D-FINE is the only proven true-batch path today** (all variants). RF-DETR,
  RF-DETR-Seg, LibreYOLO, SAM/SAM2 run internal-loop and say so.

### Proof (RTX 5080, dfine-n, 8 real frames)
- 8 images → **1** forward call; input batch dim 8.
- Same image at batch-position 0 with different batch-mates → **bitwise-identical
  logits (max diff 0.0)** ⇒ no cross-image contamination.
- Throughput 77→231 fps (1→16), VRAM 71→724 MB. (`docs/audits/evidence/dfine_n_truebatch_proof.json`)

## Adaptive scheduler (`runtime.adaptive_batch.AdaptiveBatchScheduler`)
- Non-power-of-two ladder `1,2,3,4,6,8,12,16,24,32,48,64,96,128` (model-capped).
- Grows only with VRAM headroom (target 85%) AND throughput gain (hysteresis).
- Shrinks on OOM (halve + lower ceiling), hard VRAM (92%), emergency (95%),
  latency spike (low_latency), cancel, or queue backup.
- Modes: `balanced | max_throughput | low_memory | low_latency | small_objects |
  segmentation_quality`.
- **Bottleneck attribution**: classifies each wave as forward/preprocess/postprocess/
  vram-bound and STOPS growing when CPU-bound — no blind VRAM filling.

### Proof (live, max_throughput)
Climbed 4→6→8→12 (71→210 fps) then held with reason: *"preprocess-bound (cpu 29 ms
vs forward 27 ms) and throughput plateaued → growing batch won't help the GPU."*
(`docs/audits/evidence/adaptive_scheduler_live.json`)

## Why the GPU often looks idle
For small detectors the **CPU preprocess** (HF image processor) and **frame decode**
dominate; the GPU forward is a small fraction. v3.22 measures and reports this
(`batch.preprocess_ms` vs `forward_ms`, `gpu_util_avg`) instead of guessing. The
fix for higher utilization is a larger model and/or faster decode/preprocess — not
a bigger batch.

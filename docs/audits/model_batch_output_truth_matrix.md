# VisionServeX — Model Batch / Output Truth Matrix (Phase 1)

**Branch:** `fix/video-true-batch-gpu-memory-vsx` · **GPU:** RTX 5080 (sm_120, 16 GB) · **Date:** 2026-06-22

This matrix reconciles **registry claims** against **measured / code-audited behavior**. Two columns
are authoritative for the owner's concerns:

- **Output type** — what the engine adapter actually returns (boxes / masks / etc.).
- **Batch path** — `true_tensor_batch` (one forward on a stacked batch), `internal_loop` (Python loop
  over single-image predict), `package_list_predict` (delegates a list to the upstream package —
  forward-batch truth depends on the package and is verified separately), or `single_image_only`.

Evidence type is labelled per row: **LIVE** (measured on this GPU this sprint),
**CODE** (audited from source, file:line in the baseline report), **REGISTRY** (unverified claim).

---

## 1. Headline reconciliation (the owner's central claims)

| Claim (owner / registry) | Truth | Evidence |
| ------------------------ | ----- | -------- |
| "D-FINE is the only true tensor-batch path" | **Half-true.** D-FINE *can* true-batch (proven), but it was NEVER exercised — the adapter always ran batch=1. RF-DETR's package *also* accepts a list. | LIVE + CODE |
| Registry `dfine-* batch_support=False/None` | **WRONG** — D-FINE true-batches (1 forward for N images, 3× throughput). To be corrected. | LIVE proof |
| Registry `rfdetr-base/nano/small batch_support=True` | **Plausible** — `rfdetr.predict()` accepts `list[...]` → `list[sv.Detections]`. Forward-batch truth verified in Phase 2. Adapter currently single-image. | CODE (signature) |
| "RF-DETR-Seg outputs mostly boxes" | **Adapter keeps masks; the JSON contract drops them** (`SegmentationResult.to_dict` emits only mask_shape). Fixed in Phase 5. | CODE |
| "LibreYOLO / RF-DETR-Seg report worker_internal_loop" | **Correct today** — both run single-image; no batch wired. | CODE |

## 2. D-FINE — LIVE true-batch proof (dfine-n, RTX 5080, real video frames)

`scripts/bench/_dfine_truebatch_probe.py` → `docs/audits/evidence/dfine_n_truebatch_proof.json`

| batch | forward calls | latency | throughput | VRAM peak | GPU util avg |
| ----: | ------------: | ------: | ---------: | --------: | -----------: |
| 1  | 1 | 13.0 ms | 76.9 fps | 71 MB | 1% |
| 2  | 1 | 20.9 ms | 95.9 fps | 112 MB | 27% |
| 4  | 1 | 24.2 ms | 165.3 fps | 200 MB | 27% |
| 8  | 1 | 34.6 ms | 231.0 fps | 374 MB | 27% |
| 16 | 1 | 69.2 ms | 231.1 fps | 724 MB | 27% |

**Proof of true batch (not a loop):** 8 images → **1** forward call (vs 8 single-image calls);
input tensor batch dim = 8.
**Proof of per-image independence:** the same image at batch-position 0 with *different* batch-mates
gives **bitwise-identical logits (max diff 0.0)** — D-FINE/DETR has no cross-image attention.
The thr=0.3 single-vs-batched count delta (e.g. 95→96) is **benign FP/kernel nondeterminism**; at
strict thr=0.5 the per-image counts are identical (20 = 20).
**Honest bottleneck:** GPU util plateaus at ~27% — dfine-n is tiny and **CPU-preprocess-bound**, so
true batch raises *throughput* (3×) but not *utilization*. Larger D-FINE variants and the Phase-3
scheduler's bottleneck attribution address "why isn't the GPU at 100%."

## 3. Per-family truth matrix

| Family / variant | Task | Output type (real) | Single predict | Batch path | Evidence | Recommended use | Verdict vs registry |
| ---------------- | ---- | ------------------ | -------------- | ---------- | -------- | --------------- | ------------------- |
| **D-FINE** n/s/m/l/x (+coco/o365), 14 | detect | boxes | yes | **true_tensor_batch** (proven n; same code path all) | LIVE (n), CODE | high-throughput / video batch detection | registry `batch_support=False/None` → **upgrade to true** |
| **RF-DETR** base/nano/small/medium/large | detect | boxes | yes | **package_list_predict** (`list`→`list[sv.Detections]`); forward-batch truth measured in Phase 2 | CODE (sig) | detection; batch pending Phase-2 verdict | `True` plausible but adapter was single-image |
| **RF-DETR-Seg** nano/small/medium (wired) | segment | **boxes + masks** (uint8 HxW); preserved in `Segment.mask` | yes | package_list_predict (per-image mask postproc) | CODE | segmentation (masks real) | NOT boxes-only; **JSON contract dropped masks** → Phase 5 |
| RF-DETR-Seg large/xlarge/2xlarge | segment | — | **STUB** (`impl=stub status=experimental`) | unsupported | REGISTRY | not recommended (not implemented) | honest: keep marked stub |
| **LibreYOLO** yolox/yolov9/rtdetr/dfine, 14 | detect | boxes (+NMS via `class_aware_nms`) | yes | **internal_loop** (single-image `self._model(image)`) | CODE | detection (single/loop) | matches "worker_internal_loop" report |
| **SAM / SAM2 / SAM2.1** | foundation_segment | **masks** (best-IoU mask, uint8) | yes (prompt) | single_image_only | CODE | prompt segmentation | masks real; contract dropped → Phase 5 |
| **GroundedSAM / GroundedSAM2** | grounded_segment | boxes + masks | yes | **fake batch** (SAM looped per detection) | CODE | grounded segmentation | honest: per-detection loop |
| **GroundingDINO** ×7 | open_vocab_detect | boxes + labels | yes | single_image_only | CODE | prompt detection | — |
| **DINOv2** small/base/large/giant | embed | embeddings | yes | single_image_only | CODE | embeddings / similarity | — |
| **OneFormer** ×3 | segment | masks (semantic/instance) | yes | single_image_only | CODE | segmentation | — |
| **torchvision-classify** ×13 | classify | top-k logits | yes | single_image_only | CODE | classification | — |
| **OpenMMLab** rtmdet/rtmpose/internimage ×21 | detect/pose/obb | boxes/keypoints | sidecar/stub | unsupported in-process | REGISTRY/CODE | sidecar only | hidden/sidecar-gated |
| **Florence-2** ×2 | vlm | caption/OD | sidecar | unsupported in-process | CODE | sidecar only | sidecar-gated |
| **INSID3** small/base/large | in-context segment | correspondence/masks | yes (runtime) | single_image_only | CODE | in-context seg | — |

## 4. Output-type honesty rules applied
- **No model advertised as segmentation that returns only boxes.** RF-DETR-Seg, SAM*, OneFormer,
  GroundedSAM produce real masks at the adapter level (`Segment.mask`). The defect is the **JSON
  serialization contract**, fixed in Phase 5 (RLE/polygon), not the model.
- **No model advertised as true-batch unless a forward-call test proves a single forward for N>1.**
  Only D-FINE currently qualifies (LIVE). RF-DETR is verified in Phase 2 before any upgrade.
- Stub seg variants (rfdetr-seg large/xlarge/2xlarge) remain marked stub — not advertised as usable.

## 5. Not live-measured this phase (honest gap)
Per-variant VRAM/GPU/CPU/preprocess/forward/postprocess timings are LIVE only for dfine-n.
The Phase-9 benchmark (`scripts/bench/model_variant_matrix.py`) extends LIVE timings to additional
variants via the new `predict_batch` API. Rows marked CODE are reconciled by source audit
(file:line in `video_true_batch_gpu_memory_baseline.md` §7), not by a batch run.

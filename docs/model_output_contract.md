# Model Output Contract (v3.22.0)

## Detection
`DetectionResult.detections[]` → `{box:{x1,y1,x2,y2}, score, label, class_id}`.
`metadata` may carry `batch_mode`, `preprocess_ms`, `forward_ms`, `postprocess_ms`,
and (tiled) `tile_count`, `raw_count`, `final_count`.

## Segmentation
`SegmentationResult` serializes (see `core/results.py`):
- `output_mode` ∈ `boxes+masks_rle | boxes+masks_rle+polygons | boxes+masks |
  boxes_only_masks_unavailable | empty`
- `masks_available` (bool)
- per segment: `box`, `score`, `label`, `class_id`, `mask_shape`, `mask_pixels_on`,
  `rle` (COCO, default ON), `polygons` (on request), `mask_quality`
- Flags: `return_boxes`, `return_masks`, `return_rle` (default True), `return_polygons`
  (default False), `return_quality`, `max_polygon_points`, `polygon_simplification_tolerance`.

**Truth rules.** Masks are produced by the engine and now transmitted as RLE by
default (round-trips to the exact mask, IoU 1.0). A seg model with no real masks
yields `boxes_only_masks_unavailable` + a `SEGMENTATION_MASKS_UNAVAILABLE` warning;
it is never silently presented as segmentation.

## Output-type per family (measured/audited)
| Family | Output | True batch |
| ------ | ------ | ---------- |
| D-FINE | boxes | **yes** (proven) |
| RF-DETR | boxes | no (internal loop) |
| RF-DETR-Seg (nano/small/medium) | boxes + masks (RLE/polygon) | no |
| RF-DETR-Seg (large/xlarge/2xlarge) | unsupported (registry stub) | — |
| LibreYOLO | boxes (+NMS) | no |
| SAM / SAM2 / SAM2.1 | masks | no |
| GroundedSAM | boxes + masks | no (per-detection loop) |

See `docs/audits/model_batch_output_truth_matrix.md` for the full matrix and
evidence type per row.

## Capability fields (`model_capabilities(id)`)
`supports_true_batch`, `batch_path`, `max_batch_size_hint`, `preferred_batch_sizes`,
`registry_batch_support_claim` — the engine class is authoritative; the registry
`batch_support` field was corrected to match (only `dfine` is true).

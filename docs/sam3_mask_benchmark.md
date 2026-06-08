# SAM3 / SAM3.1 Real Mask Benchmark (v3.10.0)

## What this is

A functional mask benchmark for SAM3 and SAM3.1 in VisionServeX BYOT mode.
This is **not** a full COCO-style AP benchmark. It confirms that the models
produce real, non-zero mask pixels on a representative image, using the
threshold fix described below.

---

## Results

| Model | HF repo | Mask area (px) | Masks returned | Latency (CPU) | State |
|---|---|---|---|---|---|
| SAM3 | `facebook/sam3` | 62,423 | 5 | ~12 s | `benchmark_passed_byot_mask` |
| SAM3.1 Base-Plus | `facebook/sam3.1` | 306,808 | 5 | ~14 s | `benchmark_passed_byot_mask` |

Image: `tests/assets/smoke/coco_person_car.jpg` (480×640, person + car).
Text prompt: `"person"`.

---

## Root cause of the smoke-only state (before v3.10.0)

SAM3 outputs instance logits where most values are negative (max ~0.69 across
200 proposals). Using the default threshold of `0.5` in
`post_process_instance_segmentation` filters *all* proposals out — resulting in
zero masks returned and `benchmark_passed_byot` (smoke only).

**Fix:** Use `threshold=0.0` to collect all 200 proposals, then sort by score
descending and keep the top-5. This is how SAM3's intended post-processing works
when scores are in logit space.

---

## Code

```python
# From byot_runtime.sam3_segment()
results = proc.post_process_instance_segmentation(
    outputs, target_sizes=target_sizes, threshold=0.0
)
# Sort top-5 by score
r0 = results[0]
masks = r0["masks"]          # shape [200, H, W]
scores = r0["scores"]        # logit scores
top_idx = scores.argsort(descending=True)[:5]
masks = masks[top_idx]
mask_area_px = int(masks.float().sum(dim=0).gt(0).sum())
state = "benchmark_passed_byot_mask" if mask_area_px > 0 else "benchmark_passed_byot"
```

---

## SAM3.1 loading quirk

SAM3.1 checkpoints ship as `sam3.1_multiplex.pt`, not `pytorch_model.bin` or
`model.safetensors`. `Sam3Model.from_pretrained` raises `OSError` when the
standard filename is missing. Fix: `snapshot_download`, then symlink
`pytorch_model.bin → sam3.1_multiplex.pt` in a working directory.

This is handled automatically in `byot_runtime.sam3_segment()`.

---

## Artifacts

Locally generated (gitignored):

```
notebook/99_final_report/artifacts/v310/sam3_benchmark/
  sam3/
    mask.png        # binary mask image
    overlay.png     # original + mask + boxes overlay
    boxes.json      # top-5 boxes and scores
    metadata.json   # model info, mask_area_px, benchmark_state
    latency.json    # load_ms, infer_ms
  sam3_1_base_plus/
    mask.png, overlay.png, boxes.json, metadata.json, latency.json
  sam3_benchmark_summary.json
```

---

## Requirements

```bash
pip install 'visionservex[hf]'
huggingface-cli login   # token with access to facebook/sam3 and facebook/sam3.1
# Accept licenses at:
#   https://huggingface.co/facebook/sam3
#   https://huggingface.co/facebook/sam3.1
```

`transformers>=5.0` for `Sam3Model` and `Sam3Processor`.

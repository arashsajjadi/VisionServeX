# DINOv3 CHMv2 DPT Depth Head (v3.10.0)

## What this is

VisionServeX BYOT support for `facebook/dinov3-vitl16-chmv2-dpt-head`, a ViT-L/16
backbone (~337M parameters) with a CHMv2 DPT depth estimation head for monocular
depth estimation.

---

## Requirements

| Requirement | Minimum version | Notes |
|---|---|---|
| `transformers` | `>=5.10` | `CHMv2ForDepthEstimation` and `CHMv2ImageProcessorFast` added in 5.10.0 |
| `torch` | `>=2.0` | CPU inference works; GPU recommended for speed |
| HF token | required | `facebook/dinov3-vitl16-chmv2-dpt-head` is gated |
| Upstream license | required | Accept on the model page before use |

> **Conflict warning:** `transformers>=5.10` conflicts with the Florence-2 extra
> which requires `<5.0`. Install DINOv3 in a separate virtual environment or use
> the `[dino]` extra explicitly in a fresh env.

---

## Results (v3.10.0 local benchmark, CPU)

| Metric | Value |
|---|---|
| Load time | ~753 ms |
| Inference time | ~1,917 ms |
| Depth tensor shape | `[1, 480, 640]` |
| Non-zero pixels | 307,200 / 307,200 (full image coverage) |
| State | `benchmark_passed_byot_depth` |

---

## Usage

```python
from visionservex.byot_runtime import dinov3_depth

result = dinov3_depth(
    "facebook/dinov3-vitl16-chmv2-dpt-head",
    "image.jpg",
    device="cpu",
)
# result["depth_shape"]        → [1, 480, 640]
# result["depth_nonzero_px"]   → 307200
# result["state"]              → "benchmark_passed_byot_depth"
# result["load_ms"]            → 753.4
# result["infer_ms"]           → 1917.2
```

If `transformers < 5.10` is installed, the function returns immediately with
`state = "runtime_blocked_byot"` and explains the version requirement.

---

## Policy

| Field | Value |
|---|---|
| Model ID | `dinov3-vitl16-chmv2-dpt-head` |
| HF repo | `facebook/dinov3-vitl16-chmv2-dpt-head` |
| Policy bucket | `byot_license_required` |
| `can_ship_weights` | `False` |
| License | Custom DINOv3 License |
| Commercial use | Depends on upstream DINOv3 License terms you accepted |

---

## Artifacts

Locally generated (gitignored):

```
notebook/99_final_report/artifacts/v310/dinov3_chmv2_dpt/
  depth.png       # depth map visualization
  metadata.json   # model info, depth stats, benchmark_state
```

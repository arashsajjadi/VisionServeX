# Smart Annotation — Classic (weight-free) Interactive Segmentation

VisionServeX ships a **commercial-safe, CPU-only, weight-free** smart-annotation
toolkit for turning a coarse user prompt (box / points / scribble / polygon /
mask hint) into a clean instance mask. These are **classic algorithms, not
pretrained models** — there are no model weights, so the only legal surface is
the dependency license, and every dependency is permissive (OpenCV Apache-2.0;
scikit-image / scikit-learn / scipy / numpy BSD-3). No GPL dependency is pulled
in (in particular **PyMaxflow is deliberately avoided**; the graph-cut methods
use OpenCV's own Apache-2.0 min-cut).

Because they are not models, they are tracked **separately** from the pretrained
model leaderboard in `notebook/99_final_report/reports/smart_tool_coverage_ledger.csv`
(V3 gate V3-13).

Install: `pip install "visionservex[classic-ml]"`

## Methods

| method | algorithm | dependency license | best prompt |
|---|---|---|---|
| `classic-grabcut` | GrabCut GMM graph-cut | OpenCV Apache-2.0 | box + pos/neg points |
| `classic-marker-watershed` | marker-controlled watershed | OpenCV Apache-2.0 | fg/bg seeds |
| `classic-random-walker` | random-walker diffusion | scikit-image BSD-3 | fg/bg scribbles |
| `classic-slic-graphcut` | SLIC superpixels + GrabCut snap | skimage BSD-3 + OpenCV Apache-2.0 | box |
| `classic-intelligent-scissors` | live-wire min-cost contour | OpenCV Apache-2.0 | polyline / polygon |
| `classic-interactive-rf` | per-pixel RandomForest | scikit-learn BSD-3 | dense scribbles |
| `classic-slic-rf-smooth` | RandomForest over SLIC superpixels | skimage + sklearn BSD-3 | scribbles |
| `classic-edge-plus` | GrabCut + Canny edge snap + cleanup | OpenCV Apache-2.0 + skimage BSD-3 | box |

## Output contract

Every method returns the same contract (`smart_annotation.contracts.RefineResult`):

```json
{
  "mask": "H x W uint8 {0,1}",
  "polygon": "optional COCO polygon [[x,y],...]",
  "bbox": "xyxy",
  "confidence": "optional float or null",
  "method": "classic-grabcut",
  "latency_ms": 0.0,
  "device": "cpu",
  "license_safe": true
}
```

## Python API

```python
from visionservex.smart_annotation import Prompt, refine

result = refine(image_bgr, Prompt(box=(10, 20, 200, 220),
                                  positive_points=[[100, 120]],
                                  negative_points=[[5, 5]]),
                method="classic-grabcut")
result.mask           # H x W uint8
result.to_contract_dict()
```

Accepted prompt modalities: `box`, `polygon`, `positive_points`,
`negative_points`, `scribble`, `polyline`, `mask_hint`.

## Benchmark (real)

`scripts/v3_classic_smart_refine_benchmark.py` runs a **real** promptable
benchmark: real COCO val2017 images + real GT instance masks, with the user
prompt derived from the GT mask (the standard SAM-style protocol). The metric
(IoU / boundary-IoU vs GT) is therefore a real measurement, not a synthetic toy.

40 COCO instances, max-side 256, CPU:

| method | mean IoU | boundary IoU | SR@0.5 | latency |
|---|---|---|---|---|
| classic-intelligent-scissors | 0.774 | 0.796 | 0.93 | 133 ms |
| classic-grabcut | 0.423 | 0.358 | 0.35 | 164 ms |
| classic-marker-watershed | 0.384 | 0.360 | 0.40 | 1.3 ms |
| classic-edge-plus | 0.367 | 0.335 | 0.38 | 167 ms |
| classic-slic-graphcut | 0.333 | 0.289 | 0.20 | 190 ms |
| classic-random-walker | 0.270 | 0.234 | 0.23 | 112 ms |
| classic-slic-rf-smooth | 0.135 | 0.118 | 0.03 | 53 ms |
| classic-interactive-rf | 0.100 | 0.075 | 0.00 | 39 ms |

Honest reading: with a strong polygon prompt, live-wire (intelligent-scissors)
dominates; the pixel-RF methods need **dense** scribbles and underperform on
cluttered real scenes with only sparse seeds. All methods are CPU-only and
license-safe. For high-accuracy promptable segmentation prefer a SAM/SAM2 model
(`visionservex benchmark-promptable-segmentation`); the classic tools are for
offline, weight-free, fully-permissive annotation refinement.

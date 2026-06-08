# SAM2.1 ONNX Image-Encoder Export (v3.10.0)

## What this is

Export of the SAM2.1 image encoder (vision encoder) to ONNX format for deployment
with ONNX Runtime. This covers the **image encoder only** — the full interactive
decoder (point/box prompt + mask prediction) is not yet exported to ONNX.

---

## Results (v3.10.0)

| Metric | Value |
|---|---|
| Model | `facebook/sam2.1-hiera-base-plus` |
| ONNX export | SUCCESS |
| ONNX file size | ~1.2 MB |
| Export time | ~14.9 s (CPU, opset 17) |
| ONNX Runtime infer (CPU) | ~1.3 s per image |
| Output shape | `image_embeddings (1, 32, 256, 256)` + 2 multi-scale feature maps |
| State | `benchmark_passed_byot_onnx` |

---

## Root cause of prior failure

`transformers.Sam2Model` does **not** expose `image_encoder` as a named submodule.
It exposes `vision_encoder` as a submodule, and `get_image_embeddings()` as a
convenience method. Direct `torch.onnx.export` requires a callable `nn.Module`,
not a bound method.

**Fix:** Wrap `get_image_embeddings()` in a `nn.Module` shim:

```python
class _EmbedShim(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self._m = m

    def forward(self, pixel_values):
        return self._m.get_image_embeddings(pixel_values=pixel_values)

shim = _EmbedShim(model).eval()
torch.onnx.export(
    shim,
    (pixel_values,),
    "sam21_encoder.onnx",
    input_names=["pixel_values"],
    output_names=["image_embeddings"],
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=17,
)
```

---

## Limitations

- This is **image encoder ONNX only**, not the full SAM2.1 interactive pipeline.
- The full interactive decoder (point/box prompting → mask) requires the
  `facebookresearch/sam2` package (`python3 -m sam2.onnx_exporter`) which is
  separate from `transformers.Sam2Model`.
- ONNX artifacts are generated locally and are gitignored; they are not included
  in any VisionServeX release.

---

## Requirements

```bash
pip install 'visionservex[hf]' onnx onnxruntime
huggingface-cli login
# Accept the license at: https://huggingface.co/facebook/sam2.1-hiera-base-plus
```

---

## Artifacts

Locally generated (gitignored):

```
notebook/99_final_report/artifacts/v310/
  sam21_onnx_attempt.json          # export result metadata
  sam21_hiera_base_plus_encoder.onnx  # the ONNX file (NOT committed to git)
```

---

## Policy

| Field | Value |
|---|---|
| Model | `facebook/sam2.1-hiera-base-plus` |
| Weights license | Apache-2.0 |
| `can_ship_weights` | `False` (download-on-demand only) |
| ONNX artifact | NOT shipped in any VisionServeX release |

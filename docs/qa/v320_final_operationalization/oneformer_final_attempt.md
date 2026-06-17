# OneFormer final attempt (v3.20)

Confirmed v3.19 findings; no new feasible path. Both stay blocked with exact reasons.

## `oneformer-convnext-large` → `WEIGHTS_MISSING_HF_404`
Re-verified: enumerating `shi-labs` on HF Hub yields only `{ade20k,cityscapes,coco} ×
{dinat,swin}` + `ade20k_swin_tiny`. **No ConvNeXt OneFormer checkpoint exists on HF
in any transformers-loadable form** — only raw `.pth` in SHI-Labs GitHub releases.
The 404 is permanent. Not renamed/re-pointed (that would misrepresent the model).
The sibling `oneformer-swin-large` is already `SEGMENTATION_READY_LIVE`.

## `oneformer-dinat-large` → `NATTEN_API_INCOMPATIBLE` (GPU-only)
NATTEN 0.21.6 is installed but removed the `natten2dav`/`natten2dqkrpb` functional
API that `transformers/models/dinat/modeling_dinat.py` imports → `ImportError` at
module import, before weights matter. No CPU path (NATTEN is a CUDA extension with
no generic CPU wheel). A shim that re-aliases the legacy names onto NATTEN's new
`na2d` ops is possible but signature-risky and GPU-only for a single model; not
shipped. The sibling `oneformer-coco-dinat-large` is already inference-live (it
predates the broken import path in its cached snapshot).

**Next work:** for dinat, align transformers↔NATTEN (a transformers release whose
`modeling_dinat` targets `na2d`, or a vetted alias shim), GPU-only.

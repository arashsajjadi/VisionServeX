# OneFormer repair (v3.19)

Two OneFormer variants are blocked. Both **MIT**. Neither was operationalized this
sprint; both keep an **exact, corrected blocker**. (Note: `oneformer-coco-swin-large`
and `oneformer-dinat-large`'s sibling `oneformer-coco-dinat-large` are already
`SEGMENTATION_READY_LIVE` — only the two below are blocked.)

## `oneformer-convnext-large` → `WEIGHTS_MISSING` (permanent, not a typo)

- Configured `hf_repo_id` 404s on the HF Hub.
- **Root cause:** there is **no ConvNeXt OneFormer checkpoint on HF Hub under any
  name.** Enumerating the entire `shi-labs` namespace
  (`HfApi().list_models(author="shi-labs", search="oneformer")`) returns exactly
  `{ade20k, cityscapes, coco} × {dinat, swin}` + `ade20k_swin_tiny`. ConvNeXt
  OneFormer weights exist only as raw `.pth` in SHI-Labs GitHub releases — never
  converted/pushed to HF in a `transformers`-loadable format.
- **Verdict:** the 404 is permanent. Stays `WEIGHTS_MISSING`.
- **Next work (optional, not done — would change the model's identity):** either
  remove the row, or re-point + rename to a real backbone that exists, e.g.
  `shi-labs/oneformer_ade20k_swin_large` (then the id must not claim `convnext`).
  Not done because it would misrepresent the model.

## `oneformer-dinat-large` → `DEPENDENCY_MISSING` (corrected diagnosis)

- The v3.18 label "NATTEN op missing" is **imprecise.** NATTEN **is** installed
  (`0.21.6+torch2110cu130`, compiled `.so` present). The real failure is a
  **NATTEN ↔ transformers API mismatch**:
  `transformers/models/dinat/modeling_dinat.py` does
  `from natten.functional import natten2dav, natten2dqkrpb`, but NATTEN 0.21.6
  **removed** that legacy functional API (it now exports `na1d`/`na2d`/`na3d`).
  Because `is_natten_available()` only checks the import succeeds, transformers
  takes the broken branch → `ImportError: cannot import name 'natten2dav'`, and
  the whole `modeling_dinat` module fails to import before weights matter.
- **No CPU path:** DiNAT's forward calls the NATTEN CUDA kernels and gates on
  `requires_backends(self, ["natten"])`; NATTEN ships no generic CPU wheel.
- **Verdict:** `DEPENDENCY_MISSING`, GPU-only, **FIXABLE_MODERATE** but not done.
- **Next work:** either (a) align transformers to a release whose `modeling_dinat`
  targets NATTEN's `na2d` API (risks the same SAM3 transformers-coupling concerns),
  or (b) inject `natten2dav`/`natten2dqkrpb` shim aliases wrapping the new `na2d`
  ops into `natten.functional` before `modeling_dinat` imports. Not shipped this
  sprint (signature-mapping risk + GPU-only payoff for one model).

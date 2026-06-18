# DEIM / DEIMv2 / RT-DETRv4 custom-loader feasibility (v3.19)

Families `deim` (10 ids) + `rtdetrv4` (4 ids). All are `engine: _stub`,
`download_type: not_available`. **None operationalized this sprint** —
in-process loading is infeasible; each keeps `CUSTOM_LOADER_REQUIRED` with an
exact reason (see `deim_rtdetrv4_live_matrix.json`).

## What's actually on disk (HF cache)

| Cached repo | config | Load path | License | Gated |
|---|---|---|---|---|
| `Intellindust/DEIMv2_DINOv3_S_COCO` | nested non-HF (`DEIMTransformer`/`DINOv3STAs`/`HybridEncoder`) | needs official DEIM repo | **none declared** | no |
| `Intellindust/DEIMv2_DINOv3_X_COCO` | nested non-HF (same) | needs official DEIM repo | **none declared** | no |
| `carpedm20/DEIMv2` | no `config.json` (raw `.pth`/`.onnx`/`.engine`) | needs official DEIM repo or raw ONNX | **Apache-2.0** | no |

## Why in-process HF AutoModel is INFEASIBLE this sprint

1. **Cached configs are non-HF.** The Intellindust checkpoints use a nested
   `PyTorchModelHubMixin` config with no `model_type`/`auto_map` → `AutoModel`
   cannot resolve them.
2. **Native `deimv2` is incomplete in this transformers.** transformers 5.10.2
   ships a `deimv2` module (`Deimv2ForObjectDetection`), **but** `deimv2` is
   absent from `configuration_auto.py`'s `CONFIG_MAPPING` (so `AutoConfig` can't
   resolve it), and its `Deimv2Config` expects a **flat** HF schema — a different
   format from the cached nested mixin configs. The cached checkpoints are not
   loadable by native HF.
3. **torch-pin conflict.** The upstream `Intellindust-AI-Lab/DEIMv2` loader pins
   `torch==2.5.1`; the host runs `torch 2.11.0`.
4. **RT-DETRv4** is released on **GitHub/Google-Drive only — never on HF**; no
   `hf_repo` exists, weights need manual `gdown`. Apache-2.0 (v4 unverified).

## License caveat (must resolve before any commercial exposure)

The `DEIMv2_DINOv3_*` checkpoints embed a **DINOv3 backbone**, whose upstream
license is restrictive/research-leaning. The Apache-2.0 / "no license" metadata
covers the DEIM wrapper, **not** the DINOv3 weights. Do not expose these as
commercial-safe until the DINOv3-backbone license is cleared.

## Feasible next steps (not done — sidecar/isolated-env work)

- **carpedm20 DEIMv2 ONNX** (Apache-2.0) via `onnxruntime` — the only path that
  avoids both the custom loader and the torch-pin conflict; needs the `.onnx`
  artifact pulled (not in the local cache, only `.log` files are).
- **DEIMv2 / RT-DETRv4 sidecar** — the scaffolding already exists
  (`sidecars/manager.py` `SidecarSpec`s for `deimv2` + `rtdetrv4`,
  `sidecars/{deimv2,rtdetrv4}_normalize.py`, `cli/{deimv2,rtdetrv4}_commands.py`).
  Needs the clone + isolated `torch==2.5.1` env wired and smoke-tested; RT-DETRv4
  additionally needs manual `gdown`.

# DEIM / DEIMv2 / RT-DETRv4 final attempt (v3.20)

Confirmed v3.19 findings; no in-process path. All 14 stay `CUSTOM_LOADER_REQUIRED`
(see `custom_loader_live_matrix.json`). No representative operationalized — every
candidate path is blocked by a reproduced conflict:

- **In-process HF AutoModel:** the cached Intellindust `DEIMv2_DINOv3_*` configs are
  nested non-HF (`PyTorchModelHubMixin`, no `model_type`/`auto_map`) → `AutoModel`
  cannot resolve them. transformers 5.10.2 ships a `deimv2` module but it is **absent
  from `configuration_auto.py`'s CONFIG_MAPPING** and expects a different flat schema.
- **Official upstream loader:** `Intellindust-AI-Lab/DEIMv2` pins `torch==2.5.1`; the
  host runs torch 2.11 → must be an isolated env (sidecar), not base.
- **RT-DETRv4:** GitHub/Google-Drive only, no HF repo, manual `gdown`.
- **License:** carpedm20 DEIMv2 Apache-2.0; Intellindust repos declare **no license**
  (registry flags `license_uncertain`). All `DEIMv2_DINOv3_*` embed a **DINOv3 backbone**
  whose license is restrictive — **must not** be exposed commercial-safe until cleared.

## Exact next work
- carpedm20 DEIMv2 **ONNX** (Apache-2.0) via `onnxruntime` — the only path avoiding both
  the custom loader and the torch-pin conflict; needs the `.onnx` artifact pulled.
- DEIMv2-DINOv3 / RT-DETRv4 via an isolated `torch==2.5.1` sidecar
  (`sidecars/manager.py` already scaffolds it); clear the DINOv3-backbone license first.

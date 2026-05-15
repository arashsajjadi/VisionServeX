# Model zoo

The bundled registry (`visionservex list-models`) is authoritative. This
page is a curated, beginner-friendly overview.

## Beginner table

| Model ID                | Task                  | Difficulty | Auto-DL | CPU? | Recommended VRAM | Status         | Wired? |
| ----------------------- | --------------------- | ---------- | ------- | ---- | ---------------- | -------------- | ------ |
| mock-detect             | detect                | very_easy  | yes     | yes  | n/a              | stable         | wired  |
| mock-classify           | classify              | very_easy  | yes     | yes  | n/a              | stable         | wired  |
| mock-segment            | segment               | very_easy  | yes     | yes  | n/a              | stable         | wired  |
| grounding-dino-tiny     | open-vocab detect     | easy       | yes     | yes  | 4 GB             | beta           | wired  |
| grounding-dino-swin-t   | open-vocab detect     | easy       | yes     | yes  | 6 GB             | beta           | wired  |
| grounding-dino-swin-b   | open-vocab detect     | medium     | yes     | no   | 12 GB            | beta           | wired  |
| rfdetr-nano             | detect                | very_easy  | no      | yes  | 1.5 GB           | experimental   | stub   |
| rfdetr-small            | detect                | easy       | no      | yes  | 3 GB             | experimental   | stub   |
| dfine-n                 | detect                | very_easy  | no      | yes  | 1 GB             | experimental   | stub   |
| dfine-s                 | detect                | easy       | no      | yes  | 2 GB             | experimental   | stub   |
| swinv2-tiny             | classify              | very_easy  | no      | yes  | 1 GB             | experimental   | stub   |
| swinv2-small            | classify              | easy       | no      | yes  | 2 GB             | experimental   | stub   |
| sam2-hiera-tiny         | foundation-segment    | medium     | no      | yes  | 4 GB             | experimental   | stub   |
| rtmpose-s               | pose                  | medium     | no      | yes  | 2 GB             | manual         | stub   |
| oneformer-swin-large    | segment               | hard       | no      | yes  | 12 GB            | experimental   | stub   |
| grounded-sam2           | grounded segment      | hard       | no      | no   | 12 GB            | experimental   | stub   |

"Wired" means **this build runs the real backend** when its dependencies
are installed. "Stub" means the registry entry is in place but the real
backend is not yet implemented — calls will raise a clear error (and only
fall back to the mock engine if you set
`VISIONSERVEX_MODELS__ALLOW_MOCK_FALLBACK=true`).

## Which model should I start with?

- **I just want detection**: `mock-detect` first, then `rfdetr-nano` once
  the RF-DETR backend lands.
- **I want detection by text**: `grounding-dino-tiny` (wired today).
- **I have a weak laptop / CPU only**: stick with `mock-*` and
  `grounding-dino-tiny` at small image sizes.
- **I have an NVIDIA GPU**: `grounding-dino-swin-b` for open-vocab;
  experiment with `sam2-hiera-tiny`/`small`.
- **I want segmentation**: `rfdetr-seg-small` (stub) for now; consider
  `oneformer-swin-large` via HF.
- **I want prompt-driven segmentation**: `sam2-hiera-tiny` for free-form,
  or `grounded-sam2` once the composed pipeline ships.
- **I want pose**: `rtmpose-s` via the OpenMMLab toolchain (manual install).
- **I want OBB**: `rtmdet-r2-s` — treat as expert/experimental, no
  benchmark winner claims.
- **I want classification**: `swinv2-tiny` (HF) or `swinv2-small`.

## Least painful path

1. `pip install 'visionservex[server]'`
2. `visionservex doctor`
3. `visionservex recommend --task detect --simple`
4. `visionservex pull mock-detect`
5. `visionservex run-example detect`
6. `visionservex serve`
7. Only then explore advanced backends.

## License-uncertain models

The following entries set `license_uncertain=true`. Verify upstream before
commercial use:

- `co-dino-inst-vit-l-coco`, `co-dino-inst-vit-l-lvis`
- `rtmdet-r-*`, `rtmdet-r2-*`
- `internimage-*`
- `seem-*`
- `grounding-dino-1.5`, `grounding-dino-1.6` (also `external`)

See [`model_licenses.md`](model_licenses.md) for the per-model table.

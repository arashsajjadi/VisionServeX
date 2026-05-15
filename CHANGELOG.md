# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-05-15

### Added
- **D-FINE real backend** via HF Transformers (`ustc-community` checkpoints).
  Model IDs: `dfine-n`, `dfine-s`, `dfine-m`, `dfine-l`, `dfine-x`.
  Uses `AutoModelForObjectDetection` with `d_fine` model type. Status: beta/wired.
- **SAM 2 via HF Transformers** (`Sam2Model` / `Sam2Processor`). Model IDs:
  `sam2-hiera-tiny/small/base-plus/large`. Supports point and box prompts. No
  CUDA extension build required. Status: beta/wired.
- **OneFormer universal segmentation** via HF Transformers. Model IDs:
  `oneformer-swin-large`, `oneformer-dinat-large`, `oneformer-convnext-large`.
  Supports `semantic`, `instance`, and `panoptic` tasks. Status: beta/wired.
- New engine files: `engines/dfine.py` (rewritten), `engines/sam2_hf.py`,
  `engines/oneformer.py`. Registered engines: `dfine`, `sam2_hf`, `oneformer`.
- Tests in `tests/test_phase_h_backends.py` (16 tests: registry + real smoke tests).
- README rewritten as a current-state product document (not a changelog).
- Version bumped to 0.4.0.

### Changed
- D-FINE registry entries (`dfine-*`) updated to `download_type: huggingface`,
  `hf_repo_id: ustc-community/*`, `implementation_status: wired`, `status: beta`.
- SAM2 registry entries updated to `engine: sam2_hf`, `hf_repo_id: facebook/sam2-hiera-*`,
  `implementation_status: wired`, `status: beta`.
- OneFormer registry entries updated to `engine: oneformer`,
  `implementation_status: wired`, `status: beta`.
- `dfine[server]` extra renamed to `dfine[hf]` in registry metadata.

### Fixed
- SAM2 box prompt nesting level: boxes use 3-level nesting, points use 4-level.

## [0.3.0] - 2026-05-15

### Added (Pass 3 through Pass 7)
- **RF-DETR real backend** (detection + segmentation) via the `rfdetr` PyPI
  package. Model IDs: `rfdetr-nano`, `rfdetr-small`, `rfdetr-base`,
  `rfdetr-medium`, `rfdetr-large`, `rfdetr-seg-nano`, `rfdetr-seg-small`,
  `rfdetr-seg-medium`. Status: beta/wired.
- **SwinV2 real classification backend** via HF Transformers
  (`AutoModelForImageClassification`). Model IDs: `swinv2-tiny` through
  `swinv2-large`. Status: beta/wired.
- **SAM v1 real backend** via HF Transformers (`SamModel`, `SamProcessor`).
  Model IDs: `sam-vit-base`, `sam-vit-large`, `sam-vit-huge`. Supports point
  and box prompts. Status: beta/wired.
- **Grounded SAM composed pipeline** (`grounded-sam`) combining Grounding
  DINO Tiny + SAM v1 Base for text-prompted segmentation. Status: beta/wired.
- **Grounding DINO fp16 fix**: cast float tensors to model dtype before
  forward pass; integer token tensors are not cast. Fallback-to-fp32 on
  dtype errors.
- `package_managed` download type for models that manage their own cache
  (RF-DETR). `is_downloadable()` includes this type.
- New engine files: `engines/rfdetr.py`, `engines/swinv2.py`,
  `engines/sam_hf.py`, `engines/grounded_sam.py`.
- New registry entries: SAM v1 variants, Grounded SAM pipeline.
- Tests: `test_rfdetr_engine.py`, `test_new_backends.py` with `@real_model`
  and `@gpu` marks registered in `pyproject.toml`.
- Version bumped to 0.3.0.

### Changed
- RF-DETR and RF-DETR-Seg registry entries updated to `download_type:
  package_managed`, `implementation_status: wired`, `status: beta`.
- SwinV2 registry entries updated to `engine: swinv2`, `implementation_status:
  wired`, `status: beta`.
- SAM 2 entries (`sam2-hiera-*`) remain `stub` / `experimental` with improved
  warning: "Use `sam-vit-base` instead."
- `grounded-sam` added as wired alternative to `grounded-sam2` (which needs
  the sam2 package not on PyPI).

## [0.2.0] - 2026-05-15

### Added (Pass 2)
- Grounding DINO real backend (wired via HF Transformers).
- First-class downloader (HF, GitHub, direct URL, resume, SHA-256).
- Job system for async model downloads.
- Recommendation engine and `recommend` CLI command.
- Beginner-friendly `doctor` command (system, devices, deps, recommendations).
- `devices`, `pull-easy`, `pull-recommended`, `pull-all`, `cache verify/repair`
  commands.
- Auto-pull config, job-mode server prediction.
- Expanded registry (63 models with difficulty/vram/download metadata).
- Beginner examples (10 scripts), synthetic sample images.
- Author attribution (Arash Sajjadi, University of Saskatchewan).
- CITATION.cff, NOTICE.

## [0.1.0] - 2026-05-15

### Added
- Initial public scaffold.
- Model registry with permissive-license-first defaults.
- `VisionModel` high-level Python API with stable result schemas.
- FastAPI server with `/health`, `/ready`, `/version`, `/models`,
  `/predict`, `/detect`, `/segment`, `/pose`, `/classify`,
  `/open-vocab/detect`, `/grounded-segment`, and `/metrics`.
- Typer CLI: `doctor`, `list-models`, `info`, `pull`, `cache`, `predict`,
  `serve`, `tunnel`, `benchmark`, `export`, `config`.
- Security middleware: API key auth, rate limit, body-size limit,
  image validators, SSRF guard, log redaction.
- Cloudflare Tunnel integration via external `cloudflared` binary, with a
  safe ingress config generator that includes a catch-all 404.
- `MockEngine` for tests and engine stubs for D-FINE, RF-DETR, SAM 2,
  Grounding DINO with actionable install hints.
- LRU model cache with VRAM-aware lazy loading.
- Docker (CPU) image and `docker-compose.yml` with optional cloudflared
  sidecar.
- Documentation covering installation, quickstart, model zoo, tasks,
  Cloudflare Tunnel, security, deployment, performance, troubleshooting,
  model licenses, and an LLM-agent guide.

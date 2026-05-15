# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.8.0] - 2026-05-15

### Added — Local Model Gateway
- **`visionservex gateway start/status/doctor/profile/preload/client-example/openapi`** —
  new `gateway` CLI sub-app for local model gateway management.
- **`visionservex suite pull/list`** — pull curated model suites (beginner, gpu-demo,
  server-demo, detection, segmentation, classification).
- **`visionservex pull-suite SUITE`** — top-level alias for quick suite downloads.
- **`visionservex scheduler profile/recommend`** — model-aware concurrency policy
  inspection. dfine-n → queue_recommended (max_concurrency=1); swinv2-tiny →
  acceptable_parallelism (max_concurrency=2); all GPU-exclusive models documented.
- **`visionservex.Client`** — synchronous Python HTTP client for the local gateway
  with `detect`, `classify`, `segment`, `open_vocab_detect`, `grounded_segment`,
  `pose`, `pull`, `load`, `unload`, `warmup`, `job_status`, `poll_job` methods.
  Retries on 503 SERVER_BUSY.
- **`/gateway/status`** — server endpoint reporting loaded models, device, queue, jobs.
- **`/gateway/warmup`** — batch model preload endpoint.
- **SSE job events** — `/jobs/{id}/events?sse=true` now streams Server-Sent Events
  polling until the job reaches a terminal state.
- **Gateway profiles** — `laptop`, `gpu-workstation`, `cpu-safe`, `public-tunnel-safe`
  with per-profile env var sets.
- `docs/local_gateway.md` — full gateway usage guide.
- `examples/client/local_gateway_quickstart.py` and `curl_gateway.sh`.
- 18 new tests in `tests/test_v080.py` (174 total passing).
- `Client`, `ClientResult`, `GatewayError` exported from `visionservex` top-level.

## [0.7.0] - 2026-05-15

### Added
- **CUDA runtime fix** — `_patch_nvrtc_ld_path()` in `device.py` automatically
  adds `/usr/local/lib/ollama/mlx_cuda_v13` (and other known CUDA 13 dirs) to
  `LD_LIBRARY_PATH` so `libnvrtc-builtins.so.13.0` is found by PyTorch CUDA 13
  wheels without manual env setup.
- **CUDA verification** — All 6 wired model families verified on CUDA (RTX 5080,
  PyTorch 2.11.0+cu130): rfdetr-nano, dfine-n, swinv2-tiny, sam2-hiera-tiny,
  grounding-dino-tiny, grounded-sam2.
- **GPU benchmark matrix** — 12 benchmark runs (6 models × 2 devices): all
  CUDA models pass with 5–20× GPU vs CPU speedup.  Saved to
  `reports/v070_cuda_benchmark_matrix.json`.
- **Real parallel tests on CUDA** — dfine-n and swinv2-tiny tested at
  concurrency=2 on cuda; scheduler protects throughput via per-model semaphore.
- **fp32 auto-precision policy** — `precision='auto'` now always defaults to
  `fp32` instead of `fp16` to prevent embedding-table dtype mismatches in
  models with text encoders (Grounding DINO, SAM2, etc.).
- **OpenMMLab Docker sidecar engine** (`openmmlab_sidecar`) — new engine that
  proxies requests to the OpenMMLab FastAPI sidecar container. RTMPose-s and
  RTMDet-R2-s updated to `engine: openmmlab_sidecar, status: experimental`.
- `docker/openmmlab/sidecar_app.py` — internal FastAPI sidecar service for
  `/health`, `/models`, `/predict/pose|obb|segment|classify`.
- 4 GPU tests in `tests/test_v070.py` marked `@pytest.mark.gpu`; pass with
  `VISION_SERVEX_RUN_GPU_TESTS=1`.
- 16 new unit/integration tests in `tests/test_v070.py` (156 total passing).

### Fixed
- Grounding DINO CUDA fp16 crash: `precision='auto'` now returns `fp32`
  universally; explicit `precision='fp16'` still respected.
- `test_cache_verify_returns_report` leaked a test-only registry entry causing
  subsequent `downloads audit` tests to fail; added cleanup.
- `test_sidecar_health_false_for_unreachable` was directly mutating env without
  monkeypatch — fixed to use monkeypatch for clean state.

## [0.6.0] - 2026-05-15

### Added
- **`visionservex gpu smoke-test`** — runs a real end-to-end prediction on
  every listed model on the specified device; reports cold-load time, warm
  inference latency, selected device, precision, backend.
- **`visionservex gpu doctor`** — CUDA diagnostics with actionable fix
  suggestions (driver mismatch, libnvrtc, LD_LIBRARY_PATH, Docker path).
- **`visionservex benchmark-matrix`** — latency matrix over ≥1 model × ≥1
  device combination; JSON output; table summary.
- **`visionservex parallel-test`** — concurrency test with slowdown % and
  status: `excellent_parallelism` / `acceptable_parallelism` /
  `scheduler_needs_queueing` / `protected_throughput`.
- **`visionservex benchmark benchmark-server`** — HTTP load test at multiple
  concurrency levels against a running server.
- **`visionservex downloads audit`** — scans all 68 registry entries and
  reports missing required/recommended metadata.
- **`visionservex openmmlab doctor/docker-build/docker-run/status/smoke-test/list`**.
- **`visionservex tensorrt doctor/build/benchmark`** — TensorRT dry-run and
  real build when `trtexec` is available.
- **`/metrics/prometheus`** endpoint — standard Prometheus text-format
  scrape endpoint with request counters, latency quantiles, gauge for loaded
  models and active requests.
- `device_helpers.py` — shared helpers: `select_dtype`, `move_inputs_to_device`
  (never casts integer token tensors to fp16), `safe_model_to_device`,
  `device_is_available`.
- `exports/` and `reports/` added to `.gitignore`.
- 15 new tests in `tests/test_v060.py`.

### Changed
- `pyproject.toml`: `project.urls.Homepage` etc. still placeholder; version 0.6.0.
- CLI: `benchmark` sub-app wired as `visionservex benchmark`; top-level
  `benchmark-matrix`, `parallel-test`, `mps`, `downloads-audit` aliases added.

### Fixed
- `gpu smoke-test --json` no longer prints "Best device:" text before the
  JSON array.
- `benchmark_server` closure-over-loop-variable fixed (B023).
- All ruff lint errors resolved including B904 (per-file ignores for
  `cli/*` and `server/*` with rationale comments).

## [0.5.1] - 2026-05-15

### Fixed (CI fix pass)
- **Lint**: applied `ruff check . --fix` + `ruff format .`; fixed 165 auto-fixable
  violations; manually fixed F821 (`_log` undefined in `grounding_dino.py`, `Path`
  undefined in `swinv2.py`), B017 (blind except → specific exception), B904 (added
  per-file ignore for `server/` where exception chaining is intentionally omitted),
  B008 (Typer/FastAPI argument defaults, per-file ignore for `cli/` and `server/`),
  SIM102 (combined nested ifs).
- **Test markers**: `conftest.py` now skips `@pytest.mark.real_model`, `@pytest.mark.gpu`,
  and `@pytest.mark.slow` tests unless `VISION_SERVEX_RUN_REAL_MODEL_TESTS=1` or
  `VISION_SERVEX_RUN_GPU_TESTS=1` is set. This fixes the CI matrix failure caused by
  OneFormer tests trying to download weights.
- **OneFormer scipy**: `scipy>=1.10` added to `[project.optional-dependencies].hf`
  since `OneFormerLoss` requires scipy via HF Transformers.
- **CI workflow**: `pip install -e ".[dev,server,hf]"` in test matrix (was `[dev,server]`);
  `VISION_SERVEX_RUN_REAL_MODEL_TESTS` not set, so real model tests are correctly skipped.
- **`engines/__init__.py`**: added `# ruff: noqa: F401` file-level directive; rewrote
  to single-import-per-line style to survive ruff auto-formatting.
- **Git hygiene**: removed stale `outputs/swinv2-tiny.onnx.data` file.

## [0.5.0] - 2026-05-15

### Added
- **Grounded-SAM2 pipeline** (`grounded-sam2`) — composes Grounding DINO Tiny
  + SAM2 Tiny (both via HF Transformers). Status: beta/wired. Auto-download: yes.
- **Device sanity checks** — each CUDA device is now validated with a tiny
  tensor allocation before being selected. Broken CUDA runtimes fall back to
  CPU with an explicit `sanity_ok=False` flag and human-friendly error.
- **Multi-GPU support** — among all CUDA GPUs, the one with the highest free
  VRAM (passing sanity) is selected automatically.
- **`visionservex devices --benchmark`** — runs a synthetic matrix-multiply
  benchmark on all healthy devices and reports GFlops.
- **SwinV2 ONNX export** — `visionservex export swinv2-tiny --format onnx`
  produces a valid 2.8 MB ONNX file (opset 17, dynamic batch, checker passes).
- **`Retry-After` header** on `503 SERVER_BUSY` responses; configurable via
  `VISIONSERVEX_RUNTIME__SERVER_BUSY_RETRY_AFTER_S`.
- **Enhanced benchmark command** — `--warmup`, `--runs`, `--device` flags;
  shows cold-load time, warm p50/p90/p99, throughput estimate.
- **OpenMMLab Docker expert path** — `docker/openmmlab/Dockerfile` and
  `docker-compose.yml` for RTMPose, RTMDet-R/R2, Co-DINO-Inst, InternImage.
- New docs: `docs/device_selection.md`, `docs/concurrency.md`,
  `docs/export.md`, `docs/tensorrt.md`, `docs/openmmlab_expert_models.md`.
- `RuntimeConfig` gains `max_global_concurrency`, `prefer_fastest_device`,
  `allow_device_fallback`, `require_gpu`, `min_free_vram_gb`, `gpu_sanity_check`,
  `server_busy_retry_after_s`, `busy_status_code`.
- All placeholder repo URLs updated to `github.com/arashsajjadi/VisionServeX`.
- Version bumped to 0.5.0; CITATION.cff updated.

### Changed
- `grounded-sam2` registry entry updated to `engine: grounded_sam2`,
  `implementation_status: wired`, `status: beta`, `auto_download: true`.
- Device selection in `device.py` now probes all CUDA devices and picks
  the one with most free VRAM; CUDA sanity check runs automatically.

### Fixed
- Bench improvements: warmup runs excluded from latency measurements.
- `Retry-After` HTTP header now included in all `503 BUSY` responses.

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

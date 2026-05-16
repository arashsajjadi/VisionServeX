# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.8.1] - 2026-05-16

### Patch — fast-CI compatibility for v1.8.0 OWLv2/Florence-2 mocked tests

Three new tests added in v1.8.0 imported `torch` at the function-body level:
`test_owlv2_predict_with_mocked_outputs`, `test_owlv2_accepts_comma_separated_prompt`,
and `test_florence2_unknown_task_raises`. Because the fast CI environment
deliberately omits the `[hf]` extra (to keep wall-time under 10 minutes), torch
is not installed there, and these three tests failed with `ModuleNotFoundError`.

This patch wraps each `import torch` in `pytest.importorskip("torch", reason=...)`
so the tests skip cleanly when torch is absent. No production code changed; the
OWLv2 and Florence-2 engines added in v1.8.0 remain fully wired.

## [1.8.0] - 2026-05-16

### OWLv2 + Florence-2 runnable engines, SAM3 auth wrapper, expert-sidecar dry-run commands

Real model-capacity expansion on top of the v1.7.x test/CI infrastructure.
Two model families that had been declared as stubs since v1.6.0 are now
genuinely wired: OWLv2 for open-vocabulary detection and Florence-2 for
multi-task VLM. A SAM3 auth-aware wrapper exposes structured access errors
for the gated facebookresearch release. New `visionservex expert *` and
`visionservex sam3 *` CLI groups make heavy sidecar and gated-model
workflows explicit. **No model was fake-wired**: every previously-stub
model in this prompt is either now runnable, has an exact auth/install
recipe, or remains audit-only with a documented blocker.

#### OWLv2 — open-vocabulary detection (new — `engines/owlv2.py`)
- `Owlv2Processor` + `Owlv2ForObjectDetection` via HF Transformers.
- Accepts prompts as a list or comma-separated string.
- Returns `OpenVocabularyResult` with one `Detection` per matched query.
- Threshold + post_process_object_detection wired correctly.
- Models: `owlv2-base-patch16`, `owlv2-large-patch14`.
- Registry: `implementation_status="wired"`, `engine="owlv2"`,
  `auto_download=false` (explicit pull required — first-run safety).

#### Florence-2 — multi-task VLM (new — `engines/florence2.py`)
- `AutoProcessor` + `AutoModelForCausalLM` with `trust_remote_code=True`.
- Task-to-token mapping for: caption, detailed_caption, more_detailed_caption,
  object_detection, dense_caption, phrase_grounding, ocr, region_ocr.
- Generated-string parser (`parse_florence2_generation`) extracts boxes +
  labels for box-producing tasks and text for caption/OCR; handles
  `bboxes`, `boxes`, `quad_boxes` shapes across upstream versions.
- Models: `florence-2-base`, `florence-2-large`.
- Registry: `implementation_status="wired"`, `engine="florence2"`,
  `auto_download=false`.

#### SAM3 / SAM3.1 — auth-aware wrapper (new — `cli/sam3_commands.py`)
- `visionservex sam3 status [--model MODEL] [--json]` — structured snapshot:
  HF token presence (redacted, never logs the full token), transformers
  installed, sam3 repo installed, checkpoint cached, blocker code, fix.
- `visionservex sam3 login-help` — exact authentication recipe.
- `visionservex sam3 supported-prompts` — honestly reports zero wired
  prompt types in v1.8.0; users should use the upstream facebookresearch/sam3
  repo directly for inference until the engine is implemented.
- Structured error codes: `HF_AUTH_REQUIRED`, `MODEL_ACCESS_GATED`,
  `SAM3_REPO_REQUIRED`, `CHECKPOINT_REQUIRED`, `PROMPT_TYPE_UNSUPPORTED`.
- **No fake SAM3 inference path.** No mock fallback for gated weights.

#### Expert sidecars (new — `cli/expert_commands.py`)
- `visionservex expert list` — all sidecars with current install state.
- `visionservex expert install <id> [--dry-run]` — prints exact install
  recipe; **dry-run is the default** and `subprocess` is never called.
- `visionservex expert doctor` — multi-framework dependency check.
- Sidecars covered: `openmmlab`, `mmdet`, `mmrotate`, `mmpose`,
  `detectron2`, `maskdino`, `co-detr`.
- Structured error codes per sidecar: `OPENMMLAB_REQUIRED`, `MMDET_REQUIRED`,
  `MMROTATE_REQUIRED`, `MMPOSE_REQUIRED`, `DETECTRON2_REQUIRED`,
  `MASKDINO_REQUIRED`, `CO_DETR_REQUIRED`.

#### Manifest accuracy (corrected from v1.7.0)
- `florence-2-base/large` and `owlv2-base-patch16/large-patch14` were
  honestly downgraded to `runnable_in_visionservex=False` in v1.7.0
  because no engine existed. v1.8.0 flips them back to `True` because
  the engines now exist and tests verify the parsers.
- Registry YAML for those four entries: `engine` switched from generic
  `huggingface` to dedicated `owlv2` / `florence2`; `backend` switched
  to `huggingface_owlv2` / `huggingface_florence2`;
  `implementation_status` → `wired`; `status` → `beta`.

#### Tests (new — `tests/test_v180.py`, 20 tests)
- `test_owlv2_engine_registered` — factory + registry wiring.
- `test_owlv2_predict_with_mocked_outputs` — full inference path with
  mocked HF processor/model returning fake boxes/scores/labels.
- `test_owlv2_accepts_comma_separated_prompt` — CLI shape compatibility.
- `test_florence2_engine_registered`.
- `test_florence2_parse_caption_text` — text-only task parsing.
- `test_florence2_parse_object_detection_bboxes` — OD box+label parsing.
- `test_florence2_parse_phrase_grounding`.
- `test_florence2_parse_quad_boxes_to_axis_aligned` — OCR polygon to
  axis-aligned bbox conversion.
- `test_florence2_unknown_task_raises` — clean ValueError.
- `test_florence2_task_token_mapping` — token table sanity.
- `test_sam3_status_without_token` — HF_AUTH_REQUIRED behavior.
- `test_sam3_status_redacts_token` — never exposes full token.
- `test_sam3_status_short_token_redacted`.
- `test_sam3_models_mapping`.
- `test_sam3_supported_prompts_returns_empty` — no fake predict()/infer().
- `test_expert_list_includes_required_sidecars`.
- `test_expert_install_dry_run_does_not_execute` — patches subprocess
  to fail if any install command is actually run.
- `test_expert_module_missing_returns_structured_code`.
- `test_expert_install_commands_reference_official_tools` — only
  pip/mim/git/cd/python/comment lines allowed.
- `test_manifest_florence_and_owlv2_runnable_again` — manifest tracks the
  new wired state.

#### Updated regression tests
- `tests/test_v160.py::test_florence_in_registry` updated: now asserts
  `implementation_status == "wired"` and `engine == "florence2"`.

#### Validation
- Quick suite: **592 passed, 37 deselected — ~31 s** (was 572 in v1.7.1).
- Ruff lint: clean. Ruff format: clean (162 files).
- Artifact hygiene: zero tracked weights/reports/indexes.

#### What did NOT land in v1.8.0 (honest)
- **Minimal surveillance-search pipeline**: not implemented in this pass.
  The Phase 3 recipe in this prompt would require ~1500 lines of new
  code (video sampler + simple tracker + embedding index + timeline
  exporter). Recommend as primary v1.9.0 feature.
- **Anomalib PatchCore optional extra**: not implemented. The PatchCore
  pipeline needs train/predict/heatmap commands + an `[anomaly]` extra in
  `pyproject.toml`. Recommend for v1.9.0.
- **OpenMMLab/Detectron2 actual `smoke-test` over heavy frameworks**: the
  *commands* are wired but the smoke-test paths still require the heavy
  frameworks to be installed by the user. The `expert install --dry-run`
  output gives the exact recipe.
- **DEIMv2 / RT-DETRv4 native loaders**: still blocked upstream
  (HF Transformers issue #41211). Manifest blocker text unchanged.
- **Medical / agriculture / aerial extras**: domain-zoo recipes exist
  from v1.6.0; no new runnable wiring this pass.

## [1.7.1] - 2026-05-16

### Patch — fast CI compatibility fix

`tests/test_v050.py::test_device_benchmark_cpu` now uses `pytest.importorskip("torch")` so it skips cleanly when torch is not installed (e.g. in the new fast-CI environment that intentionally omits `[hf]` to keep wall-time under 10 minutes). No production code changed. All v1.7.0 features remain intact.

## [1.7.0] - 2026-05-16

### Resource guard, dev safety commands, fast test strategy, model health report

Adds a central resource guard that prevents RAM/VRAM/disk exhaustion during
testing and development; new `visionservex dev *` and `visionservex models
health` CLI subcommands; opt-in real_model/gpu/benchmark smoke test modes; a
pytest lockfile that blocks concurrent test runs; and a split CI workflow
(fast on every push, full only on tag/dispatch).

#### Motivation
A prior development session ran multiple concurrent background pytest
processes without resource checks. RAM and VRAM were saturated, the SSD
was hit hard, and the desktop GUI froze. This release makes that class of
incident structurally impossible by design rather than by discipline.

#### Resource guard (new — `src/visionservex/runtime/resource_guard.py`)
- Reads system RAM/swap, GPU VRAM, disk free, CPU usage, and the running
  process tree via `psutil` (already a runtime dep).
- `assert_safe_to_start_test()` / `_model_load()` / `_benchmark()` refuse to
  start when thresholds are violated, with explicit fix suggestions.
- Pytest lockfile at `/tmp/visionservex_pytest.lock`: contains PID + command +
  start time; stale locks (dead PID) are auto-cleaned; `pytest_sessionstart`
  acquires, `pytest_sessionfinish` releases.
- Default budgets (env-overridable): 8 GB free RAM, 2 GB free VRAM (desktop
  reserve), 10 GB free disk, RAM usage ≤ 80%.
- `cleanup_after_test()` performs the full GC + CUDA cache flush + IPC
  collect + peak-stats reset sequence after every heavy test.
- **Production CLI is untouched.** `visionservex predict/embed/similarity`
  never call the resource guard; only `dev *` subcommands and pytest do.

#### Developer commands (new — `visionservex dev *`)
- `dev test quick` — quick safe tests (target < 60 s).
- `dev test targeted PATH` — single file/keyword with resource pre-check.
- `dev test full-release` — full suite with pre-check + cleanup.
- `dev test real-smoke [--allow-download] [--model KEYWORD]` — opt-in real
  model smoke tests; sets `VISIONSERVEX_RUN_REAL_MODEL_TESTS=1` internally.
- `dev test gpu-smoke --allow-gpu` — opt-in GPU smoke tests; refuses if
  free VRAM < 1 GB + 2 GB reserve.
- `dev test benchmark-smoke [--out DIR]` — process-isolated benchmarks,
  max 3 images, output goes to tmp dir by default.
- `dev resources` — full resource report.
- `dev kill-tests` — kill pytest processes inside this repo only.
- `dev clean-temp` / `clean-reports` / `clean-cache` / `disk-report`.

#### Model health (new — `visionservex models *`)
- `models health [--runnable-only] [--model KEYWORD]` — per-model report:
  checkpoint cached, can-run-CPU/CUDA, VRAM/RAM requirements, smoke test
  status (passed/failed/not_run/skipped_resource_guard/...), suggested
  next command. Renders as a rich table or `--json`.
- `models health-json` — JSON variant for tooling.

#### Test markers (new — full set)
- Added markers in `pyproject.toml` and `tests/conftest.py`:
  `fast`, `integration`, `slow`, `real_model`, `gpu`, `network`, `sidecar`,
  `release`, `benchmark`, `memory`, `disk_heavy`, `download`, `smoke`.
- All heavy markers are opt-in via `VISIONSERVEX_RUN_*_TESTS=1` env vars.
- Backward-compat: old `VISION_SERVEX_RUN_REAL_MODEL_TESTS=1` /
  `VISION_SERVEX_RUN_GPU_TESTS=1` are still accepted.

#### New smoke test files
- `tests/test_real_model_smoke.py` — D-FINE-S, RF-DETR-small, DINOv2-small,
  SAM2-tiny, D-FINE-S-GPU. All 64×64 synthetic images, all resource-guarded,
  all skip cleanly when checkpoint missing.
- `tests/test_benchmark_smoke.py` — mock-detect/segment benchmarks +
  optional real-model benchmark; max 3 iterations; output ≤ 10 KB.
- `tests/test_resource_guard.py` — 17 tests, all mocked (no real memory
  consumed).
- `tests/test_dev_safety.py` — marker skip behavior, dev command structure,
  cleanup repo-scoping, marker registration in pyproject.toml.

#### Standalone scripts
- `scripts/test_quick_safe.py` — quick safe runner.
- `scripts/test_targeted_safe.py` — targeted runner.
- `scripts/test_release_safe.py` — full release runner with pre-check.
- `scripts/kill_visionservex_tests.py` — repo-scoped pytest killer.
- `scripts/diagnose_resources.py` — full diagnostic with warning list.

#### CI split (`.github/workflows/ci.yml`)
- **Fast CI** (push/PR): ubuntu-latest only, Python 3.12 only. Runs lint,
  format, type-check, security scan, quick pytest (~30 s on GH runners),
  build, twine check, docker build. Timeout 10–20 min per job.
- **Full CI** (release tag `v*` or `workflow_dispatch`): 3 OS × 3 Python
  matrix, full pytest. Timeout 30 min.
- Concurrency cancellation: new pushes on the same branch auto-cancel
  in-progress runs. Old "still running for previous version" CI runs no
  longer block release publishes.

#### Documentation
- New: `AGENT_RULES.md` (concise rule set for AI agents — "Follow this
  strictly" reference).
- New: `docs/agent_safety.md` (full incident context, safety rationale,
  system design).
- README: new "Resource Safety & Developer Commands" section.

#### Manifest accuracy fix
- `florence-2-base/large` and `owlv2-base-patch16/large-patch14` were
  marked `runnable_in_visionservex=True` in the source manifest but are
  `implementation_status="stub"` in the registry (no engine wired). The
  manifest now honestly reports `runnable_in_visionservex=False` with
  explicit `known_blockers` listing what's missing: no engine module,
  prompt-token builder, output parser, etc. **No engine was fake-wired.**

#### Validation
- Quick suite: 572 passed, 37 deselected — **~31 s** (target < 60 s).
- Targeted safety: 26 passed, 5 skipped (heavy markers) — 1.2 s.
- Ruff lint: clean. Ruff format: clean (154 files).
- Artifact hygiene: no `.pt/.onnx/.parquet/...` or `outputs/reports/indexes/`
  contents tracked in git.

#### What did NOT land in this release (honest)
- Florence-2 engine: still stub. Needs HF AutoModelForCausalLM wiring with
  trust_remote_code, task-specific prompt tokens (`<OD>`, `<CAPTION>`,
  `<DENSE_REGION_CAPTION>`, ...), and a generated-string parser.
- OWLv2 engine: still stub. Needs `Owlv2Processor` +
  `Owlv2ForObjectDetection.post_process_object_detection()` wiring + result
  normalization.
- SAM3 / SAM3.1: still gated/external. No auth-aware wrapper yet.
- Surveillance-search pipeline: not implemented in this release.
- Anomalib, RTMDet-R/R2, MedSAM2, TotalSegmentator, Prithvi, AgriCLIP:
  remain audit-only or expert-sidecar per the manifest.

## [1.6.0] - 2026-05-16

### Source-grounded model zoo, DINOv2 feature intelligence, domain-zoo recommender

Adds a link-grounded model manifest, a runnable DINOv2/SigLIP2 feature
backbone with image embedding / retrieval / deduplication / dataset
intelligence, a domain-zoo recommender, and structured registry entries for
Florence-2, OWLv2, SAM3 (gated), and unverified models.

#### Source-grounded model manifest (new)
- Added `src/visionservex/model_zoo/manifest.py` with `ModelSource` dataclass.
- Every model entry cites: official_repo, official_docs, paper_url, hf_repo,
  checkpoint_url, license, license_risk, install_command, hf_class,
  runnable_in_visionservex, access_status (open/api_token/gated), domain,
  known_blockers, recommended_action.
- Initial coverage: D-FINE, RF-DETR, DEIMv2, RT-DETRv4, Co-DINO, MaskDINO,
  SAM/SAM2/SAM2.1/SAM3, DINOv2/DINOv3, Florence-2, OWLv2, SigLIP2, Grounding
  DINO 1.5/1.6/DINO-X, Anomalib, Torchreid/OSNet, ByteTrack, TotalSegmentator,
  MedSAM/MedSAM2, nnU-Net, RTMDet-R, Prithvi, AgriCLIP, YOLO-World.
- New CLI: `visionservex model-zoo sources/verify-links/export/show`.

#### Domain-zoo recommender (new)
- Added `src/visionservex/model_zoo/domain_zoo.py` with `DomainRecipe` dataclass.
- Domains: yolo26-competitors, sam-family, promptable, feature-intelligence,
  surveillance, industrial, medical, agriculture, aerial.
- Each recipe has: pipeline steps, recommended models, install commands,
  quick commands, expected hardware, runnable_today flag, limitations,
  license notes.
- New CLI: `visionservex domain-zoo list/recommend/<domain>/export`.

#### DINOv2 feature backbone (new — runnable!)
- Added `src/visionservex/engines/dinov2.py` — wraps HF AutoModel for
  facebook/dinov2-{small,base,large,giant} and google/siglip2-*.
- New task type: `embed`. Returns `EmbeddingResult` (L2-normalized vector).
- New `embed` task added to Task literal.
- `EmbeddingResult`, `SimilarityResult`, `SearchResult`, `SearchHit`,
  `DatasetReport` result classes in `src/visionservex/core/embedding_results.py`.

#### Embedding runtime (new)
- Added `src/visionservex/runtime/embeddings.py`:
  - `embed_folder()` — batch image embedding.
  - `EmbeddingIndex` — flat numpy index with manifest.json.
  - `search_index()` — top-k nearest neighbor by cosine.
  - `deduplicate_index()` — find pairs above similarity threshold.
  - `build_dataset_report()` — mean similarity, diversity, suggested clusters.
  - `active_learning_select()` — farthest-point sampling on embeddings.
  - `domain_shift_report()` — train/test centroid + mean nearest similarity.

#### Embedding CLI (new top-level commands)
- `visionservex embed MODEL image_or_folder --out path`
- `visionservex similarity MODEL image_a image_b`
- `visionservex index MODEL folder/ --out indexes/name`
- `visionservex search MODEL query.jpg --index indexes/name --top-k 10`
- `visionservex deduplicate MODEL folder/ --threshold 0.98 --out duplicates.csv`
- `visionservex dataset-report MODEL folder/ --out report.md`
- `visionservex active-select MODEL folder/ --budget 100`
- `visionservex domain-shift MODEL train/ test/`
- `visionservex benchmark-embeddings --model MODEL --dataset folder:<path>` — kNN accuracy if labels.csv present.

#### New model registry entries
- `dinov2-small/base/large/giant`: runnable, feature_backbone, Apache-2.0.
- `siglip2-base-patch16-224`: runnable, text-image retrieval.
- `florence-2-base/large`: stub, MIT, prompt-format wiring pending.
- `owlv2-base-patch16` / `owlv2-large-patch14`: stub, Apache-2.0, engine pending.
- `sam3-base`: external_api stub, gated access.

#### Task / category taxonomy extensions
- New `Task` values: `embed`, `vlm`, `anomaly`, `track`, `reid`.
- New `ModelCategory` values: `feature_backbone`, `promptable_foundation`,
  `surveillance_pipeline_component`, `medical_extra`, `industrial_extra`,
  `geospatial_extra`, `agriculture_extra`, `non_core_license_optional`,
  `audit_only`.

### Decisions
- **DEIMv2**: still audit_only (no HF Transformers support per upstream).
- **SAM3/SAM3.1**: external_api stub (gated access at facebook namespace).
- **MaskDINO/Co-DINO**: expert_sidecar (Detectron2/MMDet required).
- **YOLO-World**: do_not_add (license likely GPL/AGPL — excluded from permissive core).
- **TotalSegmentator/MedSAM**: non_core_license_optional (medical/regulatory care).
- **Anomalib/torchreid/ByteTrack**: expert_sidecar (heavy deps, not in core).
- **Florence-2 / OWLv2 engines**: registered as stub. HF Transformers backend
  wiring with task-specific prompts/processors is roadmap v1.7.

### Known limitations
- DINOv2 returns L2-normalized embeddings; do not feed them to detection AP.
- Embedding search uses numpy nearest neighbors (no FAISS dep). For >100k
  images, consider exporting embeddings and using FAISS externally.
- DINOv3 entries remain audit_only — HF model card names not verified live.
- Video search pipeline (surveillance) is recipe-only, not yet wired.

## [1.5.0] - 2026-05-16

### VRAM lifecycle fix, process-isolated benchmarking, real mask AP evaluator

Prevents stepwise VRAM accumulation during repeated model loads and benchmarks.
Adds process-isolated benchmark mode and a real mask AP evaluator for instance
segmentation.

#### VRAM lifecycle fix (Phase 1)
- Added `src/visionservex/runtime/gpu_lifecycle.py` — central GPU memory manager:
  - `get_gpu_memory_state()` — snapshot allocated/reserved/peak VRAM.
  - `get_process_gpu_memory()` — per-process VRAM via nvidia-smi.
  - `clear_torch_cuda_cache()` — synchronize + empty_cache + ipc_collect + reset_peak.
  - `cleanup_gpu_after_model(model)` — full cleanup after a model run.
  - `assert_memory_returned_to_baseline()` — compare memory growth vs threshold.
  - `MemoryState` dataclass with growth arithmetic.
- `VisionModel.unload()` now runs the full cleanup sequence after engine.unload():
  Python GC → CUDA sync → empty_cache → ipc_collect → reset_peak_stats.
- `VisionModel.close()` alias for `unload()`.
- `VisionModel.predict(..., unload_after=True)` — unload after a single prediction.
- `VisionModel.__exit__` now calls `unload()` with full GPU cleanup.
- All benchmark runs default to `--unload-between-models` (GPU cache flushed after each model).

#### New GPU CLI commands (Phase 1)
- `visionservex gpu cleanup-cache` — flush CUDA allocator cache (no process kill).
- `visionservex gpu explain-memory` — show allocated vs reserved with explanation.
- `visionservex gpu memory-test MODEL --runs 5` — check VRAM growth over N runs.
- `visionservex gpu memory-test-suite --models ... --max-growth-mb 512` — multi-model test.
- `visionservex gpu unload-all` — GC + CUDA flush for current process.

#### Process-isolated benchmark (Phase 2)
- `benchmark-competitiveness --isolate-process` runs each model in a child process.
- Child uses `multiprocessing.spawn` context. No CUDA tensors cross process boundary.
- Child exits cleanly, releasing its CUDA context. Parent collects JSON results.
- Protects parent from CUDA OOM in child.

#### Mask AP evaluator (Phase 6)
- Added `src/visionservex/runtime/segmentation_eval.py`:
  - `load_coco_segmentation_json()` — load COCO segmentation JSON with polygon/RLE masks.
  - `MaskDetectionEvaluator` — mask IoU matching, cumulative TP/FP, 101-point AP.
  - `run_segmentation_evaluation(model_id, samples)` — full evaluation runner.
  - `SegEvaluationResult` — mask_ap50, mask_map50_95, box_ap50, latency, n_no_mask.
- `benchmark-segmentation` upgraded from a stub to a real command:
  - Synthetic mode: latency + detection count.
  - COCO JSON mode: real mask AP50 and mAP50:95.
  - `--unload-between-models` (default: on) — flush GPU between models.
  - Results exported as JSON.
- Mask AP uses binary mask IoU (not box IoU). Not comparable to detection mAP.

### Decisions
- **DEIMv2**: checkpoint download path not verified — remains experimental_sota stub.
- **RT-DETRv4**: no official release numbering confirmed — remains experimental_sota stub.
- **Co-DINO/MaskDINO**: expert_sidecar stubs — OpenMMLab/Detectron2 required.
- **RF-DETR-Seg Large/XL/2XL**: HF checkpoints not published — remain unavailable_with_reason.
- **process isolation**: uses multiprocessing.spawn (not fork) for CUDA safety.

### Known limitations
- CUDA allocator retains a reserved pool even after empty_cache. This is expected
  behavior. "reserved" memory (CUDA cache) can be larger than "allocated" (live tensors).
- process-isolated mode is slower per model due to spawn overhead (~5-10s per model).
- Mask AP with polygon GT requires polygon-to-binary conversion; RLE GT requires
  either pycocotools (fast) or manual decode (slower, built-in).

## [1.4.0] - 2026-05-16

### Ultralytics-like ergonomics, output normalizer, model lifecycle CLI

VisionServeX gains Ultralytics-style ergonomics, a robust multi-schema output
normalizer, model lifecycle CLI commands, training/export capability matrices,
task alias commands, and video/tracking stubs.

#### Output normalizer (Phase 1 / Phase 13F)
- Added `src/visionservex/core/normalizer.py` — accepts all common box schemas:
  `xyxy:[...]`, `box:[...]`, `bbox:[...]`, `bbox_format=xywh`,
  `box:{"x1":...}`, `xyxy:{"x1":...}`, `box:{"xmin":...}`, `coordinates:{...}`.
- Accepted score keys: `score`, `confidence`, `conf`, `probability`, `prob`.
- Accepted label keys: `class_name`, `label`, `category`, `name`, `phrase`,
  `class_id`, `category_id`, `label_id`, `cls`.
- COCO official category IDs 1-90 → contiguous 0-79 mapping built-in.
- `AllPredictionsDroppedWarning` emitted if normalization drops all predictions.
- `parse_api_response()` handles the VisionServeX HTTP API JSON format directly.
- Exported at top-level: `from visionservex import normalize_detections, parse_api_response`.

#### Ultralytics-like Python API (Phase 13A/B)
- `VisionModel.from_pretrained()`, `from_registry()` — factory class methods.
- `VisionModel.from_checkpoint()` — returns `CHECKPOINT_LOAD_UNSUPPORTED` structured error.
- `VisionModel.to(device)` — move to device (returns self).
- `VisionModel.pull(force=False)` — download weights.
- `VisionModel.cache_info()` — cache path, size, HF path.
- `VisionModel.checkpoint_info()` — provenance, trust level, AP verification status.
- `VisionModel.clear_cache()` — delete cached weights.
- `VisionModel.names` — COCO80 class names for detection models.
- `VisionModel.supports(operation)` — check predict/val/export/train/track support.
- `VisionModel.training_info()` — per-family training capability dict.
- `VisionModel.export_info()` — per-family export capability dict.
- `VisionModel.val(dataset=..., max_images=...)` — evaluates AP50/mAP50:95 when detection model.

#### Results objects (Phase 13E)
- Added `BaseResult.to_csv()` — CSV-formatted string of predictions.
- Added `BaseResult.to_pandas()` — pandas DataFrame (requires pandas installed).
- Added `BaseResult.debug()` — multi-line debug string with full result details.
- Added `BaseResult.show()` — best-effort image display in notebooks/windows.

#### Model lifecycle CLI (Phase 13D)
- `visionservex model info MODEL` — registry + cache status.
- `visionservex model pull MODEL [--force] [--dry-run]` — download checkpoint.
- `visionservex model checkpoint-info MODEL` — provenance, trust level.
- `visionservex model cache MODEL` — cache size and path.
- `visionservex model verify MODEL` — SHA-256 verification.
- `visionservex model clear-cache MODEL` — delete cached weights.
- `visionservex model list-local` — all locally cached models.

#### Training / export capabilities (Phase 13G/H)
- `visionservex training capabilities [--model MODEL]` — table of train/finetune/resume support.
- `visionservex training train MODEL --data ... --epochs N` — structured TRAINING_NOT_SUPPORTED.
- `visionservex training finetune MODEL --data ... --epochs N` — structured error.
- `visionservex training val MODEL --dataset yolo:<path>` — detection AP evaluation.
- `visionservex export-cmd capabilities [--model MODEL]` — ONNX/TRT/other export status.
- `visionservex export-cmd export MODEL --format onnx --out path` — structured EXPORT_UNSUPPORTED.
- RF-DETR: train_supported=True, finetune_supported=True (rfdetr package).
- All others: train_supported=False with explicit notes and upstream docs link.

#### CLI task aliases (Phase 13C)
- `visionservex detect MODEL IMAGE [--conf 0.25] [--device auto]`
- `visionservex segment MODEL IMAGE [--conf 0.25]`
- `visionservex classify MODEL IMAGE [--top-k 5]`
- `visionservex open-vocab MODEL IMAGE --prompt "car,person"`
- `visionservex grounded-segment MODEL IMAGE --prompt "person"`
- `visionservex val MODEL --dataset yolo:<path>` (detection only)
- `visionservex train MODEL --data ... --epochs N` (structured error for most)
- `visionservex finetune MODEL --data ... --epochs N` (structured error)

#### Video/tracking stubs (Phase 13I)
- `visionservex video predict MODEL SOURCE` — VIDEO_NOT_IMPLEMENTED (exit 2).
- `visionservex video track MODEL SOURCE` — TRACKING_NOT_IMPLEMENTED (exit 2).
- `visionservex video stream MODEL --source webcam` — STREAMING_NOT_IMPLEMENTED (exit 2).
- Roadmap: v1.5.0.

### Decisions
- **Training**: Only RF-DETR has train/finetune=True. All HF Transformers backends (D-FINE,
  SwinV2, Grounding DINO, OneFormer, SAM, SAM2) return TRAINING_NOT_SUPPORTED — HF
  inference API does not expose training. Use upstream repos directly for training.
- **ONNX export**: rfdetr=supported, others=experimental or unsupported.
- **DEIMv2/RT-DETRv4**: still experimental_sota stubs — no change from v1.3.0.
- **Mask AP**: still roadmap v1.5.
- **Video inference**: roadmap v1.5.

### Known limitations
- `VisionModel.val()` only works for detect/open_vocab_detect tasks.
- Training is only semantically supported for RF-DETR (rfdetr package exposes training).
- `to_pandas()` requires `pip install pandas`.
- Video, tracking, and stream operations return structured NOT_IMPLEMENTED errors.

## [1.3.0] - 2026-05-15

### Evaluation and scientific usability upgrade

VisionServeX is upgraded from an accuracy-tagged model gateway to a
**scientifically usable evaluation platform**. Real AP/mAP computation,
structured model cards, an Ultralytics replacement map, a comprehensive
capabilities report, and honest task-specific benchmark stubs are added.

#### Capabilities report (Phase 1)
- Added `visionservex capabilities report` command (human/json/markdown formats).
  Reports: package version, Python, OS, devices, installed extras, model counts
  by task/category, runnable models, unavailable models with reasons, goal-based
  recommendations, security status, and known limitations.
- `--out <file>` writes the report to disk.

#### Model card system (Phase 2)
- Added `visionservex model-card show MODEL_ID` (human/json/markdown).
- Added `visionservex model-card list` and `visionservex model-card export`.
- Explicit supplementary card data for 24 model families including:
  dfine-n/s/m/l/x-o365-coco, rfdetr-nano/small/medium/large,
  rfdetr-seg-nano/small/medium, sam-vit-base, sam2-hiera-tiny,
  grounding-dino-tiny/swin-b, grounded-sam/sam2,
  swinv2-tiny/base, oneformer-swin-large, rtmpose-s, internimage-t.
- Every card includes: recommended_for, not_recommended_for,
  replaces_or_competes_with, hardware requirements, official_benchmark_note,
  visionservex_benchmark_status.
- Demo_fast model cards explicitly warn against using them for AP comparison.
- SAM/SAM2 cards explicitly warn against mixing with detection mAP.

#### Replacement map (Phase 3)
- Added `visionservex replacement-map map` (human/json/markdown, `--task` filter).
- Covers: detect, segment, pose, obb, classify, open-vocab, sam → ultralytics.
- Every replacement entry has ap_claim=false; honest_caveats included.
- Pose/OBB explicitly state no verified winner over YOLO; expert_sidecar required.
- Does not claim 'better' without evidence.

#### Real AP/mAP evaluation (Phase 4)
- Added `src/visionservex/runtime/evaluation.py` with COCO-style 101-point
  interpolated AP computation engine.
- Supports: YOLO-format datasets (images/ + labels/ + data.yaml),
  COCO JSON annotation format, class-aware and class-agnostic matching.
- Metrics: AP50, mAP50:95 (IoU 0.50→0.95 sweep), precision, recall, F1
  (per-class and macro-averaged), latency P50/P95, no-detection count.
- `benchmark-competitiveness --dataset yolo:<path>` activates real AP mode.
- `benchmark-competitiveness --dataset coco-json:<img_dir>:<ann_file>` for COCO JSON.
- Ultralytics baseline (ultralytics:yolo11n) also evaluated with full AP when dataset provided.
- Results saved as JSON + CSV summary when `--out` is specified.
- Honest conclusion: reports which model has best AP50/mAP50:95, warns on small datasets.

#### Debug-output improvements (Phase 5)
- Added `--save-json <file>`: save full diagnostics to JSON.
- Added `--visualize <file>`: save annotated image with detection boxes drawn.

#### Non-detection benchmark stubs (Phase 8)
- Added honest BENCHMARK_NOT_IMPLEMENTED stubs (exit code 2, structured JSON):
  - `visionservex benchmark benchmark-segmentation` (roadmap: v1.4)
  - `visionservex benchmark benchmark-classification` (roadmap: v1.4)
  - `visionservex benchmark benchmark-open-vocab` (roadmap: v1.4)
  - `visionservex benchmark benchmark-pose` (roadmap: v1.4)
  - `visionservex benchmark benchmark-obb` (roadmap: v1.4)
  Each stub reports task, expected annotation format, recommended dataset,
  correct metrics, expected models, and roadmap note.
  Detection AP is the only task currently implemented.

### Decisions
- **Segmentation mask AP**: not implemented. Requires polygon/RLE IoU — roadmap v1.4.
- **Classification top-k**: not implemented — roadmap v1.4.
- **Open-vocab zero-shot AP**: not implemented — roadmap v1.4.
- **OKS/rotated IoU AP**: not implemented — roadmap v1.4.
- **COCO128 auto-download**: not bundled. Users provide dataset path via --dataset.
- **InternImage/Co-DINO/MaskDINO**: still stubs — no change from v1.2.0.

### Known limitations
- `benchmark-competitiveness` AP results depend on class label matching between
  model outputs (strings) and YOLO/COCO GT labels. Mismatches produce 0 AP.
- AP estimates from <100 images have high variance; conclusions warn about this.
- mAP50:95 computation (10 IoU thresholds) is slower than AP50 alone.
- The Ultralytics AP baseline in `benchmark-competitiveness` requires `pip install ultralytics`.

## [1.2.0] - 2026-05-15

### Accuracy-aware model gateway upgrade

VisionServeX is upgraded from a demo-friendly multi-backend gateway into an
**accuracy-aware model gateway**. The registry now carries model taxonomy,
explicit Objects365+COCO model IDs, experimental SOTA candidates with honest
status labels, and competitiveness/debug tooling.

#### Model taxonomy (Phase 1)
- Added `model_category` field to `ModelEntry` with values:
  `demo_fast`, `production_recommended`, `accuracy_grade`,
  `experimental_sota`, `expert_sidecar`, `external_api`,
  `unavailable_with_reason`, `utility`.
- All 87 registry entries carry an explicit `model_category`.
- `dfine-n` / `rfdetr-nano` / `grounding-dino-tiny` / `rfdetr-seg-nano` →
  `demo_fast`. Do not use these as accuracy-grade claims.
- `dfine-s/m/l/x`, `rfdetr-small/medium/large` → `accuracy_grade`.
- `swinv2-base/large`, `sam2-hiera-large`, `oneformer-swin-large` →
  `accuracy_grade` for their respective tasks.

#### D-FINE official checkpoint upgrade (Phase 2)
- Added 9 new model IDs with explicit COCO / Objects365+COCO naming:
  - `dfine-n-coco` — COCO-only Nano, `demo_fast`, same as `dfine-n`.
  - `dfine-s-coco` / `dfine-m-coco` / `dfine-l-coco` / `dfine-x-coco` —
    COCO-only S/M/L/X. Repo availability note added; use o365 variants if
    uncertain.
  - `dfine-s-o365-coco` / `dfine-m-o365-coco` / `dfine-l-o365-coco` /
    `dfine-x-o365-coco` — Objects365+COCO, `accuracy_grade`, wired via
    existing ustc-community HF checkpoints. Recommended for competitiveness
    benchmarks.
- `dfine.py` updated with all new ID → HF repo mappings.

#### RF-DETR model categorisation (Phase 3)
- `rfdetr-nano` / `rfdetr-seg-nano` → `demo_fast` with explicit
  `not_good_for` notes.
- `rfdetr-small` / `rfdetr-seg-small` → `production_recommended`.
- `rfdetr-base/medium/large` / `rfdetr-seg-medium` → `accuracy_grade`.
- `rfdetr-seg-large/xlarge/2xlarge` → `unavailable_with_reason` with honest
  blocker message.

#### Experimental SOTA candidates (Phase 4)
- Added `deim-s`, `deim-m`, `deimv2-s`, `deimv2-m` as `experimental_sota` /
  `stub`. Blockers: no HF path, custom loader required, license pending
  verification.
- Added `rtdetrv4-s/m/l/x` as `experimental_sota` / `stub`. Blockers: no
  verified release numbering, no HF checkpoint confirmed.
- All experimental entries include `unavailable_reason` explaining exact
  blockage.

#### Segmentation upgrade (Phase 5)
- Added `maskdino-r50-coco` and `maskdino-r50-panoptic` as
  `experimental_sota` / `stub` with honest blocker (detectron2 required).
- Co-DINO-Inst → `expert_sidecar`.
- `rfdetr-seg-small` elevated to `production_recommended`.

#### Open-vocabulary upgrade (Phase 6)
- `grounding-dino-swin-b` → `accuracy_grade` with note on stronger accuracy.
- `grounding-dino-1.5` / `grounding-dino-1.6` → `external_api`.

#### Classification taxonomy (Phase 7)
- `swinv2-tiny/small` → `production_recommended`.
- `swinv2-base/large` → `accuracy_grade`.
- InternImage → `expert_sidecar` (DCNv3 custom ops, not pip-installable).

#### Competitiveness benchmark harness (Phase 8)
- Added `visionservex benchmark benchmark-competitiveness` CLI command.
  Compares detection models on latency, detection counts, and output schema
  validity. Supports Ultralytics baseline via `ultralytics:yoloXXX` prefix.
- Generates honest conclusion that reports if YOLO wins.
- Note: AP50/mAP require ground-truth annotations; this tool reports latency
  and detection health, not accuracy.

#### Postprocessing debug tool (Phase 9)
- Added `visionservex debug-output MODEL_ID IMAGE` command.
  Prints: raw keys, normalized detections, score histogram, label histogram,
  first 10 boxes, invalid boxes, unmapped labels, image size, preprocessing
  notes. Diagnose parser/postprocess bugs before blaming the checkpoint.

#### Model recommender update (Phase 10)
- Added `--goal` flag to `visionservex recommend`:
  `accuracy`, `fastest_demo`, `best_open_license`, `best_colab`,
  `best_gpu`, `best_cpu`, `best_segmentation`, `best_open_vocab`.
- For `--goal accuracy --task detect`: surfaces `dfine-s/m-o365-coco` and
  `rfdetr-small/medium`, not nano variants.
- `recommend` UI shows `model_category` column with colour coding.
- `unavailable_with_reason` and `experimental_sota` entries are penalised
  unless `--goal accuracy` explicitly requests them.

### Decisions
- **Real AP50/mAP benchmark**: not implemented. AP requires ground-truth
  COCO annotations. The `benchmark-competitiveness` command reports latency
  and detection health only. Full AP evaluation is out of scope for v1.2.0.
- **DEIM/DEIMv2 real inference**: not wired. Blockers: no HF or pip path
  verified, license and checkpoint availability unclear.
- **RT-DETRv4 real inference**: not wired. RT-DETRv4 is not an officially
  released version number; blocked on checkpoint source and loader.
- **MaskDINO**: not wired. detectron2 environment required; no HF path.

### Known limitations
- D-FINE COCO-only variants (`dfine-s-coco` etc.) point to HF repos that
  may not exist (ustc-community/dfine-small-coco). Use o365 variants for
  guaranteed availability.
- Competitiveness benchmark uses synthetic images; results are latency proxies
  only, not accuracy indicators.
- VisionServeX does not claim to beat Ultralytics globally. The
  `benchmark-competitiveness` tool is designed to reveal the honest truth.

## [1.1.0] - 2026-05-15

### Colab GPU worker mode

VisionServeX can now run as a temporary remote GPU worker on Google Colab.
Marked **optional** and **non-production**. The CLI refuses to expose a tunnel
without auth and explicit user acknowledgement.

### Added
- **`visionservex colab` subgroup** with 10 commands:
  - `colab doctor` — environment + GPU + Drive + auth + cloudflared diagnostic.
    Returns `COLAB_NOT_DETECTED`, `COLAB_GPU_UNAVAILABLE`, or `ok` with safe
    VRAM budget.
  - `colab status` — single-line status.
  - `colab gpu-check` — GPU health + recommended profile.
  - `colab mount-drive` — print exact Drive-mount snippet (cannot mount on
    user's behalf).
  - `colab cache-path` — show recommended cache path (Drive if mounted,
    `/content` otherwise with persistence warning).
  - `colab setup-cache [--drive]` — print exact `VISIONSERVEX_CACHE_DIR`
    env-var setup commands; refuses `--drive` if Drive not mounted.
  - `colab cleanup` — remove Colab session-specific temp files only.
  - `colab token` — generate a one-time API key (URL-safe 32 bytes).
  - `colab tunnel-start --domain <D> --i-understand-this-is-public` —
    refuses without auth, refuses without acknowledgement, refuses without
    `cloudflared` installed. Structured errors: `AUTH_REQUIRED`,
    `EXPOSURE_NOT_ACKNOWLEDGED`, `CLOUDFLARED_MISSING`.
  - `colab tunnel-stop` — SIGTERM to any running cloudflared tunnel process.
  - `colab test-remote <URL> [--api-key K]` — probe `/health` and `/models`
    of a remote worker. Returns `ok`, `AUTH_REQUIRED`, `UNREACHABLE`, or
    `ERROR` with hints.
- **`colab-gpu-worker` gateway profile**:
  - bind: `127.0.0.1`
  - max loaded models: 1
  - per-model concurrency: 1
  - queue size: 4
  - max VRAM fraction: 0.85
  - min free VRAM: 1.5 GB
  - desktop GUI reserve: off (Colab is headless)
  - auto-pull: off
  - retention: `metadata_only`, save_inputs/outputs: false
- **`examples/colab/VisionServeX_Colab_GPU_Worker.ipynb`** — copy-paste Colab
  notebook covering install → diagnose → optional Drive cache → pull suite →
  start gateway → run inference → optional tunnel → cleanup.
- **`examples/colab/colab_quickstart.py`** — Python script form of the
  notebook for non-notebook use.
- **`docs/colab_gpu_worker.md`** — full guide: when to use Colab, profile
  defaults, CLI reference, Drive persistence, secure tunnel exposure rules,
  structured error codes, privacy notes, known limitations.
- **README**: short "Temporary Colab GPU worker" section and docs link.
- **19 new tests in `tests/test_colab_commands.py`** with mocks for Colab
  detection, GPU state, Drive mount, auth, and tunnel safety rules. No tests
  require an actual Colab session.

### Decisions
- **OpenMMLab real inference**: not closed in v1.1.0. The current environment
  has no `mmpose`/`mmdet`/`mmrotate` installed. Status remains
  `docker_checkpoint_required`. `visionservex openmmlab pull <model_id>`
  continues to return `CHECKPOINT_REQUIRED` with official instructions.
- **MPS verification**: not closed. No Apple Silicon hardware available to
  maintainers. Status remains `implemented_unverified`.
- **TensorRT real engine**: not closed. `trtexec` is not on PATH and the
  `tensorrt` Python package is not installed. Status remains
  `experimental/dry-run`. ONNX export for SwinV2 still works as in v1.0.0.
- **Cooperative in-flight cancellation**: not added in v1.1.0. Queued-job
  cancellation continues to work; in-flight inference remains best-effort.

### Known limitations
- Colab support is intentionally minimal. The CLI exposes diagnostics and a
  profile; the user is responsible for the notebook flow. Drive mount and
  cloudflared install are operations the user must perform inside Colab.
- The previous v1.0.0 limitations (OpenMMLab, MPS, TensorRT, in-flight
  cancellation) are unchanged and still honestly documented.

## [1.0.0] - 2026-05-15

### First stable release

**Scope of stable v1.0.0 core:**

The following model families are part of the stable core: Mock (all tasks),
RF-DETR, RF-DETR-Seg (nano/small/medium), D-FINE, Grounding DINO, SwinV2,
SAM v1, SAM 2, Grounded SAM, Grounded-SAM2, and OneFormer. All are `beta`
status or higher, wired via HF Transformers or the rfdetr package.

The following are **explicitly outside the stable v1.0.0 core**:
- OpenMMLab (RTMPose, RTMDet-R/R2, Co-DINO, InternImage): `docker_checkpoint_required`.
  Returns `CHECKPOINT_REQUIRED` structured error — no fake output.
  See `visionservex openmmlab pull <model_id>` for instructions.
- TensorRT: dry-run/experimental. ONNX export works; engine build requires `trtexec`.
- MPS (Apple Silicon): implemented, not maintainer-verified (no test hardware).

### Summary of complete v1.0.0 implementation

**Local gateway and API:**
- Full local HTTP gateway (FastAPI, `visionservex serve`)
- CLI predict, batch-predict, benchmark-matrix, parallel-test
- Python `VisionModel` API — direct inference without gateway
- `Client` and `AsyncClient` for gateway access
- SSE job events, SQLite job store, cancellation for queued jobs

**Supported model families (wired):**
- Mock (8 tasks, stable, CPU-only, no download)
- RF-DETR / RF-DETR-Seg (nano through medium, beta, rfdetr package)
- D-FINE (n/s/m/l/x, beta, HF Transformers)
- Grounding DINO (tiny/swin-t/swin-b, beta, HF Transformers)
- SwinV2 (tiny/small/base/large, beta, HF Transformers)
- SAM v1 (vit-base/large/huge, beta, HF Transformers)
- SAM 2 (hiera-tiny/small/base-plus/large, beta, HF Transformers)
- Grounded SAM (Grounding DINO + SAM v1, beta)
- Grounded-SAM2 (Grounding DINO + SAM 2, beta)
- OneFormer (swin-large/dinat-large/convnext-large, beta, HF Transformers)
- ONNX export for SwinV2

**Security and privacy:**
- Local-only by default (`127.0.0.1`)
- No E2E encryption claimed (server must see plaintext tensors)
- `metadata_only` retention default — no image or prompt logging
- Log redaction (API keys, HF_TOKEN, base64, CF secrets)
- Optional encryption-at-rest for SQLite job store
- Auth modes: `local_private` / `lan_private` / `cloudflare_private` / `production_multi_user`
- SSRF protection, path traversal blocked, decompression bomb protection
- `visionservex security audit --json` → score=100, e2e_encryption_claimed=false

**GPU / VRAM safety:**
- VRAM safety guard: 80% cap, 3 GB min free, 3 GB GUI reserve on desktop GPU
- `visionservex gpu guard-status` / `gpu processes` / `gpu cleanup` / `gpu reset-advice`
- GPU tests run serially by default
- `GPU_MEMORY_GUARD` structured error instead of raw OOM
- `SERVER_BUSY` with `Retry-After` when queue full

**Scheduler:**
- Model-aware concurrency policies (gpu_exclusive, queue_recommended, acceptable_parallelism)
- `visionservex scheduler profile --json` — 12 models with benchmark-derived policies
- `visionservex scheduler set-policy` / `scheduler benchmark-policy`

**Syntax contract:**
- 222 examples, failing=0, release_ready=true

**OpenMMLab sidecar (not stable core):**
- `visionservex openmmlab pull <model_id>` prepares cache + prints official instructions
- `CHECKPOINT_REQUIRED` structured response — no fake output
- Docker path documented

**Docs updated:**
- `docs/model_zoo.md` — regenerated from registry (was stale)
- `docs/gpu_safety.md` — new, covers VRAM guard, cleanup, emergency recovery
- `docs/parallel_safety.md` — new, covers policies, benchmark results, serial GPU tests
- README — no "What remains" section; known limitations documented honestly

### Validated
- ruff clean, format clean
- pytest: 261 passed, 24 skipped
- build/twine: PASSED
- security audit: score=100
- syntax audit: failing=0
- models audit: 0 issues
- downloads audit --strict: 0 missing
- artifact check: clean

## [1.0.0rc3] - 2026-05-15

### Release Audit and GPU Safety Pass

**Decision:** v1.0.0rc3 (not final). Remaining blockers documented below are
honest, not silent — OpenMMLab requires docker/manual, TensorRT is dry-run,
MPS is implemented but unverified. No fake predictions returned.

### Added
- **`visionservex gpu guard-status`** — live VRAM safety guard report. Shows
  total/used/free VRAM, safety budget, active GPU processes, and policy.
- **`visionservex gpu processes`** — list GPU compute processes with
  VisionServeX/pytest marked safe-to-terminate and GUI processes protected.
- **`visionservex gpu cleanup`** — safely terminate VisionServeX/pytest/
  benchmark GPU processes. Never kills GUI processes (gnome-shell, Xwayland,
  browsers, editors, terminals). Requires confirmation unless `--yes`.
- **`visionservex gpu cleanup --dry-run`** — preview what would be terminated.
- **`visionservex gpu reset-advice`** — print emergency VRAM recovery commands.
  VisionServeX never auto-resets the GPU.
- **`visionservex gpu smoke-test --serial --max-vram-fraction --min-free-vram-gb
  --stop-on-vram-risk --allow-high-vram`** — VRAM safety flags on smoke-test.
  Runs serially by default and clears torch cache between models.
- **`visionservex openmmlab pull <model_id>`** — prepare OpenMMLab checkpoint
  cache directory; print exact download instructions with official model zoo
  links. Supports `--from-url` for direct download if user provides URL.
  Returns `CHECKPOINT_REQUIRED` structured response (not fake output) when no
  auto-download URL is available.
- **`visionservex scheduler set-policy <model_id>`** — runtime policy override
  for concurrency (runtime-only, not persisted).
- **`visionservex scheduler benchmark-policy <model_id>`** — show
  benchmark-derived concurrency policy for a model.
- **`visionservex downloads-audit --strict`** — exit 1 if any model has
  missing required download metadata (currently always passes: 0 missing).
- **`visionservex parallel-test --stop-on-vram-risk --max-vram-fraction
  --min-free-vram-gb`** — VRAM guard flags for parallel inference test.
- **VRAM safety guard** (`gpu_commands.py`): `_get_vram_state()`,
  `_compute_safety_budget()`, `_get_gpu_processes()`. Configurable via
  `VISIONSERVEX_RUNTIME__MAX_VRAM_FRACTION`, `MIN_FREE_VRAM_GB`,
  `RESERVE_GUI_VRAM`, `DESKTOP_GPU`, `ALLOW_HIGH_VRAM`.

### Fixed
- **Models audit** now correctly treats `status: manual` and `status: external`
  as self-documenting statuses — stubs with these statuses no longer produce
  spurious "stub without notes/warnings" audit warnings.
- **Registry**: Added `notes` fields to `rfdetr-seg-large`, `rfdetr-seg-xlarge`,
  `rfdetr-seg-2xlarge` (experimental stubs with no prior explanation).
- **Registry**: Added `notes` to `grounding-dino-1.6` (external/API, now
  clearly documented as API-gated).
- **Models audit** also checks `warnings` (not only `notes`) for external
  status documentation — `grounding-dino-1.5` already had `warnings:` and
  no longer triggers a false positive.

### Remaining Blockers Before v1.0.0 Final
- OpenMMLab checkpoint auto-download: `openmmlab pull` exists, but real
  inference still requires mmpose/mmdet packages and a valid checkpoint URL.
  Returns `CHECKPOINT_REQUIRED` structured error — no fake output.
- RTMPose / RTMDet-R2 real end-to-end inference: requires mmpose/mmrotate
  and actual checkpoint file. Status: `docker_checkpoint_required`.
- MPS verification: not verified (no Apple Silicon hardware). Documented.
- TensorRT: dry-run only. Documented. No overclaim.

## [1.0.0rc2] - 2026-05-15

### Security and Privacy Hardening Pass

**Honest disclaimer:** VisionServeX does NOT provide end-to-end encryption.
The inference server must see plaintext image tensors. This release provides
local-first processing, no-retention defaults, encrypted transport, optional
encryption-at-rest for job metadata, and auth for public mode.

### Added
- **Security modes**: `local_private` (default), `lan_private`, `cloudflare_private`,
  `production_multi_user`. Configure with `visionservex security mode MODE`.
- **`visionservex security audit --json`** — structured security posture report.
  Always includes `e2e_encryption_claimed: false`.
- **`visionservex security doctor`** — security health checks with actionable fixes.
- **`visionservex security checklist`** — deployment checklist including no-E2E note.
- **`visionservex security test-redaction`** — verify log redaction works.
- **`visionservex security keygen`** — generate Fernet key for encryption-at-rest.
- **`visionservex security check-key`** — verify key configuration.
- **`visionservex security mode MODE`** — show/apply security mode env vars.
- **`visionservex privacy cleanup --dry-run`** — list/delete vsx_* temp files.
- **`visionservex privacy inspect-cache`** — show temp files without revealing content.
- **`visionservex privacy retention [MODE]`** — show/explain retention mode.
- **`PrivacyConfig`** — `retention_mode`, `save_inputs`, `save_outputs`, `save_prompts`,
  `job_payload_retention`, `encrypt_job_store`, `encryption_key_file/env`, `temp_dir`.
- **`SecurityModeConfig`** — `mode`, `require_cloudflare_access`, `trust_cf_headers`,
  `sidecar_token`, `sidecar_url`, `tls_cert_file/key_file`.
- **`FieldEncryptor`** / `generate_key` — Fernet-based field-level encryption for
  SQLite job store metadata (requires `pip install cryptography`).
- **Secure temp files** (`secure_temp_file` context manager) — 0600 permissions,
  auto-deleted after use or on exception.
- **Enhanced log redaction** — HF tokens, Cloudflare secrets, base64 JPEG/PNG magic,
  `image_b64=` fields all scrubbed.
- **`docs/privacy.md`** — comprehensive privacy guide with honest E2E disclaimer.
- **`docs/threat_model.md`** — 4-mode threat model with what we protect and what we don't.
- **README rewritten** — current-state, security-first, honest about E2E and status levels.
- 28 new security/privacy tests in `tests/test_security_privacy.py`.

## [1.0.0rc1] - 2026-05-15

This is the first release candidate for v1.0.0. All 222 documented syntax
examples are classified, tested, or produce structured actionable errors.

### Fixed — API compatibility gaps (release blockers)
- **AsyncClient.segment**: `box`, `boxes`, `points`, `point_labels`, `labels`
  kwargs are now correctly forwarded to `/segment/b64`. Previous implementation
  silently dropped all prompts.
- **tunnel config --domain / --local-url**: the syntax contract specified
  `--domain API_HOSTNAME` and `--local-url http://...` flags; both are now
  supported in addition to the original positional `hostname` argument.

### Added
- **`visionservex syntax audit`** — classifies all 222 documented examples as
  `working / structured_error / external / unverified`. Failing count must be 0
  before v1.0.0.
- **`visionservex validation run [release|local|gpu|syntax]`** — run the test
  suite with a named profile. `release` profile matches CI; `gpu` enables GPU
  tests.
- **SQLite job store** (`VISIONSERVEX_JOBS__STORE=sqlite`) — optional persistent
  job backend with TTL-based cleanup and cancellation support.
- **`SQLiteJobStore`** — thread-safe, with `create/get/list/update/cancel/purge_old`.
- **`visionservex gateway health/logs/config/profile-list/token`** — new gateway
  diagnostics and dev tooling commands.
- **OpenMMLab sidecar honesty** — sidecar now returns `CHECKPOINT_REQUIRED`
  (HTTP 503 with structured error) instead of fake stub predictions when
  checkpoint/config files are absent. Prediction routes now attempt real
  MMPose/MMDet inference when the checkpoint IS present.

### Docs
- **`docs/gpu_validation.md`** updated to clearly distinguish:
  CPU-verified, CUDA-verified (RTX 5080 in v0.7.0+), MPS-implemented-unverified.
- TensorRT is `export_onnx_supported=true` (SwinV2) / `tensorrt_supported=false`
  with `dry_run_supported=true`.

### Tests
- 23 new tests in `tests/test_v100rc1.py` (233 total passing).

## [0.9.0] - 2026-05-15

### Added — Syntax Contract + Developer Experience
- **222-example syntax contract** (`docs/syntax_contract.md`) — every CLI/Python/API pattern documented and covered by tests.
- **`VisionServeXError` typed exception hierarchy** — `ModelNotFoundError`, `InputNotFoundError`, `DeviceUnavailableError`, `ModelMissingWeightsError`, `SidecarNotRunningError`, `ExternalModelError`, `ManualModelError`, `EngineDependencyError`. All carry `code`, `message`, `hint`, `details`.
- **`AsyncClient`** — full async HTTP client for the gateway with `detect/classify/segment/grounded_segment/predict/batch_detect`.
- **`VisionModel.loaded`** property.
- **`VisionModel.predict` convenience kwargs** — `prompt`, `box`, `labels`, `top_k`, `threshold`, `task` (so callers never need to know backend parameter names).
- **`BaseResult.save_json` and `save_image`** convenience methods.
- **`predict` CLI enriched** — `--device`, `--precision`, `--top-k`, `--point`, `--box`, `--task`, `--threshold`, `--save-json`, `--save-image`, `--auto-pull`, `--no-auto-pull`, `--timeout`, `--debug`.
- **`batch-predict`** CLI command (directory input, `--save-dir`, `--save-json`).
- **`gateway loaded-models`**, `gateway memory`, `gateway stop` commands.
- **`gateway start --auto-pull`**, `--auth`, `--config` flags.
- **`models-audit`** command.
- **`onnx-validate`** and **`onnx-parity`** commands.
- **`parallel-test-pair`** command.
- **`cache clean --model` / `--all`** flags.
- **`recommend --include-docker`**, `--vram` flags.
- **`pull-recommended --task TASK`** flag.
- **`pull-suite full-auto`** suite.
- **`/obb`** and **`/segment/b64`** server endpoints.
- **`Client.obb`, `Client.batch_detect`, `Client.job_events`, `Client.cancel_job`, `Client.job()`**.
- 36 new tests in `tests/test_syntax_contract.py` (210 total passing).

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

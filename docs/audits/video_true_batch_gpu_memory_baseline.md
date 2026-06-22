# VisionServeX — Video / True-Batch / GPU / Memory Sprint — Baseline Report

**Sprint branch:** `fix/video-true-batch-gpu-memory-vsx`
**Baseline captured:** 2026-06-22
**Purpose:** Record the real, measured starting state before any implementation. No claim in this
document is derived from registry assertions alone — every measurement below was produced by
running a command on this machine.

> Hard rule for this sprint: no "GPU saturation" claim without NVML + nvidia-smi evidence; no "true
> batch" claim if the implementation loops over single-image `predict()`; no "segmentation" claim if
> the output is only boxes.

---

## 1. Environment (measured)

| Item | Value | Source |
| ---- | ----- | ------ |
| Git HEAD (pre-branch) | `7a8e0f8` (main, clean) | `git rev-parse HEAD` |
| Sprint branch | `fix/video-true-batch-gpu-memory-vsx` | `git branch --show-current` |
| Package version | `3.21.0` | `visionservex.__version__` |
| Python | 3.13.12 | `python --version` |
| torch | 2.11.0+cu130 | `torch.__version__` |
| CUDA toolkit (nvcc) | 12.4.131 | `nvcc --version` |
| CUDA available (torch) | **True**, device_count=1 | `torch.cuda.is_available()` |
| GPU | **NVIDIA GeForce RTX 5080**, 16.6 GB (16303 MiB), **sm_120** | `torch.cuda.get_device_properties` |
| NVIDIA driver | 580.159.03 | `nvidia-smi` |
| GPU idle util / VRAM | ~1–2% / 1689–2294 MiB used of 16303 MiB | `nvidia-smi`, `pynvml` |
| NVML (pynvml) | **WORKS** — util + mem_get_info readable | `pynvml.nvmlDeviceGetUtilizationRates` |

This is a **real GPU host**, so live GPU/true-batch/memory evidence is achievable for this sprint
(not an "honest blocker"-only situation).

## 2. Video toolchain (measured)

| Item | Value | Source |
| ---- | ----- | ------ |
| ffmpeg | 8.0.1-3ubuntu2 | `ffmpeg -version` |
| ffprobe | 8.0.1-3ubuntu2 | `ffprobe -version` |
| NVENC encoders | `h264_nvenc`, `hevc_nvenc`, `av1_nvenc` | `ffmpeg -encoders` |
| NVDEC/CUVID decoders | `h264_cuvid`, `hevc_cuvid`, `vp9_cuvid`, `av1_cuvid`, … | `ffmpeg -decoders` |
| PyAV (`av`) | 17.0.0 | import |
| OpenCV (`cv2`) | 4.13.0 | import |
| decord | **ABSENT** (frame streaming will use PyAV / ffmpeg) | `find_spec` |
| pycocotools | PRESENT (RLE available) | import |
| supervision | 0.27.0.post2 (overlay) | import |

**Full hardware decode + encode path is available.** Worker-side remux/transcode/NVDEC/NVENC is
feasible — no "ffmpeg missing" blocker.

## 3. Owner video diagnosis (measured) — `/home/arash/Downloads/lv_0_20260617224920.mp4`

| Property | Value |
| -------- | ----- |
| Size | 89,880,941 bytes (~85.7 MiB) |
| Container | mp4 (`mov,mp4,m4a,…`) |
| Duration | 91.67 s |
| Overall bitrate | 7.84 Mbit/s |
| Video codec | **H.264 High profile, level 3.1** |
| Pixel format | **yuv420p (8-bit)** — already browser-compatible |
| Resolution | 1280×720 |
| FPS | 30/1 |
| Audio | AAC LC |
| Top-level atom order | `ftyp, free, mdat, moov` |
| **moov position** | **AFTER mdat → NOT faststart → needs lossless remux** |

**Diagnosis:** the codec/pixfmt are already web-safe; only the moov atom is misplaced.
The correct fix is a **lossless faststart remux** (`-c copy -movflags +faststart`), **not** a
re-encode. This is the canonical Phase 4 acceptance case.

## 4. Key dependencies (measured)

rfdetr (present), libreyolo 1.1.1, transformers 5.10.2, fastapi 0.136.1, uvicorn 0.46.0,
pydantic 2.12.4, torchvision 0.26.0+cu130, onnxruntime 1.26.0, starlette 1.0.0, pynvml (present).

## 5. Model registry (measured: 151 models)

`src/visionservex/registry/models.yaml` — 151 entries.

**Task distribution:** detect 48, classify 27, foundation_segment 17, segment 14,
open_vocab_detect 12, embed 10, obb 9, pose 7, grounded_segment 5, vlm 2.

**Engine distribution:** openmmlab 21, dfine 14, libreyolo 14, _stub 14, torchvision_classify 13,
rfdetr 11, mock 8, sam2_hf 8, grounding_dino 7, swinv2 4, sam_hf 4, dinov2 4, … .

### 5a. Registry `batch_support` claims (to be verified vs measured behavior in Phase 1)

Distribution of the registry's own `batch_support` field: **True=10, False=9, None(unspecified)=134.**

> **Truth-vs-claim flag #1 (the owner's central concern):** the owner believes "D-FINE is the only
> true tensor-batch path." The registry says the **opposite**:
> - **D-FINE**: `batch_support=False` for dfine-n/s/m (+coco/o365); `None` for dfine-l/x. Never `True`.
> - **RF-DETR**: `batch_support=True` for rfdetr-base/nano/small; `None` for medium/large.
> - **RF-DETR-Seg**: `batch_support=None` for all; **nano/small/medium = `wired/beta`**, but
>   **large/xlarge/2xlarge = `impl=stub status=experimental` (NOT real)**.
> - **LibreYOLO**: `batch_support=None` for all 14.
> - **SAM/SAM2/SAM2.1**: `batch_support=None` for all.
>
> Phase 1 will measure the actual forward path for each and reconcile this. The registry field is
> currently inconsistent and cannot be trusted as-is.

## 6. Support matrix (engine availability — measured by registry+engine presence)

| Family | Variants (registry) | Engine | impl/status | Notes |
| ------ | ------------------- | ------ | ----------- | ----- |
| D-FINE | 14 (n/s/m/l/x ×{base,coco,o365}) | `dfine` | wired/beta | claimed by owner as true-batch path — VERIFY |
| RF-DETR | base/nano/small/medium/large (5 detect, 11 total w/ seg) | `rfdetr` | wired/beta | base/nano/small claim batch_support=True |
| RF-DETR-Seg | nano/small/medium (wired) + large/xlarge/2xlarge (STUB) | `rfdetr` | mixed | seg output truth = Phase 5 |
| LibreYOLO | 14 (yolox/yolov9/rtdetr/dfine) | `libreyolo` | wired/beta | reported as `worker_internal_loop` |
| SAM / SAM2 / SAM2.1 | sam-vit-base/large/huge, sam2(.1)-hiera-tiny/small/base+/large | `sam_hf`/`sam2_hf` | wired/beta | foundation_segment |
| GroundingDINO | 7 | `grounding_dino` | wired | open_vocab_detect |
| GroundedSAM | grounded-sam, grounded-sam2 | `grounded_sam(2)` | wired | grounded_segment |
| DINOv2 | small/base/large/giant | `dinov2` | wired | embed |
| DINOv3 | (not in registry; gated upstream) | — | — | gated/absent |
| INSID3 | small/base/large | (insid3 runtime) | — | correspondence/in-context seg |
| OneFormer | 3 | `oneformer` | mixed | segment |
| OpenMMLab (rtmdet/rtmpose/internimage) | 21 | `openmmlab`/`_sidecar` | mostly stub/sidecar | sidecar-gated |

## 7. Architecture audit (filled from code exploration — file:line cited)

_See `docs/audits/model_batch_output_truth_matrix.md` for the measured per-variant truth matrix._

### 7a. Inference / batch (the central finding)
- `engines/base.py:33-90` — `BaseEngine` is **strictly single-image**. No `predict_batch`,
  no `supports_true_batch`. `supports()` only advertises `"predict"`. **No batch API exists.**
- `core/model.py:316-326` — `VisionModel.batch_predict()` is a **FAKE BATCH** Python loop
  (`for img in images: results.append(self.predict(img))`). `stream()` (328-334) same.
- `server/app.py:539-557` — `/batch-predict` HTTP endpoint exists but maps onto the fake loop.
- `engines/dfine.py:167,176` — D-FINE calls `self._processor(images=[image])` then
  `self._model(**inputs)`: **real forward, but always batch size 1**. The HF processor *accepts a
  list*, so **true forward-batch is achievable** — currently never exercised. → Phase 2 target.
- `engines/rfdetr.py:216` — RF-DETR delegates to `self._rfdetr_model.predict(image)` (single image,
  closed-box). Registry `batch_support=True` for rfdetr-base/nano/small is **an unproven claim** to
  verify or downgrade in Phase 1.
- `engines/libreyolo.py:411` — `self._model(image)` single image (matches the `worker_internal_loop`
  report). `engines/sam_hf.py:151`, `engines/sam2_hf.py:158` single image.
  `engines/grounded_sam.py:132` — **fake batch**: SAM called once per detection in a loop.

> **Conclusion:** true-batch infrastructure does NOT exist today. Phase 2 must add a real
> `predict_batch` (forward once on a stacked batch) plus a **failing test that catches hidden
> internal loops** so no engine can falsely advertise true batch.

### 7b. GPU telemetry & cleanup — ALL REAL (Phase 3 can build on it)
- `runtime/gpu_lifecycle.py:70-146` — `torch.cuda.memory_allocated/reserved/max_allocated` (real),
  `nvidia-smi --query-compute-apps` subprocess (real), `clear_torch_cuda_cache()` =
  `synchronize+empty_cache+ipc_collect+reset_peak_memory_stats` (real).
- `runtime/gpu_profile.py:151-196` real device detection; `runtime/cuda_probe.py:35-97` real matmul
  kernel probe + nvidia-smi driver; `runtime/monitor.py:11-67` real in-memory latency metrics;
  `runtime/postprocess.py:30-80` real numpy class-aware NMS.
- `runtime/scheduler.py:45-163` — `RequestScheduler` is **concurrency-only** (per-model semaphore +
  503 backpressure). **No adaptive batch sizing.** → Phase 3 adds a new adaptive scheduler beside it.

### 7c. Server / job lifecycle / cancellation
- `server/app.py:82-136` FastAPI `create_app` factory; launched via `visionservex serve`
  (uvicorn factory, default `127.0.0.1:8080`). 27 endpoints. **No `/video*`, no `/infer-batch`.**
- `runtime/jobs.py:45-141` — `Job` has `cancelled` flag + `JobStore.cancel()` sets it.
  **Cancellation is a STUB:** the flag is set but **never checked** in any download/inference loop
  (`server/app.py:735-780` runner ignores it). → Phase 7 must make cancel real.
- `runtime/concurrency.py` + `RequestScheduler` — REAL per-model concurrency & backpressure.

### 7d. Video pipeline
- `runtime/video_io.py:106-164` — REAL cv2 streaming reader (frame-by-frame, no full RAM load),
  FPS/stride/target_fps/time-range, timestamp+index preserved. `runtime/video_search.py` REAL
  appearance index. `runtime/temp_manager.py:44-57` REAL secure temp (0600, auto-cleanup).
- **ABSENT:** ffprobe (0 matches), ffmpeg subprocess (0 matches), NVENC/NVDEC, faststart remux,
  transcode, HTTP video endpoints, video size/duration/resolution limits.
- `core/visualization.py:95-544` REAL overlay drawing (boxes/masks/tracks/pose/obb);
  `cli/annotate_commands.py:346` writes MP4 via `cv2.VideoWriter` (mp4v, CPU only).
  → Phase 4 adds ffprobe/remux/transcode/extract (NVENC/NVDEC) + HTTP endpoints + limits.

### 7e. Segmentation output (root cause of "boxes only")
- `core/results.py:87-99` — `Segment` dataclass has `mask: np.ndarray`; masks **are** produced and
  preserved through `engines/rfdetr.py:355-406` `_sv_to_segments` and through SAM engines.
- **`core/results.py:356-384` — `SegmentationResult.to_dict()` POPS the mask** and emits only
  `mask_shape` + `mask_pixels_on`. **No RLE, no polygon in the JSON response.** Over HTTP, Anastig
  receives boxes + a pixel count but **no transmittable mask** → renders as "boxes only."
- No `return_masks`/`return_rle`/`return_polygons` flags. RLE exists only in
  `runtime/rfdetr_seg_benchmark.py:216` (benchmark path), not in the API response.
  → Phase 5 adds RLE/polygon serialization + flags. The masks are real; the **contract** drops them.

## 8. /health output (measured, live via TestClient)

```
GET /health  -> {"status":"ok","version":"3.21.0","public_mode":false,"auth_enabled":false}
GET /version -> {"version":"3.21.0","python":"3.13.12","platform":"Linux-...","author":"Arash Sajjadi"}
GET /devices -> {"devices":[{"name":"cpu",...},{"name":"cuda","available":<reflects CUDA visibility>}]}
```

**Finding:** `/health` does **not** report GPU/VRAM/model state — only version/public_mode/auth. GPU
state is split across `/devices` and `/gateway/status`. (The `/devices` "cuda unavailable" seen in
the capture is an artifact of the `CUDA_VISIBLE_DEVICES=""` safety guard used for the probe, not a
real GPU-absence finding — the GPU is present per §1.)

---

*This baseline will not be edited to soften findings. Implementation phases append evidence; they do
not rewrite measured facts.*

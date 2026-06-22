# Anastig ↔ VisionServeX Integration Contract (v3.22.0)

This is the authoritative contract Anastig should code against after the v3.22.0
worker release. Every field below is backed by live evidence captured on an
RTX 5080 (see `docs/audits/vsx_3_22_video_batch_gpu_release_report.md`).

> **Honesty contract.** The worker never labels a single-image loop as a true
> batch, never claims GPU saturation it didn't measure, and never presents a
> boxes-only result as segmentation. Anastig should surface the worker's own
> `batch_mode`, `bottleneck`, and `output_mode` fields verbatim.

---

## 1. Health / capability discovery

| Endpoint | Method | Use |
| -------- | ------ | --- |
| `/health` | GET | `{status, version, public_mode, auth_enabled}` — version is now `3.22.0` |
| `/version` | GET | full version struct |
| `/devices` | GET | CPU/CUDA device list with VRAM |
| `/models/{id}` | GET | per-model capability (includes the new batch + sidecar fields) |

`model_capabilities(id)` (also surfaced per model) now includes:

```json
{
  "supports_true_batch": true,
  "batch_path": "true_tensor_batch | internal_loop | unsupported",
  "max_batch_size_hint": 32,
  "preferred_batch_sizes": [1, 2, 4, 8, 16, 32],
  "registry_batch_support_claim": true
}
```

**Anastig rule:** show a "True batch" badge only when `supports_true_batch=true`.
Today that is **D-FINE only** (all variants). RF-DETR / RF-DETR-Seg / LibreYOLO /
SAM report `batch_path="internal_loop"` — show "Sequential" honestly.

## 2. Batch inference — `POST /infer-batch`

Multipart: `images[]` (files), `model_id`, optional `threshold`.

Response:
```json
{
  "model_id": "dfine-n",
  "requested_batch_size": 8,
  "results": [ <PredictionResult>, ... ],
  "batch": {
    "batch_mode": "true_tensor_batch",
    "true_forward_batch": true,
    "internal_loop": false,
    "actual_batch_size": 8,
    "preprocess_ms": 16.4, "forward_ms": 13.5, "postprocess_ms": 1.6,
    "gpu_util_avg": 3.0, "gpu_util_peak": 3,
    "vram_used_peak_mb": 374.5, "vram_reserved_peak_mb": 512.0, "vram_free_min_mb": 13290.6,
    "oom_recovered": false, "fallback_reason": null
  }
}
```

## 3. Video pipeline endpoints

| Endpoint | Method | Body | Returns |
| -------- | ------ | ---- | ------- |
| `/video/probe` | POST | `video` | `VideoProbe` JSON (codec, pix_fmt, faststart, `recommended_action`) |
| `/video/remux-faststart` | POST | `video` | **lossless** faststart MP4 (file) |
| `/video/transcode-browser-h264` | POST | `video`, `preset∈{480p,720p,1080p,source}` | browser H.264 MP4 (file, NVENC if available) |
| `/video/extract-frames` | POST | `video`, `sample_fps`, `stride`, `max_frames` | `{n_frames, frames:[{frame_index,time_sec,width,height}]}` |
| `/video/infer` | POST | `video`, `model_id`, `sample_fps`, `max_frames`, `mode`, `threshold` | **202** `{job_id, run_id, poll, cancel}` |
| `/video/export-overlay` | POST | `video`, `model_id`, ... | annotated MP4 (file) |

**Recommended flow for an uploaded video:**
1. `POST /video/probe`. If `recommended_action == "none"` → play as-is.
2. If `"remux_faststart"` → `POST /video/remux-faststart` (lossless, ~70 ms for a
   90 s clip). **Do not transcode.**
3. If `"transcode"` → `POST /video/transcode-browser-h264?preset=720p`.
4. For analysis → `POST /video/infer`, then poll the job.

`VideoProbe.recommended_action ∈ {none, remux_faststart, transcode, unsupported}`.

## 4. Job lifecycle (cancellable)

| Endpoint | Method | Use |
| -------- | ------ | --- |
| `/jobs/{job_id}` | GET | full job snapshot (`status`, `progress`, `result`, `cancel_requested`) |
| `/jobs/{job_id}/events` | GET | poll snapshot or `?sse=true` stream |
| `/jobs/{job_id}/cancel` | POST | request cancel (also `DELETE /jobs/{job_id}`) |
| `/jobs/{job_id}/artifacts` | GET | artifact + result summary |
| `/jobs/{job_id}/cleanup` | POST | release temp files + GPU cache |

Job statuses: `queued → running_inference → completed | failed | cancelled`.
Cancellation is **real**: it sets a signal the worker checks between waves, stops
submitting new waves, releases tensors, and reports a partial result with
`cancelled: true`. Switching models = cancel the old `/video/infer` job, then
start a new one — no browser refresh needed. A second heavy video job returns
**409 WORKER_BUSY**.

`/video/infer` job `result` (on completion) carries:
```json
{
  "frames_processed": 48, "waves": 6, "throughput_fps": 17.0, "cancelled": false,
  "batch_trajectory": [4,6,8,12,12,6], "scheduler_converged": true,
  "bottleneck_summary": {"preprocess_ms_total":124.7,"forward_ms_total":130.2,
                          "postprocess_ms_total":22.2,"gpu_util_avg":2.9,
                          "vram_used_peak_mb":553.5,"vram_after_cleanup_mb":24.0},
  "frames": [{"frame_index":0,"time_sec":0.0,"n_objects":96,"detections":[...]}],
  "scheduler_decisions": [{"wave":4,"action":"hold","bottleneck":"preprocess",
                            "reason":"preprocess-bound ... growing batch won't help the GPU"}]
}
```

## 5. Segmentation output contract

Every `SegmentationResult` (`/segment`, `/video/infer` on a seg model) now serializes:
```json
{
  "output_mode": "boxes+masks_rle",
  "masks_available": true,
  "segments": [{
     "box": {"x1":..,"y1":..,"x2":..,"y2":..}, "score": 0.9, "label": "car",
     "mask_shape": [720,1280], "mask_pixels_on": 266,
     "rle": {"size":[720,1280], "counts":"...", "format":"coco_rle"},
     "mask_quality": {"valid":true,"area_frac":0.0003,"mask_box_iou":0.66,"warnings":["tiny_mask"]}
  }]
}
```
- **RLE is emitted by default** (compact, COCO-decodable; round-trips to the exact
  mask). Request `return_polygons=true` for vector outlines (`max_polygon_points`,
  `polygon_simplification_tolerance` control cost).
- `output_mode ∈ {boxes+masks_rle, boxes+masks_rle+polygons, boxes+masks,
  boxes_only_masks_unavailable, empty}`.
- If a "segmentation" model returns no real masks, `masks_available=false`,
  `output_mode="boxes_only_masks_unavailable"`, and a `SEGMENTATION_MASKS_UNAVAILABLE`
  warning is included — **never silently present boxes as masks.**

## 6. Exact UI messages Anastig should display

| Worker condition (field) | UI message |
| ------------------------ | ---------- |
| `batch.batch_mode=="true_tensor_batch"` | "True GPU batch (N images / 1 forward)." |
| `batch.batch_mode=="internal_loop"` | "Sequential inference — this model has no true batch path." |
| bottleneck `preprocess`/`postprocess` + `gpu_util_avg` low | "GPU is not the limit — CPU {preprocess|postprocess} is the bottleneck ({x} ms vs {y} ms forward). Throughput won't improve with a bigger batch." |
| `output_mode=="boxes_only_masks_unavailable"` | "Segmentation unavailable for this model/frame — showing boxes only." |
| `output_mode` has `masks_rle` but not `polygons` | "Masks available (RLE). Polygons deferred — request full polygons for selected frames." |
| overlay export `encoder=="cv2_mp4v"` (ffmpeg missing) | "MP4 (H.264) unavailable on the worker — returned a fallback that may not play in-browser. Install ffmpeg/NVENC for browser-safe MP4." |
| 409 `WORKER_BUSY` | "Worker is busy with another GPU video job. Retry shortly." |
| job `cancelled:true` | "Canceled. Worker released GPU memory ({vram_after} MB)." |
| `batch.oom_recovered==true` | "Recovered from a GPU out-of-memory by halving the batch — continuing." |
| probe `recommended_action=="remux_faststart"` | "Preparing video (lossless faststart, no quality loss)…" |
| probe `recommended_action=="transcode"` | "Converting video to browser-safe H.264…" |

## 7. What Anastig can stop working around

- **No more manual ffmpeg.** The worker probes and remuxes/transcodes (`/video/*`).
- **No more "GPU idle = broken".** The worker reports the measured bottleneck;
  display it instead of assuming the worker is stuck.
- **No more "segmentation = boxes".** Masks are transmitted as RLE by default.
- **No more stuck cancel.** `/jobs/{id}/cancel` actually interrupts and frees VRAM.
- **No more guessing batch.** `supports_true_batch`/`batch_path` are authoritative.

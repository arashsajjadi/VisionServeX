# Video Inference (v3.22.0)

VisionServeX runs worker-side video decode, batched inference, and export — no
manual ffmpeg required.

## Pipeline
1. **Probe** (`runtime.ffmpeg_tools.probe_video`) — container/codec/pix_fmt/fps/
   duration + moov/faststart status → `recommended_action`.
2. **Prepare** — lossless `remux_faststart` when only the moov atom is misplaced;
   otherwise `transcode_browser_h264` (NVENC when available).
3. **Stream + infer** (`runtime.video_pipeline.infer_video`) — frames are streamed
   from disk (never the whole video in RAM), grouped into adaptive microbatches,
   and run through `model.batch_predict`. Each frame result is INDEPENDENT and
   carries its original `frame_index` + `time_sec`.
4. **Export** — overlay MP4 (`runtime.video_export.export_overlay_video`) or
   annotation files (JSON/CSV/COCO).

## Frame sampling & long videos
- `sample_fps` (e.g. `2.0` = one frame / 0.5 s; values >5 give <0.2 s spacing),
  `stride`, `start_s`/`end_s` windows, `max_frames`.
- Long videos are handled by streaming + per-wave tensor release; memory does not
  grow with video length (proven: 300-frame run, VRAM released to baseline).

## CLI / HTTP
- HTTP: `/video/probe`, `/video/remux-faststart`, `/video/transcode-browser-h264`,
  `/video/extract-frames`, `/video/infer` (job-based), `/video/export-overlay`.
- See `docs/anastig_integration_contract.md` for schemas + UI messages.

## Security
- ffmpeg args are **preset-only**; no request string is interpolated into an
  ffmpeg argument; `shell=False` always.
- Uploads land in 0600 sandboxed temp files, deleted after the job.
- Hard caps: size (2 GiB), duration (1 h), resolution (8K), fps (240) — enforced
  before any heavy operation (`enforce_limits`).

## Evidence (RTX 5080, owner video 1280×720 h264)
- ffprobe 37 ms; faststart remux 74 ms (lossless); 720p NVENC transcode 4.46 s;
  decode 61–245 fps. See `docs/audits/evidence/v322_video_decode_matrix.md`.

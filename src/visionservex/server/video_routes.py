# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""HTTP routes for batch + video inference and the worker video toolkit (v3.22.0).

Mounted by ``create_app`` via ``app.include_router(build_media_router())``. Reuses
the existing model cache, job store, and auth. Endpoints:

    POST /infer-batch                     true/honest-loop batch inference + telemetry
    POST /video/probe                     ffprobe diagnostics + recommended action
    POST /video/remux-faststart           lossless moov-atom relocation
    POST /video/transcode-browser-h264    safe H.264/yuv420p transcode (NVENC if avail)
    POST /video/extract-frames            streamed frame metadata extraction
    POST /video/infer                     job-based, cancellable, adaptive-batched video inference
    POST /video/export-overlay            annotated MP4 (NVENC if available)
    POST /jobs/{job_id}/cancel            cancel alias (also DELETE /jobs/{job_id})
    GET  /jobs/{job_id}/artifacts         list job artifacts
    POST /jobs/{job_id}/cleanup           release job artifacts/temp

Security: ffmpeg args are preset-only (see runtime.ffmpeg_tools); uploads are
written to sandboxed temp files (0600) and cleaned up; size/duration/resolution
caps are enforced before heavy work.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# One heavy GPU video job at a time per worker (Phase 7.5 concurrency policy).
_heavy_lock = threading.Lock()
# job_id -> list of artifact paths to clean up
_job_artifacts: dict[str, list[str]] = {}


def _save_upload(upload: UploadFile, suffix: str = ".mp4") -> str:
    """Stream an upload to a sandboxed 0600 temp file. Returns the path."""
    import stat as _stat

    fd, path = tempfile.mkstemp(suffix=suffix, prefix="vsx_vid_")
    os.chmod(fd, _stat.S_IRUSR | _stat.S_IWUSR)
    with os.fdopen(fd, "wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


async def _auth(request: Request) -> None:
    from visionservex.server.app import _require_auth

    await _require_auth(request)


def build_media_router() -> APIRouter:
    router = APIRouter()

    # ---------------- /infer-batch (Phase 2) ----------------
    @router.post("/infer-batch", tags=["inference"])
    async def infer_batch(
        request: Request,
        images: list[UploadFile] = File(...),
        model_id: str = Form(...),
        threshold: float | None = Form(default=None),
    ) -> JSONResponse:
        await _auth(request)
        from PIL import Image

        from visionservex.runtime.batch_infer import run_batch_with_telemetry
        from visionservex.runtime.cache import get_model_cache

        pil_images = []
        for up in images:
            data = up.file.read()
            pil_images.append(Image.open(__import__("io").BytesIO(data)).convert("RGB"))

        model = get_model_cache().get(model_id)
        kwargs: dict[str, Any] = {}
        if threshold is not None:
            kwargs["threshold"] = threshold

        def _run():
            return run_batch_with_telemetry(model, pil_images, **kwargs)

        import asyncio

        results, tel = await asyncio.get_running_loop().run_in_executor(None, _run)
        return JSONResponse(
            {
                "model_id": model_id,
                "requested_batch_size": len(pil_images),
                "results": [r.to_dict() for r in results],
                "batch": tel.to_dict(),
            }
        )

    # ---------------- /video/probe ----------------
    @router.post("/video/probe", tags=["video"])
    async def video_probe(request: Request, video: UploadFile = File(...)) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.ffmpeg_tools import FFmpegUnavailableError, probe_video

        path = _save_upload(video)
        try:
            try:
                p = probe_video(path)
            except FFmpegUnavailableError as exc:
                return JSONResponse(
                    {"ok": False, "error": "FFMPEG_UNAVAILABLE", "detail": str(exc)},
                    status_code=503,
                )
            return JSONResponse(p.to_dict())
        finally:
            with contextlib.suppress(Exception):
                os.remove(path)

    # ---------------- /video/remux-faststart ----------------
    @router.post("/video/remux-faststart", tags=["video"])
    async def video_remux(request: Request, video: UploadFile = File(...)) -> Any:
        await _auth(request)
        from visionservex.runtime.ffmpeg_tools import (
            FFmpegUnavailableError,
            VideoLimitError,
            enforce_limits,
            probe_video,
            remux_faststart,
        )

        path = _save_upload(video)
        out = path + ".faststart.mp4"
        try:
            try:
                enforce_limits(probe_video(path))
                res = remux_faststart(path, out)
            except FFmpegUnavailableError as exc:
                return JSONResponse(
                    {"ok": False, "error": "FFMPEG_UNAVAILABLE", "detail": str(exc)}, 503
                )
            except VideoLimitError as exc:
                return JSONResponse({"ok": False, "error": "VIDEO_LIMIT", "detail": str(exc)}, 413)
            if not res["ok"]:
                return JSONResponse(
                    {"ok": False, "error": "REMUX_FAILED", "detail": res.get("stderr_tail", "")},
                    500,
                )
            return FileResponse(
                out,
                media_type="video/mp4",
                filename="faststart.mp4",
                background=_cleanup_bg(path, out),
            )
        except Exception:
            for p in (path, out):
                with contextlib.suppress(Exception):
                    os.remove(p)
            raise

    # ---------------- /video/transcode-browser-h264 ----------------
    @router.post("/video/transcode-browser-h264", tags=["video"])
    async def video_transcode(
        request: Request,
        video: UploadFile = File(...),
        preset: str = Form(default="720p"),
    ) -> Any:
        await _auth(request)
        from visionservex.runtime.ffmpeg_tools import (
            BROWSER_PRESETS,
            FFmpegUnavailableError,
            VideoLimitError,
            enforce_limits,
            probe_video,
            transcode_browser_h264,
        )

        if preset not in BROWSER_PRESETS:
            return JSONResponse(
                {"ok": False, "error": "BAD_PRESET", "allowed": sorted(BROWSER_PRESETS)}, 400
            )
        path = _save_upload(video)
        out = path + f".{preset}.mp4"
        try:
            try:
                enforce_limits(probe_video(path))
                res = transcode_browser_h264(path, out, preset=preset)
            except FFmpegUnavailableError as exc:
                return JSONResponse(
                    {"ok": False, "error": "FFMPEG_UNAVAILABLE", "detail": str(exc)}, 503
                )
            except VideoLimitError as exc:
                return JSONResponse({"ok": False, "error": "VIDEO_LIMIT", "detail": str(exc)}, 413)
            if not res["ok"]:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "TRANSCODE_FAILED",
                        "detail": res.get("stderr_tail", ""),
                    },
                    500,
                )
            return FileResponse(
                out,
                media_type="video/mp4",
                filename=f"browser_{preset}.mp4",
                background=_cleanup_bg(path, out),
            )
        except Exception:
            for p in (path, out):
                with contextlib.suppress(Exception):
                    os.remove(p)
            raise

    # ---------------- /video/extract-frames ----------------
    @router.post("/video/extract-frames", tags=["video"])
    async def video_extract(
        request: Request,
        video: UploadFile = File(...),
        sample_fps: float | None = Form(default=None),
        stride: int = Form(default=1),
        max_frames: int | None = Form(default=None),
    ) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.video_pipeline import extract_frames_to_list

        path = _save_upload(video)
        try:
            frames = extract_frames_to_list(
                path, sample_fps=sample_fps, stride=stride, max_frames=max_frames
            )
            return JSONResponse({"ok": True, "n_frames": len(frames), "frames": frames})
        finally:
            with contextlib.suppress(Exception):
                os.remove(path)

    # ---------------- /video/infer (job-based, cancellable) ----------------
    @router.post("/video/infer", tags=["video"])
    async def video_infer(
        request: Request,
        video: UploadFile = File(...),
        model_id: str = Form(...),
        sample_fps: float | None = Form(default=2.0),
        max_frames: int | None = Form(default=None),
        mode: str = Form(default="balanced"),
        threshold: float | None = Form(default=None),
    ) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.cache import get_model_cache
        from visionservex.runtime.jobs import get_job_store
        from visionservex.runtime.video_pipeline import CancelToken, infer_video

        if not _heavy_lock.acquire(blocking=False):
            return JSONResponse(
                {
                    "error": "WORKER_BUSY",
                    "detail": "another heavy GPU video job is running; retry later",
                },
                status_code=409,
            )
        path = _save_upload(video)
        store = get_job_store()
        job = store.create(model_id=model_id, kind="video_infer")
        _job_artifacts.setdefault(job.job_id, []).append(path)
        token = CancelToken(_event=job.cancel_event)
        store.update(
            job.job_id, status="running_inference", progress={"frames_done": 0, "stage": "starting"}
        )

        def _worker() -> None:
            try:
                model = get_model_cache().get(model_id)

                def _progress(p: dict[str, Any]) -> None:
                    cur = store.get(job.job_id)
                    if cur is not None:
                        store.update(job.job_id, progress={**p, "stage": "inferring"})

                report = infer_video(
                    model,
                    path,
                    sample_fps=sample_fps,
                    max_frames=max_frames,
                    mode=mode,
                    threshold=threshold,
                    cancel=token,
                    on_progress=_progress,
                )
                final = store.get(job.job_id)
                if final is not None and final.cancel_requested:
                    store.update(
                        job.job_id, status="cancelled", result=report, message="cancelled by client"
                    )
                else:
                    store.update(
                        job.job_id,
                        status="completed",
                        result=report,
                        message=f"{report['frames_processed']} frames",
                    )
            except Exception as exc:
                store.update(
                    job.job_id,
                    status="failed",
                    error={"type": type(exc).__name__, "message": str(exc)[:300]},
                )
            finally:
                with contextlib.suppress(Exception):
                    os.remove(path)
                _heavy_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return JSONResponse(
            {
                "job_id": job.job_id,
                "run_id": job.run_id,
                "status": "running_inference",
                "poll": f"/jobs/{job.job_id}",
                "cancel": f"/jobs/{job.job_id}/cancel",
            },
            status_code=202,
        )

    # ---------------- /video/export-overlay ----------------
    @router.post("/video/export-overlay", tags=["video"])
    async def video_export_overlay(
        request: Request,
        video: UploadFile = File(...),
        model_id: str = Form(...),
        sample_fps: float | None = Form(default=None),
        max_frames: int | None = Form(default=None),
        threshold: float | None = Form(default=None),
    ) -> Any:
        await _auth(request)
        from visionservex.runtime.cache import get_model_cache
        from visionservex.runtime.video_export import export_overlay_video

        path = _save_upload(video)
        out = path + ".overlay.mp4"
        try:
            model = get_model_cache().get(model_id)
            res = export_overlay_video(
                model, path, out, sample_fps=sample_fps, max_frames=max_frames, threshold=threshold
            )
            if not res.get("ok"):
                return JSONResponse({"ok": False, "error": "OVERLAY_FAILED", "detail": res}, 500)
            return FileResponse(
                out,
                media_type=res.get("media_type", "video/mp4"),
                filename=os.path.basename(res["output"]),
                background=_cleanup_bg(path),
            )
        except Exception:
            for p in (path, out):
                with contextlib.suppress(Exception):
                    os.remove(p)
            raise

    # ---------------- job lifecycle extras ----------------
    @router.post("/jobs/{job_id}/cancel", tags=["jobs"])
    async def job_cancel_post(request: Request, job_id: str) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.jobs import get_job_store

        ok = get_job_store().cancel(job_id)
        if not ok:
            return JSONResponse(
                {
                    "job_id": job_id,
                    "cancelled": False,
                    "detail": "job not found or already terminal",
                },
                404,
            )
        return JSONResponse({"job_id": job_id, "status": "cancelled", "cancel_requested": True})

    @router.get("/jobs/{job_id}/artifacts", tags=["jobs"])
    async def job_artifacts(request: Request, job_id: str) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.jobs import get_job_store

        job = get_job_store().get(job_id)
        if job is None:
            return JSONResponse({"job_id": job_id, "error": "not found"}, 404)
        result = job.result if isinstance(job.result, dict) else {}
        return JSONResponse(
            {
                "job_id": job_id,
                "status": job.status,
                "temp_artifacts": _job_artifacts.get(job_id, []),
                "result_summary": {
                    k: result.get(k)
                    for k in (
                        "frames_processed",
                        "waves",
                        "throughput_fps",
                        "cancelled",
                        "batch_trajectory",
                        "bottleneck_summary",
                    )
                }
                if result
                else None,
            }
        )

    @router.post("/jobs/{job_id}/cleanup", tags=["jobs"])
    async def job_cleanup(request: Request, job_id: str) -> JSONResponse:
        await _auth(request)
        from visionservex.runtime.gpu_lifecycle import clear_torch_cuda_cache, force_gc

        removed = []
        for p in _job_artifacts.pop(job_id, []):
            with contextlib.suppress(Exception):
                if os.path.exists(p):
                    os.remove(p)
                    removed.append(p)
        force_gc()
        clear_torch_cuda_cache()
        return JSONResponse(
            {"job_id": job_id, "removed_artifacts": removed, "gpu_cache_cleared": True}
        )

    return router


def _cleanup_bg(*paths: str):
    """Return a BackgroundTask that deletes temp files after the response is sent."""
    from starlette.background import BackgroundTask

    def _rm() -> None:
        for p in paths:
            with contextlib.suppress(Exception):
                os.remove(p)

    return BackgroundTask(_rm)


__all__ = ["build_media_router"]

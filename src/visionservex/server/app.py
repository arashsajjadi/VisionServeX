# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import base64
import platform
import sys
import threading
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image

from visionservex import __version__
from visionservex.api.errors import (
    ApiError,
    bad_request,
    busy,
    forbidden,
    not_found,
    too_large,
    unauthorized,
    unprocessable,
)
from visionservex.api.schemas import (
    DownloadingResponse,
    HealthResponse,
    JobResponse,
    ModelListItem,
    ModelListResponse,
    PredictionResponse,
    PromptRequest,
    VersionResponse,
)
from visionservex.config import Settings, get_settings
from visionservex.core.results import BaseResult
from visionservex.engines.base import MissingDependencyError
from visionservex.registry import RegistryError, default_registry
from visionservex.runtime.cache import get_model_cache
from visionservex.runtime.device import available_devices
from visionservex.runtime.downloads import (
    DownloadError,
    DownloadProgress,
    ManualDownloadRequired,
    cached_path,
    download,
    is_cached,
)
from visionservex.runtime.jobs import Job, get_job_store
from visionservex.runtime.monitor import metrics
from visionservex.runtime.scheduler import (
    BackpressureError,
    RequestTimeoutError,
    get_scheduler,
)
from visionservex.security.auth import authenticate_request
from visionservex.security.middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
)
from visionservex.security.ssrf import URLValidationError, validate_remote_url
from visionservex.security.validators import (
    InputValidationError,
    validate_image_bytes,
    validate_mime_type,
)
from visionservex.utils.ids import request_id
from visionservex.utils.images import encode_jpeg, open_safe
from visionservex.utils.logging import configure_logging, get_logger

_log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    if settings is None:
        settings = get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        for warning in settings.public_safety_warnings():
            _log.warning(warning)
        if settings.server.public_mode and settings.auth.enabled:
            _log.info("server starting in public mode with authentication enabled")
        elif settings.server.public_mode:
            _log.warning(
                "PUBLIC MODE ENABLED WITHOUT AUTHENTICATION. "
                "All requests will be accepted from any client able to reach the origin."
            )
        else:
            _log.info(
                "server starting in local-only mode on %s:%s",
                settings.server.host,
                settings.server.port,
            )
        yield

    app = FastAPI(
        title="VisionServeX",
        version=__version__,
        description=(
            "Serve permissive-license computer vision models locally and over "
            "Cloudflare Tunnel. By Arash Sajjadi, University of Saskatchewan."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=_lifespan,
    )

    if settings.cors.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.allowed_origins,
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods,
            allow_headers=settings.cors.allow_headers,
        )
    app.add_middleware(RateLimitMiddleware, limits=settings.limits)
    app.add_middleware(BodySizeLimitMiddleware, limits=settings.limits)
    app.add_middleware(RequestContextMiddleware)

    app.state.settings = settings

    _register_routes(app)
    # v3.22.0 — batch + video + job-lifecycle routes
    from visionservex.server.video_routes import build_media_router

    app.include_router(build_media_router())
    _register_exception_handlers(app)
    return app


# ---------- auth dep ----------


async def _require_auth(request: Request) -> None:
    settings: Settings = request.app.state.settings
    if not settings.auth.enabled:
        return
    outcome = authenticate_request(dict(request.headers), settings.auth)
    if not outcome.authenticated:
        raise unauthorized(outcome.reason or "authentication required")
    request.state.principal = outcome.principal
    request.state.auth_method = outcome.method


# ---------- routes ----------


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        settings = app.state.settings
        return HealthResponse(
            status="ok",
            version=__version__,
            public_mode=settings.server.public_mode,
            auth_enabled=settings.auth.enabled,
        )

    @app.get("/ready", tags=["meta"])
    async def ready() -> dict[str, Any]:
        sched = get_scheduler()
        cache = get_model_cache()
        return {
            "ready": True,
            "loaded_models": cache.info(),
            "scheduler": sched.stats(),
            "jobs": {"total": len(get_job_store().list())},
        }

    @app.get("/version", response_model=VersionResponse, tags=["meta"])
    async def version() -> VersionResponse:
        return VersionResponse(
            version=__version__,
            python=sys.version.split()[0],
            platform=platform.platform(),
        )

    @app.get("/gateway/status", tags=["meta"])
    async def gateway_status(request: Request) -> dict[str, Any]:
        """Local model gateway status — loaded models, device, queue, jobs."""
        settings: Settings = request.app.state.settings
        cache = get_model_cache()
        sched = get_scheduler()
        jobs = get_job_store().list()
        from visionservex.runtime.device import best_device

        best = best_device()
        return {
            "version": __version__,
            "bind": f"{settings.server.host}:{settings.server.port}",
            "public_mode": settings.server.public_mode,
            "auth_enabled": settings.auth.enabled,
            "auto_pull": settings.models.auto_pull,
            "auto_pull_policy": settings.models.auto_pull_policy,
            "best_device": best.to_dict(),
            "loaded_models": cache.info(),
            "scheduler": sched.stats(),
            "active_jobs": len(
                [j for j in jobs if j.status not in {"completed", "failed", "cancelled"}]
            ),
        }

    @app.post("/gateway/warmup", tags=["meta"])
    async def gateway_warmup(
        request: Request,
        model_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Preload a list of model IDs into the cache."""
        await _require_auth(request)
        if not model_ids:
            return {"warmed_up": []}
        cache = get_model_cache()
        ok = []
        errors = []
        for mid in model_ids:
            try:
                cache.get(mid)
                ok.append(mid)
            except Exception as exc:
                errors.append({"model_id": mid, "error": str(exc)[:100]})
        return {"warmed_up": ok, "errors": errors}

    @app.get("/devices", tags=["meta"])
    async def devices_endpoint() -> dict[str, Any]:
        return {"devices": [d.to_dict() for d in available_devices()]}

    @app.get("/models", response_model=ModelListResponse, tags=["models"])
    async def list_models() -> ModelListResponse:
        items = [
            ModelListItem(
                id=e.id,
                display_name=e.display_name,
                task=e.task,
                family=e.family,
                license=e.license,
                status=e.status,
                implementation_status=e.implementation_status,
                engine=e.engine,
                backend=e.backend,
                difficulty=e.difficulty,
                auto_download=e.auto_download,
                supported_devices=e.supported_devices,
                minimum_vram_gb=e.minimum_vram_gb,
                recommended_vram_gb=e.recommended_vram_gb,
                warnings=e.warnings,
            )
            for e in default_registry().list()
        ]
        return ModelListResponse(models=items)

    @app.get("/models/{model_id}", tags=["models"])
    async def model_info(model_id: str) -> dict[str, Any]:
        try:
            entry = default_registry().get(model_id)
        except RegistryError as exc:
            raise not_found("MODEL_NOT_FOUND", str(exc))
        info = entry.model_dump()
        info["cached"] = is_cached(entry)
        cp = cached_path(entry)
        info["cache_path"] = str(cp) if cp else None
        return info

    @app.post("/models/{model_id}/pull", tags=["models"])
    async def pull_endpoint(
        model_id: str,
        request: Request,
        wait: bool = Query(default=True, description="Wait for completion or return a job id."),
    ) -> dict[str, Any]:
        await _require_auth(request)
        try:
            entry = default_registry().get(model_id)
        except RegistryError as exc:
            raise not_found("MODEL_NOT_FOUND", str(exc))

        if wait:
            try:
                path = await asyncio.get_running_loop().run_in_executor(None, download, entry)
            except ManualDownloadRequired as exc:
                raise unprocessable(
                    "MANUAL_DOWNLOAD_REQUIRED", str(exc), hint=f"see {entry.upstream_url}"
                )
            except DownloadError as exc:
                raise unprocessable("DOWNLOAD_FAILED", str(exc))
            return {
                "request_id": request.state.request_id,
                "status": "completed",
                "model_id": model_id,
                "path": str(path),
            }

        job = _start_pull_job(entry)
        return {
            "request_id": request.state.request_id,
            "status": "queued",
            "job_id": job.job_id,
            "model_id": model_id,
            "progress_url": f"/jobs/{job.job_id}",
        }

    @app.post("/models/{model_id}/load", tags=["models"])
    async def load_model(model_id: str, request: Request) -> dict[str, Any]:
        await _require_auth(request)
        try:
            cache = get_model_cache()
            cache.get(model_id)
        except RegistryError as exc:
            raise not_found("MODEL_NOT_FOUND", str(exc))
        except MissingDependencyError as exc:
            raise unprocessable("ENGINE_UNAVAILABLE", str(exc))
        return {"request_id": request.state.request_id, "model_id": model_id, "loaded": True}

    @app.post("/models/{model_id}/unload", tags=["models"])
    async def unload_model(model_id: str, request: Request) -> dict[str, Any]:
        await _require_auth(request)
        ok = get_model_cache().unload(model_id)
        return {"request_id": request.state.request_id, "model_id": model_id, "unloaded": ok}

    # ---- jobs ----

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
    async def job_get(job_id: str, request: Request) -> JobResponse:
        await _require_auth(request)
        job = get_job_store().get(job_id)
        if job is None:
            raise not_found("JOB_NOT_FOUND", f"job {job_id!r} does not exist")
        return JobResponse(**job.to_dict())

    @app.get("/jobs/{job_id}/events", tags=["jobs"])
    async def job_events(
        job_id: str,
        request: Request,
        sse: bool = Query(default=False, description="Stream as Server-Sent Events"),
    ) -> Any:
        """Job progress — returns a snapshot or streams SSE when ``?sse=true``."""
        await _require_auth(request)
        job = get_job_store().get(job_id)
        if job is None:
            raise not_found("JOB_NOT_FOUND", f"job {job_id!r} does not exist")

        if not sse:
            return job.to_dict()

        # Real SSE stream — polls until terminal state or client disconnect
        import asyncio

        async def _event_stream():
            terminal = {"completed", "failed", "cancelled"}
            while True:
                if await request.is_disconnected():
                    break
                j = get_job_store().get(job_id)
                if j is None:
                    break
                import json as _json_mod

                data_line = _json_mod.dumps(j.to_dict(), default=str).replace("\n", " ")
                yield f"data: {data_line}\n\n"
                if j.status in terminal:
                    break
                await asyncio.sleep(0.5)

        from fastapi.responses import StreamingResponse as _SSEResp

        return _SSEResp(_event_stream(), media_type="text/event-stream")

    @app.delete("/jobs/{job_id}", tags=["jobs"])
    async def job_cancel(job_id: str, request: Request) -> dict[str, Any]:
        await _require_auth(request)
        ok = get_job_store().cancel(job_id)
        if not ok:
            raise not_found("JOB_NOT_FOUND", f"job {job_id!r} not cancellable")
        return {"job_id": job_id, "status": "cancelled"}

    # ---- prediction endpoints ----

    @app.post("/predict", response_model=PredictionResponse, tags=["inference"])
    async def predict(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form(...),
        prompts: str | None = Form(default=None),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request, image, model_id, prompts, wait_for_download=wait_for_download
        )

    @app.post("/detect", response_model=PredictionResponse, tags=["inference"])
    async def detect(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form("mock-detect"),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request,
            image,
            model_id,
            None,
            expected="detect",
            wait_for_download=wait_for_download,
        )

    @app.post("/segment", response_model=PredictionResponse, tags=["inference"])
    async def segment(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form("mock-segment"),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request,
            image,
            model_id,
            None,
            expected="segment",
            wait_for_download=wait_for_download,
        )

    @app.post("/segment/b64", response_model=PredictionResponse, tags=["inference"])
    async def segment_b64(request: Request, body: PromptRequest) -> PredictionResponse:
        """Segment with optional box/point prompts via base64 image body."""
        await _require_auth(request)
        pil, warns = await _read_json_image(request, body)
        options = body.options or {}
        # Parse box/point from options
        kwargs: dict = {}
        if "boxes" in options:
            kwargs["boxes"] = options["boxes"]
        if "points" in options:
            kwargs["points"] = options["points"]
        if "point_labels" in options:
            kwargs["point_labels"] = options["point_labels"]

        async def _do_seg():
            from visionservex.runtime.cache import get_model_cache

            cache = get_model_cache()
            model = cache.get(body.model_id)
            return model.predict(pil, **kwargs)

        try:
            result = await asyncio.get_running_loop().run_in_executor(None, lambda: _do_seg())
            if asyncio.iscoroutine(result):
                result = await result
        except RegistryError as exc:
            # Unknown model (e.g. a research-only sidecar that is not a runtime
            # registry model) is an expected condition — return a structured 404,
            # never a 500.
            raise not_found("MODEL_NOT_FOUND", str(exc)) from exc
        return _wire_response(request, result, warns)

    @app.post("/obb", response_model=PredictionResponse, tags=["inference"])
    async def obb(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form("mock-obb"),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request,
            image,
            model_id,
            None,
            expected="obb",
            wait_for_download=wait_for_download,
        )

    @app.post("/pose", response_model=PredictionResponse, tags=["inference"])
    async def pose(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form("mock-pose"),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request,
            image,
            model_id,
            None,
            expected="pose",
            wait_for_download=wait_for_download,
        )

    @app.post("/classify", response_model=PredictionResponse, tags=["inference"])
    async def classify(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form("mock-classify"),
        wait_for_download: bool = Query(default=True),
    ):
        return await _predict_endpoint(
            request,
            image,
            model_id,
            None,
            expected="classify",
            wait_for_download=wait_for_download,
        )

    @app.post("/open-vocab/detect", response_model=PredictionResponse, tags=["inference"])
    async def open_vocab_detect(
        request: Request,
        body: PromptRequest,
        wait_for_download: bool = Query(default=True),
    ):
        await _require_auth(request)
        pil, warns = await _read_json_image(request, body)
        return await _run_and_wire(
            request,
            body.model_id,
            pil,
            body.prompts,
            expected="open_vocab_detect",
            warns=warns,
            wait_for_download=wait_for_download,
        )

    @app.post("/grounded-segment", response_model=PredictionResponse, tags=["inference"])
    async def grounded_segment(
        request: Request,
        body: PromptRequest,
        wait_for_download: bool = Query(default=True),
    ):
        await _require_auth(request)
        pil, warns = await _read_json_image(request, body)
        return await _run_and_wire(
            request,
            body.model_id,
            pil,
            body.prompts,
            expected="grounded_segment",
            warns=warns,
            wait_for_download=wait_for_download,
        )

    @app.post("/batch-predict", response_model=list[PredictionResponse], tags=["inference"])
    async def batch_predict(
        request: Request,
        images: list[UploadFile] = File(...),
        model_id: str = Form(...),
    ) -> list[PredictionResponse]:
        await _require_auth(request)
        out: list[PredictionResponse] = []
        for upload in images:
            pil, warns = await _read_upload(request, upload)
            res = await _run_and_wire(request, model_id, pil, None, warns=warns)
            if isinstance(res, PredictionResponse):
                out.append(res)
            else:
                raise unprocessable(
                    "DOWNLOAD_REQUIRED",
                    "batch endpoint requires preloaded models; pull weights first",
                )
        return out

    @app.post("/predict/annotated", tags=["inference"])
    async def predict_annotated(
        request: Request,
        image: UploadFile = File(...),
        model_id: str = Form(...),
    ) -> Response:
        await _require_auth(request)
        pil, _ = await _read_upload(request, image)
        settings: Settings = request.app.state.settings
        result = await _do_predict(
            request, model_id, pil, None, auto_pull_allowed=settings.models.auto_pull
        )
        if not isinstance(result, BaseResult):
            raise unprocessable("DOWNLOAD_REQUIRED", "weights missing; pull first")
        annotated = result.plot(pil)
        return Response(content=encode_jpeg(annotated), media_type="image/jpeg")

    @app.get("/metrics", tags=["meta"])
    async def metrics_endpoint(request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        if settings.server.public_mode and settings.auth.enabled:
            await _require_auth(request)
        snapshot = metrics.snapshot()
        snapshot["scheduler"] = get_scheduler().stats()
        snapshot["models_loaded"] = get_model_cache().info()
        return snapshot

    @app.get("/metrics/prometheus", tags=["meta"], response_class=Response)
    async def metrics_prometheus(request: Request) -> Response:
        """Prometheus text-format metrics endpoint.

        Exposes counters, gauges, and latency observations in the standard
        Prometheus text exposition format for scraping by a Prometheus server.
        """
        settings: Settings = request.app.state.settings
        if settings.server.public_mode and settings.auth.enabled:
            await _require_auth(request)
        snapshot = metrics.snapshot()
        sched = get_scheduler().stats()

        lines: list[str] = [
            "# HELP visionservex_requests_total Total inference requests",
            "# TYPE visionservex_requests_total counter",
            f"visionservex_requests_total {snapshot.get('counters', {}).get('requests_total', 0)}",
            "",
            "# HELP visionservex_requests_rejected Total backpressure rejections",
            "# TYPE visionservex_requests_rejected counter",
            f"visionservex_requests_rejected {snapshot.get('counters', {}).get('requests_rejected_backpressure', 0)}",
            "",
            "# HELP visionservex_requests_timed_out Total timeout rejections",
            "# TYPE visionservex_requests_timed_out counter",
            f"visionservex_requests_timed_out {snapshot.get('counters', {}).get('requests_timed_out', 0)}",
            "",
            "# HELP visionservex_active_requests Currently inflight requests",
            "# TYPE visionservex_active_requests gauge",
            f"visionservex_active_requests {sched.get('total_inflight', 0)}",
            "",
            "# HELP visionservex_models_loaded Number of currently loaded models",
            "# TYPE visionservex_models_loaded gauge",
            f"visionservex_models_loaded {len(get_model_cache().info())}",
            "",
        ]
        # Latency quantiles
        obs = snapshot.get("observations", {})
        if "latency_ms" in obs:
            lat = obs["latency_ms"]
            lines += [
                "# HELP visionservex_latency_ms Inference latency in milliseconds",
                "# TYPE visionservex_latency_ms summary",
                f'visionservex_latency_ms{{quantile="0.5"}} {lat.get("p50", 0)}',
                f'visionservex_latency_ms{{quantile="0.9"}} {lat.get("p90", 0)}',
                f'visionservex_latency_ms{{quantile="0.99"}} {lat.get("p99", 0)}',
                f"visionservex_latency_ms_count {lat.get('n', 0)}",
                "",
            ]
        return Response(
            content="\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


# ---------- predict helpers ----------


async def _predict_endpoint(
    request: Request,
    image: UploadFile,
    model_id: str,
    prompts: str | None,
    *,
    expected: str | None = None,
    wait_for_download: bool = True,
):
    await _require_auth(request)
    pil, warns = await _read_upload(request, image)
    parsed = _parse_prompts(prompts)
    return await _run_and_wire(
        request,
        model_id,
        pil,
        parsed,
        expected=expected,
        warns=warns,
        wait_for_download=wait_for_download,
    )


async def _run_and_wire(
    request: Request,
    model_id: str,
    image: Image.Image,
    prompts: list[str] | None,
    *,
    expected: str | None = None,
    warns: list[str] | None = None,
    wait_for_download: bool = True,
):
    settings: Settings = request.app.state.settings
    # Validate registry first so we can be specific about auto-pull policy.
    try:
        entry = default_registry().get(model_id)
    except RegistryError as exc:
        raise not_found("MODEL_NOT_FOUND", str(exc))
    if expected and entry.task != expected:
        raise unprocessable(
            "TASK_MISMATCH",
            f"model {model_id!r} is for task {entry.task!r}, not {expected!r}",
            hint=f"see GET /models for compatible models for task {expected!r}",
        )

    auto_pull_allowed = _auto_pull_allowed(settings, entry)
    if not is_cached(entry) and entry.download_type != "synthetic":
        if not auto_pull_allowed:
            raise unprocessable(
                "MODEL_MISSING",
                f"Model weights for {model_id!r} are not available locally.",
                hint=f"Run: visionservex pull {model_id}",
            )
        if not wait_for_download:
            job = _start_pull_job(entry)
            return DownloadingResponse(
                request_id=request.state.request_id,
                status="downloading",
                model_id=model_id,
                job_id=job.job_id,
                message="Model weights are being downloaded.",
                progress_url=f"/jobs/{job.job_id}",
            )

    result = await _do_predict(
        request, model_id, image, prompts, auto_pull_allowed=auto_pull_allowed
    )
    return _wire_response(request, result, warns or [])


def _auto_pull_allowed(settings: Settings, entry) -> bool:
    if not settings.models.auto_pull:
        return False
    if (
        settings.server.public_mode
        and settings.models.auto_pull_require_auth
        and not settings.auth.enabled
    ):
        return False
    policy = settings.models.auto_pull_policy
    if policy == "never":
        return False
    if policy == "all_auto_downloadable":
        return entry.auto_download
    if policy == "easy_only":
        return entry.auto_download and entry.difficulty in {"very_easy", "easy"}
    if policy == "registry_allowed":
        return entry.auto_download
    return False


def _start_pull_job(entry) -> Job:
    store = get_job_store()
    job = store.create(model_id=entry.id, kind="pull")
    store.update(job.job_id, status="queued", message="queued for download")

    def _runner() -> None:
        try:

            def _cb(ev: DownloadProgress) -> None:
                store.update(
                    job.job_id,
                    status=(
                        "downloading"
                        if ev.phase == "downloading"
                        else "verifying"
                        if ev.phase == "verifying"
                        else "checking_dependencies"
                        if ev.phase == "starting"
                        else "loading_model"
                        if ev.phase == "loading"
                        else "downloading"
                    ),
                    message=ev.message,
                    progress=ev.to_dict(),
                )

            path = download(entry, progress=_cb)
            store.update(job.job_id, status="completed", message="ok", result={"path": str(path)})
        except ManualDownloadRequired as exc:
            store.update(
                job.job_id,
                status="failed",
                error={"code": "MANUAL_DOWNLOAD_REQUIRED", "message": str(exc)},
            )
        except DownloadError as exc:
            store.update(
                job.job_id, status="failed", error={"code": "DOWNLOAD_FAILED", "message": str(exc)}
            )
        except Exception as exc:
            store.update(
                job.job_id, status="failed", error={"code": "INTERNAL_ERROR", "message": str(exc)}
            )

    t = threading.Thread(target=_runner, daemon=True, name=f"pull-{entry.id}")
    t.start()
    return job


async def _do_predict(
    request: Request,
    model_id: str,
    image: Image.Image,
    prompts: list[str] | None,
    *,
    auto_pull_allowed: bool,
) -> BaseResult:
    sched = get_scheduler()
    cache = get_model_cache()

    try:
        async with sched.reserve(model_id):
            loop = asyncio.get_running_loop()

            def _call() -> BaseResult:
                model = cache.get(model_id)
                # Honor auto_pull at call time too — covers cases where weights
                # vanished after the cache entry was created.
                model.auto_pull = auto_pull_allowed
                return model.predict(image, prompts=prompts)

            return await loop.run_in_executor(None, _call)
    except BackpressureError as exc:
        metrics.error("BUSY")
        settings: Settings = request.app.state.settings
        retry_after = settings.runtime.server_busy_retry_after_s
        err = busy(
            str(exc),
            hint=f"Server is at capacity. Retry in ~{retry_after} seconds or reduce concurrency.",
        )
        # Attach Retry-After hint in the error so callers can respect it.
        err.error_details["retry_after_seconds"] = retry_after
        raise err
    except RequestTimeoutError as exc:
        metrics.error("TIMEOUT")
        raise unprocessable("TIMEOUT", str(exc), hint="raise request_timeout_s in config")
    except MissingDependencyError as exc:
        raise unprocessable(
            "ENGINE_UNAVAILABLE",
            str(exc),
            hint=f"install: {exc.install_hint}" if exc.install_hint else "",
        )


def _wire_response(request: Request, result: BaseResult, warns: list[str]) -> PredictionResponse:
    payload = result.to_dict()
    base_warns = list(result.warnings) + warns
    payload_results = _result_payload(payload)
    return PredictionResponse(
        request_id=request.state.request_id,
        status="completed",
        model_id=result.model_id,
        task=result.task,
        backend=result.backend or "",
        device=result.device,
        precision=result.precision,
        latency_ms=result.latency_ms,
        model_loaded_from=result.model_loaded_from,
        cache_path=result.cache_path,
        fallback_reason=result.fallback_reason,
        results=payload_results,
        warnings=base_warns,
        metadata=dict(result.metadata),
    )


async def _read_upload(request: Request, upload: UploadFile) -> tuple[Image.Image, list[str]]:
    settings: Settings = request.app.state.settings
    warns: list[str] = []
    try:
        validate_mime_type(upload.content_type, settings.inputs)
    except InputValidationError as exc:
        raise unprocessable(
            "BAD_MIME_TYPE", str(exc), hint=f"allowed: {settings.inputs.allowed_mime_types}"
        )
    data = await upload.read()
    try:
        validate_image_bytes(data, settings.limits)
    except InputValidationError as exc:
        msg = str(exc)
        if "exceeds max_upload_bytes" in msg:
            raise too_large(msg, hint="send a smaller image")
        raise unprocessable("BAD_IMAGE", msg)
    pil = open_safe(
        data,
        max_pixels=settings.limits.max_image_pixels,
        max_dim=settings.limits.max_image_dim,
    )
    return pil, warns


async def _read_json_image(request: Request, body: PromptRequest) -> tuple[Image.Image, list[str]]:
    settings: Settings = request.app.state.settings
    warns: list[str] = []

    if body.image_b64 and body.image_url:
        raise bad_request("BAD_REQUEST", "specify exactly one of image_b64 or image_url")
    if not body.image_b64 and not body.image_url:
        raise bad_request("BAD_REQUEST", "image_b64 or image_url is required")

    if body.image_url:
        if not settings.inputs.allow_url_inputs:
            raise forbidden(
                "url inputs are disabled",
                hint="set inputs.allow_url_inputs=true to enable",
            )
        try:
            url = validate_remote_url(
                body.image_url,
                allow_http=False,
                allowlist_hosts=settings.inputs.url_allowlist_hosts or None,
            )
        except URLValidationError as exc:
            raise unprocessable("BAD_URL", str(exc))
        try:
            async with httpx.AsyncClient(
                timeout=settings.inputs.url_request_timeout_s, follow_redirects=False
            ) as client:
                resp = await client.get(url)
        except httpx.HTTPError as exc:
            raise unprocessable("URL_FETCH_FAILED", f"could not fetch image: {exc}")
        if resp.status_code != 200:
            raise unprocessable("URL_FETCH_FAILED", f"HTTP {resp.status_code}")
        content = resp.content
        if len(content) > settings.inputs.url_max_bytes:
            raise too_large("remote image too large")
        data = content
    else:
        try:
            data = base64.b64decode(body.image_b64, validate=True)
        except Exception as exc:
            raise unprocessable("BAD_BASE64", f"invalid base64: {exc}")

    try:
        validate_image_bytes(data, settings.limits)
    except InputValidationError as exc:
        raise unprocessable("BAD_IMAGE", str(exc))
    pil = open_safe(
        data,
        max_pixels=settings.limits.max_image_pixels,
        max_dim=settings.limits.max_image_dim,
    )
    return pil, warns


def _parse_prompts(prompts: str | None) -> list[str] | None:
    if prompts is None or prompts == "":
        return None
    if prompts.startswith("["):
        try:
            import json

            data = json.loads(prompts)
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    return [p.strip() for p in prompts.split(",") if p.strip()]


def _result_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("detections", "segments", "persons", "top_k"):
        if key in payload:
            value = payload[key]
            if key == "top_k":
                return [{"label": lbl, "score": score} for lbl, score in value]
            return list(value)
    return []


def _register_exception_handlers(app: FastAPI) -> None:
    from visionservex.exceptions import ModelLicenseError

    @app.exception_handler(ModelLicenseError)
    async def _license_error_handler(request: Request, exc: ModelLicenseError) -> JSONResponse:
        # The commercial-safe policy applies over HTTP too: restricted models are
        # refused with a clean 403 + structured code (never a 500 or a fake result).
        rid = getattr(request.state, "request_id", request_id())
        return JSONResponse(
            status_code=403,
            content={
                "request_id": rid,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "hint": exc.hint,
                    "details": exc.details,
                },
            },
        )

    @app.exception_handler(ApiError)
    async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        rid = getattr(request.state, "request_id", request_id())
        headers: dict[str, str] = {}
        # Add Retry-After header for busy responses so compliant clients back off.
        retry_after = exc.error_details.get("retry_after_seconds")
        if retry_after is not None:
            headers["Retry-After"] = str(int(retry_after))
        return JSONResponse(
            status_code=exc.status_code,
            headers=headers,
            content={
                "request_id": rid,
                "error": {
                    "code": exc.error_code,
                    "message": exc.error_message,
                    "hint": exc.error_hint,
                    "details": exc.error_details,
                },
            },
        )


__all__ = ["create_app"]

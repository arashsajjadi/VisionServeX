# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""OpenMMLab Docker sidecar proxy engine.

When the OpenMMLab sidecar container is running (see docker/openmmlab/),
this engine forwards prediction requests to it rather than requiring a
local OpenMMLab install.

Configure via:
    VISIONSERVEX_OPENMMLAB_SIDECAR_URL=http://localhost:8090

The sidecar exposes:
    GET  /health
    GET  /models
    POST /predict/pose
    POST /predict/obb
    POST /predict/segment
    POST /predict/classify
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import (
    BaseResult,
    ClassificationResult,
    Keypoint,
    OrientedBox,
    OrientedDetection,
    OrientedDetectionResult,
    PoseInstance,
    PoseResult,
    SegmentationResult,
)
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

_SIDECAR_URL_ENV = "VISIONSERVEX_OPENMMLAB_SIDECAR_URL"
_DEFAULT_SIDECAR_URL = "http://localhost:8090"


def _sidecar_url() -> str:
    return os.environ.get(_SIDECAR_URL_ENV, _DEFAULT_SIDECAR_URL)


def _sidecar_health() -> bool:
    try:
        import httpx  # type: ignore

        r = httpx.get(f"{_sidecar_url()}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


class OpenMMLabSidecarEngine(StubEngine):
    """Forwards requests to a running OpenMMLab Docker sidecar.

    Falls back to a clear error if the sidecar is not reachable.
    """

    real_install_extra = "openmmlab"
    real_modules = ("httpx",)
    backend_label = "openmmlab_sidecar"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._sidecar_ok = False
        self._route = self._resolve_route(entry.task)

    @staticmethod
    def _resolve_route(task: str) -> str:
        return {
            "pose": "pose",
            "obb": "obb",
            "segment": "segment",
            "classify": "classify",
        }.get(task, "pose")

    def _real_load(self, *, device: str, precision: str) -> None:
        try:
            import httpx  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError(
                "httpx is required for sidecar communication",
                install_hint="pip install httpx",
            ) from exc

        if not _sidecar_health():
            raise MissingDependencyError(
                f"OpenMMLab sidecar is not reachable at {_sidecar_url()}. "
                f"Start it with: visionservex openmmlab docker-run",
                install_hint=(
                    "visionservex openmmlab docker-build && visionservex openmmlab docker-run"
                ),
            )
        self._sidecar_ok = True

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready or not self._sidecar_ok:
            return super().predict(image, prompts=prompts, **kwargs)

        try:
            import httpx  # type: ignore

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=90)
            buf.seek(0)

            r = httpx.post(
                f"{_sidecar_url()}/predict/{self._route}",
                files={"image": ("image.jpg", buf, "image/jpeg")},
                data={"model_id": self.entry.id},
                timeout=30.0,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            raise RuntimeError(f"Sidecar request failed for {self.entry.id!r}: {exc}") from exc

        return self._parse_response(data, image)

    def _parse_response(self, data: dict, image: Image.Image) -> BaseResult:
        w, h = image.size
        task = self.entry.task
        warnings = [data.get("note")] if data.get("note") else []

        if task == "pose":
            persons = []
            for p in data.get("persons", []):
                kps = [Keypoint(x=kp[0], y=kp[1], score=kp[2]) for kp in p.get("keypoints", [])]
                persons.append(PoseInstance(box=None, score=p.get("score", 1.0), keypoints=kps))
            return PoseResult(
                kind="pose",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                device=self.device,
                backend=self.backend_label,
                persons=persons,
                warnings=warnings,
            )

        if task == "obb":
            dets = []
            for d in data.get("detections", []):
                dets.append(
                    OrientedDetection(
                        box=OrientedBox(
                            cx=d.get("cx", 0),
                            cy=d.get("cy", 0),
                            w=d.get("w", 0),
                            h=d.get("h", 0),
                            theta=d.get("theta", 0),
                        ),
                        score=d.get("score", 0),
                        label=d.get("label", ""),
                    )
                )
            return OrientedDetectionResult(
                kind="obb",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                device=self.device,
                backend=self.backend_label,
                detections=dets,
                warnings=warnings,
            )

        if task == "classify":
            top_k = [(item["label"], item["score"]) for item in data.get("top_k", [])]
            return ClassificationResult(
                kind="classification",
                model_id=self.entry.id,
                task=task,
                image_size=(w, h),
                device=self.device,
                backend=self.backend_label,
                top_k=top_k,
                warnings=warnings,
            )

        # Default: segmentation
        return SegmentationResult(
            kind="segmentation",
            model_id=self.entry.id,
            task=task,
            image_size=(w, h),
            device=self.device,
            backend=self.backend_label,
            segments=[],
            warnings=warnings,
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> OpenMMLabSidecarEngine:
    return OpenMMLabSidecarEngine(entry)


# Register for all OpenMMLab families that support the Docker sidecar path
register_engine("openmmlab_sidecar", _factory)

__all__ = ["OpenMMLabSidecarEngine", "_sidecar_health", "_sidecar_url"]

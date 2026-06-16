# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""LibreYOLO inference engine.

Wires the permissively-licensed ``libreyolo`` package (MIT code; Apache-2.0 /
MIT weights) into the VisionServeX engine registry so its detectors are
reachable through the standard ``VisionModel`` / ``visionservex detect`` / HTTP
server paths — not only the dedicated ``visionservex libreyolo`` CLI subcommand.

LibreYOLO is the project's permissive alternative to AGPL-licensed Ultralytics
YOLO and YOLO-World. Only permissive families are exposed as runnable engines:

    libreyolo-yolox-*    YOLOX               (Apache-2.0)
    libreyolo-yolov9-*   YOLOv9              (MIT — MultimediaTechLab fork, not GPL)
    libreyolo-rtdetr-*   RT-DETR             (Apache-2.0)
    libreyolo-dfine-*    D-FINE              (Apache-2.0)

YOLO-NAS (Deci proprietary, non-commercial) is intentionally NOT mapped here.

Install:
    pip install 'visionservex[libreyolo]'

Weights are pulled on demand from the official ``LibreYOLO`` Hugging Face org
through the standard VisionServeX downloader. VisionServeX never bundles them
(``can_ship_weights`` stays False in the license policy table).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, Box, Detection, DetectionResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Permissive LibreYOLO families only. YOLO-NAS (non-commercial) is excluded by
# design — it must never be reachable as a default-safe runnable engine.
_FAMILY_TO_CLASS: dict[str, str] = {
    "yolox": "LibreYOLOX",
    "yolov9": "LibreYOLO9",
    "rtdetr": "LibreYOLORTDETR",
    "dfine": "LibreDFINE",
}


def _parse_model_id(model_id: str) -> tuple[str, str] | None:
    """Map ``libreyolo-<family>-<size>`` → ``(family, size)``.

    Mirrors ``visionservex.cli.libreyolo_commands._libreyolo_id_to_libre_class``
    so the engine and the CLI agree on id parsing (e.g. ``libreyolo-rtdetr-r50m``
    → ``("rtdetr", "r50m")``).
    """
    if not model_id.startswith("libreyolo-"):
        return None
    parts = model_id[len("libreyolo-") :].split("-")
    if len(parts) < 2:
        return None
    family = parts[0]
    size = "-".join(parts[1:])
    return family, size


def _load_libreyolo_class(class_name: str):
    """Import and return a libreyolo model class by name (kept lazy)."""
    try:
        import libreyolo  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "libreyolo package is not installed",
            install_hint="pip install 'visionservex[libreyolo]'",
        ) from exc
    cls = getattr(libreyolo, class_name, None)
    if cls is None:
        raise MissingDependencyError(
            f"libreyolo.{class_name} not found in installed libreyolo package",
            install_hint="pip install --upgrade 'visionservex[libreyolo]'",
        )
    return cls


class LibreYOLOEngine(StubEngine):
    """Real LibreYOLO detection engine backed by the ``libreyolo`` package."""

    real_install_extra = "libreyolo"
    real_modules = ("libreyolo",)
    backend_label = "libreyolo_package"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None

    # ------ lifecycle ------

    def _real_load(self, *, device: str, precision: str) -> None:
        parsed = _parse_model_id(self.entry.id)
        if parsed is None:
            raise MissingDependencyError(
                f"{self.entry.id!r} is not a libreyolo-<family>-<size> id",
                install_hint="check `visionservex list-models` for supported ids",
            )
        family, size = parsed
        class_name = _FAMILY_TO_CLASS.get(family)
        if class_name is None:
            raise MissingDependencyError(
                f"libreyolo family {family!r} is not exposed as a runnable engine "
                f"(permissive families only: {sorted(_FAMILY_TO_CLASS)})",
                install_hint="see docs/model_license_policy.md",
            )
        cls = _load_libreyolo_class(class_name)

        # Resolve weights via the standard VisionServeX downloader (HF org
        # `LibreYOLO`). Pulled on demand to the local cache; never bundled.
        from visionservex.runtime.downloads import cached_path, download

        weight_path = cached_path(self.entry)
        if weight_path is None:
            weight_path = download(self.entry)

        ly_device = "cuda" if str(device).startswith("cuda") else str(device)
        _log.info(
            "loading %s (size=%s) from %s on device=%s",
            class_name,
            size,
            weight_path,
            ly_device,
        )
        self._model = cls(model_path=str(weight_path), size=size, device=ly_device)
        _log.info("%s ready", class_name)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        super().unload()

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (640, 640), "black")
            self._model(dummy, save=False)
        except Exception:  # pragma: no cover - best-effort warmup only
            pass

    # ------ inference ------

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        threshold: float = 0.25,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        conf = float(kwargs.get("conf", threshold))
        # libreyolo's ImageLoader accepts a PIL Image directly and normalizes to
        # RGB, so we forward the decoded image without a temp file.
        result = self._model(image, conf=conf, save=False)
        return self._to_result(result, image)

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        """Unused: predict() is overridden to drive the package directly."""
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        """Unused: predict() is overridden to drive the package directly."""
        return self._mock.postprocess(raw, image=image, **kwargs)

    def _to_result(self, result: Any, image: Image.Image) -> BaseResult:
        """Convert a LibreYOLO ``Results`` object → VisionServeX DetectionResult."""
        w, h = image.size
        detections = _libre_to_detections(result, self._model)
        return DetectionResult(
            kind="detection",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=(w, h),
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            detections=detections,
        )


def _libre_to_detections(result: Any, model: Any) -> list[Detection]:
    """Convert a LibreYOLO ``Results`` (or list of them) → ``list[Detection]``.

    LibreYOLO returns a ``Results`` object whose ``.boxes`` exposes ``.xyxy``,
    ``.conf`` and ``.cls`` tensors; class names come from the model's COCO map.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None and isinstance(result, list) and result:
        boxes = getattr(result[0], "boxes", None)
    if boxes is None:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls_ = boxes.cls.detach().cpu().numpy()

    # Prefer the model's own id→name map, then any result/boxes-supplied names.
    names = getattr(model, "names", None)
    if not isinstance(names, dict):
        names = getattr(result, "names", None)
    if not isinstance(names, dict):
        names = getattr(boxes, "names", None)
    if not isinstance(names, dict):
        names = {}

    out: list[Detection] = []
    for i in range(len(xyxy)):
        b = xyxy[i].tolist()
        cid = int(cls_[i])
        out.append(
            Detection(
                box=Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                score=float(conf[i]),
                label=names.get(cid, f"class_{cid}"),
                class_id=cid,
            )
        )
    return out


def _factory(entry: ModelEntry) -> LibreYOLOEngine:
    return LibreYOLOEngine(entry)


register_engine("libreyolo", _factory)

__all__ = ["LibreYOLOEngine"]

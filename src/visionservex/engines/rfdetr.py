# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""RF-DETR inference engine.

Real backend uses the Roboflow ``rfdetr`` PyPI package which auto-downloads
weights on first instantiation and wraps them behind a clean predict API.

Install:
    pip install 'visionservex[rfdetr]'

The engine maps VisionServeX model IDs to rfdetr class names, instantiates
the right class, and converts ``supervision.Detections`` to our stable result
schema.

Supported model IDs:
    rfdetr-nano, rfdetr-small, rfdetr-base, rfdetr-medium, rfdetr-large
    rfdetr-seg-nano, rfdetr-seg-small, rfdetr-seg-medium

Segmentation models return masks via sv.Detections.mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    Detection,
    DetectionResult,
    Segment,
    SegmentationResult,
)
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import EngineError, MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Map model id → rfdetr class name (string to keep import lazy)
_DETECT_CLASSES: dict[str, str] = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-base": "RFDETRBase",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
}
_SEG_CLASSES: dict[str, str] = {
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
    "rfdetr-seg-xlarge": "RFDETRSegXLarge",
    "rfdetr-seg-2xlarge": "RFDETRSeg2XLarge",
}
_ALL_CLASSES = {**_DETECT_CLASSES, **_SEG_CLASSES}


def _load_rfdetr_class(class_name: str):
    """Import and return the rfdetr class by name."""
    try:
        import rfdetr  # type: ignore
    except ImportError as exc:
        raise MissingDependencyError(
            "rfdetr package is not installed",
            install_hint="pip install 'visionservex[rfdetr]'",
        ) from exc
    cls = getattr(rfdetr, class_name, None)
    if cls is None:
        raise MissingDependencyError(
            f"rfdetr.{class_name} not found in installed rfdetr package",
            install_hint="pip install --upgrade 'visionservex[rfdetr]'",
        )
    return cls


def trigger_package_download(entry: ModelEntry) -> Path:
    """Instantiate the rfdetr model on CPU to trigger checkpoint download.

    The rfdetr package downloads to ``~/.cache/rfdetr/`` and verifies the MD5
    hash. After the first call, the weights are cached and subsequent loads
    skip the download. Returns the rfdetr internal cache directory path so
    VisionServeX can write a manifest.
    """
    class_name = _ALL_CLASSES.get(entry.id)
    if not class_name:
        from visionservex.runtime.downloads import model_dir

        return model_dir(entry)

    cls = _load_rfdetr_class(class_name)
    _log.info("triggering rfdetr weight download for %s via %s", entry.id, class_name)
    instance = cls(device="cpu")  # instantiation triggers maybe_download_pretrain_weights()
    del instance

    # Locate the weight file in the rfdetr package cache.
    import pathlib

    try:
        pkg_cache = pathlib.Path.home() / ".cache" / "rfdetr"
        if pkg_cache.exists():
            return pkg_cache
    except Exception:
        pass
    from visionservex.runtime.downloads import model_dir

    return model_dir(entry)


class RFDETREngine(StubEngine):
    """Real RF-DETR and RF-DETR-Seg engine backed by the `rfdetr` package."""

    real_install_extra = "rfdetr"
    real_modules = ("rfdetr", "supervision")
    backend_label = "rfdetr_package"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._rfdetr_model: Any = None
        self._is_seg: bool = entry.id in _SEG_CLASSES
        # When set (via load_checkpoint), a trained .pth overrides the base
        # pretrained weights as the sole weight source.
        self._checkpoint_override: Path | None = None

    # ------ lifecycle ------

    def _real_load(self, *, device: str, precision: str) -> None:
        class_name = _ALL_CLASSES.get(self.entry.id)
        if not class_name:
            raise MissingDependencyError(
                f"no rfdetr class mapping for model id {self.entry.id!r}",
                install_hint="check visionservex list-models for supported ids",
            )
        cls = _load_rfdetr_class(class_name)
        # rfdetr accepts a device string; use 'cuda' or 'cpu'.
        # AMP is on by default in rfdetr (fp16 on GPU, fp32 on CPU).
        if self._checkpoint_override is not None:
            ckpt = self._checkpoint_override
            if not Path(ckpt).is_file():
                raise MissingDependencyError(
                    f"trained checkpoint not found: {ckpt}",
                    install_hint="pass an existing rfdetr training checkpoint (.pth)",
                )
            _log.info(
                "loading %s from trained checkpoint %s on device=%s", class_name, ckpt, device
            )
            self._rfdetr_model = cls(pretrain_weights=str(ckpt), device=device)
        else:
            _log.info("loading %s on device=%s", class_name, device)
            self._rfdetr_model = cls(device=device)
        _log.info("%s ready", class_name)

    def unload(self) -> None:
        if self._rfdetr_model is not None:
            del self._rfdetr_model
            self._rfdetr_model = None
        super().unload()

    def warmup(self) -> None:
        if not self._real_ready:
            return
        try:
            dummy = Image.new("RGB", (384, 384), "black")
            self._rfdetr_model.predict(dummy, threshold=0.3)
        except Exception:
            pass

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        precision: str = "fp32",
    ) -> RFDETREngine:
        """Load a trained RF-DETR checkpoint (``.pth``) for inference.

        Re-instantiates the rfdetr model with ``pretrain_weights`` so the trained
        weights are the sole source — no base-weight fallback. Raises
        :class:`MissingDependencyError` if the file is absent. RF-DETR training
        itself is performed through the mature ``rfdetr`` package's native API.
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise MissingDependencyError(
                f"trained checkpoint not found: {ckpt}",
                install_hint="train RF-DETR via the rfdetr package and pass its output .pth",
            )
        if self._real_ready or self._rfdetr_model is not None:
            self.unload()
        self._real_ready = False
        self._checkpoint_override = ckpt
        self.load(device=device or self.device or "cpu", precision=precision)
        return self

    # ------ inference ------

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        # rfdetr can accept PIL directly
        dets = self._rfdetr_model.predict(image, threshold=threshold)
        return self._sv_to_result(dets, image)

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        """Not called when predict() is overridden."""
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        """Not called when predict() is overridden."""
        return self._mock.postprocess(raw, image=image, **kwargs)

    def export(self, format: str, output_path: Path) -> Path:
        """Export the loaded/trained RF-DETR model to ONNX via the native exporter.

        Delegates to the ``rfdetr`` package's own ``model.export()`` (Apache-2.0),
        which writes an ONNX file to a directory; we move it to ``output_path``.
        Only ONNX is supported.
        """
        fmt = format.lower()
        if fmt != "onnx":
            raise NotImplementedError(f"RFDETREngine supports ONNX export only, not {format!r}")
        if not self._real_ready:
            self.load(device=self.device or "cpu", precision=self.precision)

        import shutil
        import tempfile

        out = Path(output_path)
        with tempfile.TemporaryDirectory() as td:
            self._rfdetr_model.export(output_dir=td, opset_version=17, verbose=False)
            onnx_files = sorted(Path(td).rglob("*.onnx"))
            if not onnx_files:
                raise EngineError("RF-DETR native export produced no .onnx file")
            # Prefer the full inference model over a backbone-only export.
            chosen = next(
                (p for p in onnx_files if "backbone" not in p.name.lower()), onnx_files[0]
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(chosen), str(out))
        return out

    def _sv_to_result(self, dets: Any, image: Image.Image) -> BaseResult:
        """Convert supervision.Detections → VisionServeX result."""
        w, h = image.size
        class_names = getattr(self._rfdetr_model, "class_names", None) or []

        if self._is_seg:
            segments = _sv_to_segments(dets, class_names)
            return SegmentationResult(
                kind="segmentation",
                model_id=self.entry.id,
                task=self.entry.task,
                image_size=(w, h),
                device=self.device,
                precision=self.precision,
                backend=self.backend_label,
                segments=segments,
            )
        else:
            detections = _sv_to_detections(dets, class_names)
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


def _sv_to_detections(dets: Any, class_names: list[str]) -> list[Detection]:
    """Convert sv.Detections to our Detection list.

    v2.18.0 fix: RF-DETR returns *official* COCO category ids (1..90 with
    gaps). Pre-v2.18 the engine looked them up in a contiguous 0..79
    label table, producing wrong labels (the v17 "cake/carrot/toilet on a
    street scene" bug). We now always remap through the canonical
    :mod:`visionservex.data.coco_mapping` table.
    """
    from visionservex.data.coco_mapping import (
        COCO80_CONTIGUOUS_LABELS,
        is_official_id_set,
        remap_official_to_contiguous,
    )

    out: list[Detection] = []
    if dets is None or len(dets) == 0:
        return out
    xyxy = dets.xyxy
    conf = dets.confidence if dets.confidence is not None else np.ones(len(xyxy))
    class_ids = dets.class_id if dets.class_id is not None else np.zeros(len(xyxy), dtype=int)
    raw_ids = [int(c) for c in class_ids]

    # Detect whether the engine returned official-COCO ids (any id > 79 is
    # a tell-tale sign — contiguous max is 79). If so, every id flows
    # through the official→contiguous remap regardless of whatever
    # `dets.data["class_name"]` may have claimed.
    uses_official = is_official_id_set(raw_ids)

    # The rfdetr package sometimes ships its own `class_name` array in
    # `dets.data["class_name"]`. When ids are official, that array is
    # **wrong** because the upstream package indexed a contiguous label
    # table. So we trust the remap, not the rfdetr-supplied names.
    names_arr = (dets.data or {}).get("class_name", None) if not uses_official else None

    for i in range(len(xyxy)):
        box = xyxy[i].tolist()
        cid_raw = raw_ids[i]
        if uses_official:
            # Official COCO ids → always remap via the canonical table.
            cid_contiguous, label, _src = remap_official_to_contiguous(cid_raw)
        else:
            # Contiguous ids: prefer rfdetr's own class_name array (correct
            # for both COCO and fine-tuned custom datasets), then the model's
            # class_names list, then the canonical COCO80 as a last resort.
            cid_contiguous = cid_raw
            if names_arr is not None and i < len(names_arr):
                label = str(names_arr[i])
            elif 0 <= cid_raw < len(class_names):
                label = class_names[cid_raw]
            elif 0 <= cid_raw < len(COCO80_CONTIGUOUS_LABELS):
                label = COCO80_CONTIGUOUS_LABELS[cid_raw]
            else:
                label = f"class_{cid_raw}"
                cid_contiguous = -1
        final_class_id = cid_contiguous if cid_contiguous >= 0 else cid_raw
        out.append(
            Detection(
                box=Box(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])),
                score=float(conf[i]),
                label=label,
                class_id=int(final_class_id),
            )
        )
    return out


def _sv_to_segments(dets: Any, class_names: list[str]) -> list[Segment]:
    """Convert sv.Detections (with mask) to our Segment list.

    v2.18.0 fix: same official→contiguous remap as :func:`_sv_to_detections`.
    """
    from visionservex.data.coco_mapping import (
        COCO80_CONTIGUOUS_LABELS,
        is_official_id_set,
        remap_official_to_contiguous,
    )

    out: list[Segment] = []
    if dets is None or len(dets) == 0:
        return out
    xyxy = dets.xyxy
    conf = dets.confidence if dets.confidence is not None else np.ones(len(xyxy))
    class_ids = dets.class_id if dets.class_id is not None else np.zeros(len(xyxy), dtype=int)
    masks = dets.mask if dets.mask is not None else [None] * len(xyxy)
    raw_ids = [int(c) for c in class_ids]
    uses_official = is_official_id_set(raw_ids)
    names_arr = (dets.data or {}).get("class_name", None) if not uses_official else None

    for i in range(len(xyxy)):
        box = xyxy[i].tolist()
        cid_raw = raw_ids[i]
        if uses_official:
            cid_contiguous, label, _src = remap_official_to_contiguous(cid_raw)
        else:
            cid_contiguous = cid_raw
            if names_arr is not None and i < len(names_arr):
                label = str(names_arr[i])
            elif 0 <= cid_raw < len(class_names):
                label = class_names[cid_raw]
            elif 0 <= cid_raw < len(COCO80_CONTIGUOUS_LABELS):
                label = COCO80_CONTIGUOUS_LABELS[cid_raw]
            else:
                label = f"class_{cid_raw}"
                cid_contiguous = -1
        final_class_id = cid_contiguous if cid_contiguous >= 0 else cid_raw
        mask_arr = masks[i] if masks[i] is not None else np.zeros((1, 1), dtype=np.uint8)
        if hasattr(mask_arr, "astype"):
            mask_arr = mask_arr.astype(np.uint8)
        out.append(
            Segment(
                box=Box(x1=float(box[0]), y1=float(box[1]), x2=float(box[2]), y2=float(box[3])),
                score=float(conf[i]),
                label=label,
                mask=mask_arr,
                class_id=int(final_class_id),
            )
        )
    return out


def _factory(entry: ModelEntry) -> RFDETREngine:
    return RFDETREngine(entry)


register_engine("rfdetr", _factory)

__all__ = ["RFDETREngine", "trigger_package_download"]

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

import contextlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, Box, Detection, DetectionResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import EngineError, MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class TrainingNotSupportedError(EngineError):
    """Raised when training is requested for a non-trainable LibreYOLO family.

    YOLO-NAS (Deci proprietary, non-commercial) and any non-permissive family
    must never train through VisionServeX.
    """


# Permissive LibreYOLO families only. YOLO-NAS (non-commercial) is excluded by
# design — it must never be reachable as a default-safe runnable engine.
_FAMILY_TO_CLASS: dict[str, str] = {
    "yolox": "LibreYOLOX",
    "yolov9": "LibreYOLO9",
    "rtdetr": "LibreYOLORTDETR",
    "dfine": "LibreDFINE",
}

# Trainable families == the runnable permissive families. YOLO-NAS is
# deliberately absent: training it would be a non-commercial / proprietary path.
# Single source of truth — derived from _FAMILY_TO_CLASS so the two can't drift.
_TRAINABLE_FAMILIES: frozenset[str] = frozenset(_FAMILY_TO_CLASS)


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


def _force_eval(model: Any) -> None:
    """Put a libreyolo model's inner torch module into eval mode.

    libreyolo's ``_load_weights`` rebuilds the inner ``nn.Module`` via
    ``_rebuild_for_new_classes`` when a checkpoint's class count differs from
    COCO-80, and that rebuilt module is left in *training* mode. The detection
    head then takes its training branch at ``predict()`` and crashes
    (``'NoneType' object has no attribute 'sum'``). Forcing eval makes a
    reloaded trained checkpoint behave exactly like base-weight inference; it is
    a no-op for base weights (which already load in eval mode). This is the
    v3.14.0 trained-checkpoint reload fix.
    """
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "eval"):
        inner.eval()


class LibreYOLOEngine(StubEngine):
    """Real LibreYOLO detection engine backed by the ``libreyolo`` package."""

    real_install_extra = "libreyolo"
    real_modules = ("libreyolo",)
    backend_label = "libreyolo_package"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        # When set (via load_checkpoint), this trained .pt overrides the
        # on-demand HF base-weight download as the sole weight source.
        self._checkpoint_override: Path | None = None
        # The imgsz the checkpoint was trained at (read from its config). predict()
        # must infer at the SAME resolution; the model's native input size (e.g.
        # 640) otherwise yields low-confidence/empty boxes for a model fine-tuned
        # at a different imgsz.
        self._trained_imgsz: int | None = None

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

        # Weight source: an explicitly-supplied trained checkpoint (set via
        # ``load_checkpoint``) takes precedence over the on-demand HF base-weight
        # download. This is what makes trained-checkpoint reload real — there is
        # no silent fall back to base/pretrained weights once a checkpoint is set.
        if self._checkpoint_override is not None:
            weight_path = self._checkpoint_override
            if not Path(weight_path).is_file():
                raise MissingDependencyError(
                    f"trained checkpoint not found: {weight_path}",
                    install_hint="pass an existing best.pt/last.pt from LibreYOLOEngine.train()",
                )
            # Read the imgsz this checkpoint was trained at so predict() infers at
            # the matching resolution (see _trained_imgsz).
            try:
                import torch  # type: ignore

                _ck = torch.load(str(weight_path), map_location="cpu", weights_only=False)
                _isz = (
                    int((_ck.get("config") or {}).get("imgsz") or 0) if isinstance(_ck, dict) else 0
                )
                self._trained_imgsz = _isz or None
                del _ck
            except Exception:  # pragma: no cover - best-effort metadata read
                self._trained_imgsz = None
            _log.info("loading %s from trained checkpoint %s", class_name, weight_path)
        else:
            # Resolve base weights via the standard VisionServeX downloader (HF
            # org `LibreYOLO`). Pulled on demand to the local cache; never bundled.
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
        _force_eval(self._model)
        # Infer at the resolution the checkpoint was trained at (not the native
        # input size) so a model fine-tuned at e.g. 320 predicts confidently.
        if self._checkpoint_override is not None and self._trained_imgsz:
            with contextlib.suppress(Exception):
                self._model.input_size = self._trained_imgsz
                _log.info("predict input_size set to trained imgsz=%d", self._trained_imgsz)
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

    # ------ training / fine-tuning ------

    def train(
        self,
        dataset: str | Path,
        *,
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        device: str | None = None,
        lr0: float | None = None,
        workers: int = 4,
        seed: int = 0,
        project: str | None = None,
        name: str | None = None,
        exist_ok: bool = False,
        resume: bool = False,
        amp: bool | None = None,
        patience: int = 50,
        ema: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        """Train / fine-tune this LibreYOLO detector on a YOLO-format dataset.

        Args:
            dataset: Path to a YOLO ``data.yaml`` or a directory containing one.
            epochs/batch/imgsz/workers/seed/project/name/exist_ok/resume/patience:
                forwarded to the libreyolo trainer.
            device: ``None``/``"auto"`` auto-detects GPU; or pass ``"cuda"``/``"cpu"``.
            lr0/amp: omitted → the family's own tuned default is used.
            ema: EMA is **disabled by default** (v3.16.0). libreyolo's EMA decay
                (0.9998) lags ~99% toward the initial weights for short fine-tunes,
                so the saved EMA checkpoint produces near-base, low-confidence
                predictions (eval mAP looks fine, but predict() at the default
                threshold returns ~0 boxes). Disabling EMA saves the actual trained
                weights → confident, usable checkpoints. Pass ``ema=True`` for long
                training runs where EMA has time to converge.
            **extra: family-specific training kwargs forwarded verbatim.

        Returns a normalized result dict (see ``_normalize_train_result``).

        Legal: only the permissive families (YOLOX / YOLOv9 / RT-DETR / D-FINE)
        train. YOLO-NAS (Deci, non-commercial) is rejected with
        :class:`TrainingNotSupportedError`. No Ultralytics runtime is imported.
        """
        from visionservex.data.yolo_dataset import resolve_dataset_yaml, validate_yolo_yaml

        parsed = _parse_model_id(self.entry.id)
        if parsed is None:
            raise MissingDependencyError(
                f"{self.entry.id!r} is not a libreyolo-<family>-<size> id",
                install_hint="check `visionservex list-models` for trainable libreyolo ids",
            )
        family, size = parsed
        if family not in _TRAINABLE_FAMILIES:
            raise TrainingNotSupportedError(
                f"TRAINING_NOT_SUPPORTED: libreyolo family {family!r} is not trainable. "
                f"Only permissive families {sorted(_TRAINABLE_FAMILIES)} are allowed "
                f"(YOLO-NAS is Deci proprietary / non-commercial and is excluded)."
            )

        # Resolve + validate the YOLO dataset (safe_load only; a download: block
        # is never executed here, and allow_download_scripts stays False below).
        data_yaml = resolve_dataset_yaml(dataset)
        verdict = validate_yolo_yaml(data_yaml)
        if verdict.get("status") != "ok":
            raise TrainingNotSupportedError(
                f"DATASET_INVALID: {data_yaml} failed YOLO validation: {verdict.get('issues')}"
            )

        # Ensure the model is instantiated (base/pretrained weights as init).
        if not self._real_ready:
            self.load(device=device or "auto", precision="fp32")

        train_device = "" if device in (None, "auto", "") else str(device)
        kwargs: dict[str, Any] = {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": train_device,
            "workers": workers,
            "seed": seed,
            "project": project or "runs/train",
            "name": name or self.entry.id,
            "exist_ok": exist_ok,
            "resume": resume,
            "patience": patience,
            "ema": ema,
            "allow_download_scripts": False,
        }
        if lr0 is not None:
            kwargs["lr0"] = lr0
        if amp is not None:
            kwargs["amp"] = amp
        kwargs.update(extra)

        _log.info(
            "training %s (%s-%s) on %s for %d epochs",
            self.entry.id,
            family,
            size,
            data_yaml,
            epochs,
        )
        t0 = time.time()
        raw = self._model.train(data=str(data_yaml), **kwargs)
        elapsed_h = (time.time() - t0) / 3600.0
        return _normalize_train_result(
            raw,
            model_id=self.entry.id,
            variant=f"{family}-{size}",
            data_yaml=data_yaml,
            hours=elapsed_h,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        precision: str = "fp32",
    ) -> LibreYOLOEngine:
        """Load a trained LibreYOLO checkpoint (``best.pt``/``last.pt``) for inference.

        The family and input size come from this engine's model id — never
        guessed from the file. Raises :class:`MissingDependencyError` if the
        file is absent. There is **no** silent fall back to base weights: once a
        checkpoint is supplied it is the only weight source.
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise MissingDependencyError(
                f"trained checkpoint not found: {ckpt}",
                install_hint="train one first: VisionModel('libreyolo-yolox-s').train(dataset)",
            )
        if self._real_ready or self._model is not None:
            self.unload()
        self._real_ready = False
        self._checkpoint_override = ckpt
        self.load(device=device or self.device or "cpu", precision=precision)
        return self

    def export(self, format: str, output_path: str | Path) -> Path:
        """Export the loaded LibreYOLO model to a deployment format (ONNX, ...).

        Delegates to the libreyolo package exporter. Loads base/pretrained
        weights on demand if the model is not already in memory.
        """
        if not self._real_ready:
            self.load(device=self.device or "cpu", precision="fp32")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result_path = self._model.export(format=format, output_path=str(out))
        return Path(result_path)

    # ------ inference ------

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        threshold: float = 0.25,
        nms_iou: float | None = None,
        max_det: int = 300,
        return_raw: bool = False,
        **kwargs: Any,
    ) -> BaseResult:
        """Run detection and return FINAL (post-NMS) detections by default.

        DETR-style decoders (RT-DETR / D-FINE) are set-based and emit up to ~300
        boxes with no NMS — an undertrained/low-threshold model can flood
        overlapping duplicates. We apply a class-aware NMS safety net on top of
        libreyolo's output (idempotent for already-NMS'd YOLO output). Pass
        ``return_raw=True`` to bypass NMS and get the raw proposals for debugging;
        the result's ``metadata`` always carries ``raw_count``/``post_nms_count``.
        """
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        conf = float(kwargs.get("conf", threshold))
        # libreyolo's ImageLoader accepts a PIL Image directly and normalizes to
        # RGB, so we forward the decoded image without a temp file.
        result = self._model(image, conf=conf, save=False)
        return self._to_result(
            result, image, nms_iou=nms_iou, max_det=max_det, return_raw=return_raw
        )

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        """Unused: predict() is overridden to drive the package directly."""
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        """Unused: predict() is overridden to drive the package directly."""
        return self._mock.postprocess(raw, image=image, **kwargs)

    def _to_result(
        self,
        result: Any,
        image: Image.Image,
        *,
        nms_iou: float | None = None,
        max_det: int = 300,
        return_raw: bool = False,
    ) -> BaseResult:
        """Convert a LibreYOLO ``Results`` object → VisionServeX DetectionResult.

        Applies a class-aware NMS safety net unless ``return_raw`` is set, and
        records ``raw_count``/``post_nms_count`` in the result metadata.
        """
        w, h = image.size
        raw_dets = _libre_to_detections(result, self._model)
        raw_count = len(raw_dets)
        if return_raw:
            detections = raw_dets
        else:
            from visionservex.runtime.postprocess import DEFAULT_NMS_IOU, class_aware_nms

            iou = DEFAULT_NMS_IOU if nms_iou is None else float(nms_iou)
            detections = class_aware_nms(raw_dets, iou_thres=iou, max_det=max_det)
        return DetectionResult(
            kind="detection",
            model_id=self.entry.id,
            task=self.entry.task,
            image_size=(w, h),
            device=self.device,
            precision=self.precision,
            backend=self.backend_label,
            detections=detections,
            metadata={
                "raw_count": raw_count,
                "post_nms_count": len(detections),
                "nms_applied": not return_raw,
                "checkpoint": str(self._checkpoint_override) if self._checkpoint_override else None,
            },
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


def _normalize_train_result(
    raw: Any, *, model_id: str, variant: str, data_yaml: Path, hours: float
) -> dict[str, Any]:
    """Normalize a libreyolo trainer dict into the VisionServeX train contract.

    The underlying ``libreyolo`` ``model.train()`` returns a dict with keys like
    ``save_dir``, ``best_checkpoint``, ``last_checkpoint``, ``best_mAP50``,
    ``best_mAP50_95``, ``best_epoch``, ``final_loss`` and ``epoch_losses``. We
    map those into a stable contract and attach the measured wall-clock time.
    Optional artifact paths (``results.csv``, ``train_config.yaml``) are included
    only when they actually exist on disk.
    """
    raw = raw if isinstance(raw, dict) else {}
    save_dir = str(raw.get("save_dir") or raw.get("output_dir") or "")
    weights_dir = str(Path(save_dir) / "weights") if save_dir else ""
    best_ckpt = raw.get("best_checkpoint") or (
        str(Path(weights_dir) / "best.pt") if weights_dir else ""
    )
    last_ckpt = raw.get("last_checkpoint") or (
        str(Path(weights_dir) / "last.pt") if weights_dir else ""
    )
    epoch_losses = raw.get("epoch_losses") or []
    epochs_completed = len(epoch_losses) if epoch_losses else raw.get("epochs_completed")

    def _existing(path_str: str) -> str | None:
        return path_str if path_str and Path(path_str).exists() else None

    # best.pt is only written when val mAP improves; on tiny/short runs it may
    # never be written (only last.pt). Fall back so best_checkpoint always points
    # to a file that actually exists — otherwise Anastig's reload of
    # best_checkpoint fails on a non-existent path. `checkpoint` is the one to use.
    best_exists = bool(best_ckpt and Path(best_ckpt).is_file())
    last_exists = bool(last_ckpt and Path(last_ckpt).is_file())
    if not best_exists and last_exists:
        best_ckpt = last_ckpt
        best_exists = True
    usable_ckpt = best_ckpt if best_exists else (last_ckpt if last_exists else "")

    args_yaml = str(Path(save_dir) / "train_config.yaml") if save_dir else ""
    results_csv = str(Path(save_dir) / "results.csv") if save_dir else ""
    return {
        "status": "ok",
        "model_id": model_id,
        "family": "libreyolo",
        "variant": variant,
        "dataset_format": "yolo",
        "dataset_yaml": str(data_yaml),
        "best_checkpoint": best_ckpt,
        "last_checkpoint": last_ckpt,
        "checkpoint": usable_ckpt,
        "save_dir": save_dir,
        "metrics": {
            "best_mAP50": raw.get("best_mAP50"),
            "best_mAP50_95": raw.get("best_mAP50_95"),
            "best_epoch": raw.get("best_epoch"),
            "epochs_completed": epochs_completed,
            "final_loss": raw.get("final_loss"),
            "training_time_hours": round(hours, 4),
        },
        "artifacts": {
            "weights_dir": weights_dir or None,
            "results_csv": _existing(results_csv),
            "args_yaml": _existing(args_yaml),
        },
    }


def _factory(entry: ModelEntry) -> LibreYOLOEngine:
    return LibreYOLOEngine(entry)


register_engine("libreyolo", _factory)

__all__ = ["LibreYOLOEngine", "TrainingNotSupportedError"]

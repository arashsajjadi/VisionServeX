# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""MedSAM2 in-process runtime adapter (experimental, research-only).

This wraps the upstream MedSAM2 / SAM2 image predictor so VisionServeX can run a
REAL 2D (slice/frame) segmentation when — and only when — the optional `sam2`
stack and a user-provided MedSAM2 checkpoint are present. It is import-light:
``torch`` / ``sam2`` are imported lazily inside calls, never at module load.

Truthfulness contract:
* MedSAM2 weights are RESEARCH/EDUCATION ONLY (non-commercial) — never
  commercial-safe (HF wanglab/MedSAM2 model card: "The model weights can only be
  used for research and education purposes").
* No mock masks here. If the real stack/checkpoint is missing, we raise a
  structured :class:`MedSAM2RuntimeError`; we never fabricate output.
* Only 2D slice/frame inference is wired (``medsam2_slice_inference_experimental``).
  3D-volume / video are NOT implemented in-process and must be rejected, not faked.

Upstream facts (github.com/bowang-lab/MedSAM2): python 3.12, torch==2.5.1, the
fork provides the ``sam2`` package; checkpoints (e.g. ``MedSAM2_latest.pt``) on
HF wanglab/MedSAM2; 2D path = ``build_sam2`` + ``SAM2ImagePredictor``.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from visionservex.engines.medsam2_sidecar import (
    MEDSAM2_INSTALL_HINT,
    MEDSAM2_NONCOMMERCIAL_NOTE,
)

# Structured error codes (single source of truth; mirrored in tests + CLI).
MEDSAM2_REQUIRED = "MEDSAM2_REQUIRED"
MEDSAM2_CHECKPOINT_REQUIRED = "MEDSAM2_CHECKPOINT_REQUIRED"
MEDSAM2_CHECKPOINT_INVALID = "MEDSAM2_CHECKPOINT_INVALID"
MEDSAM2_CONFIG_REQUIRED = "MEDSAM2_CONFIG_REQUIRED"
MEDSAM2_RUNTIME_UNAVAILABLE = "MEDSAM2_RUNTIME_UNAVAILABLE"
MEDSAM2_LICENSE_RESTRICTED = "MEDSAM2_LICENSE_RESTRICTED"
MEDSAM2_OOM = "MEDSAM2_OOM"
MEDSAM2_UNSUPPORTED_INPUT = "MEDSAM2_UNSUPPORTED_INPUT"

#: Required upstream module (the MedSAM2 fork / SAM2 install both expose `sam2`).
_REQUIRED_MODULES = ("sam2", "torch")

#: Candidate Hydra configs tried in order (MedSAM2 trains at 512 res).
_CANDIDATE_CONFIGS = (
    "configs/sam2.1_hiera_t512.yaml",
    "configs/sam2.1/sam2.1_hiera_t.yaml",
    "configs/sam2.1/sam2.1_hiera_t512.yaml",
)

_NOT_FOR_DIAGNOSIS = "Research/education only — NOT for diagnosis or clinical use."


class MedSAM2RuntimeError(RuntimeError):
    """Structured MedSAM2 runtime error carrying a machine-readable code."""

    def __init__(self, code: str, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "code": self.code,
            "message": str(self),
            "hint": self.hint or MEDSAM2_INSTALL_HINT,
            "commercial_safe": False,
            "license_note": MEDSAM2_NONCOMMERCIAL_NOTE,
        }


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def medsam2_doctor() -> dict[str, Any]:
    """Side-effect-free availability report (no heavy imports, no download)."""
    missing = [m for m in _REQUIRED_MODULES if not _module_present(m)]
    torch_version = None
    cuda_available = False
    if not missing:
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            missing.append("torch")
    return {
        "model_id": "medsam2",
        "runtime_type": "in_process_sam2",
        "runnable_runtime_present": not missing,
        "missing_modules": missing,
        "structured_error_code": MEDSAM2_REQUIRED if missing else "",
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "supported_input_modes": ["2d_slice"],
        "unsupported_input_modes": ["3d_volume", "video"],
        "commercial_safe": False,
        "license_note": MEDSAM2_NONCOMMERCIAL_NOTE,
        "disclaimer": _NOT_FOR_DIAGNOSIS,
        "install_hint": MEDSAM2_INSTALL_HINT,
    }


def _require_runtime() -> None:
    missing = [m for m in _REQUIRED_MODULES if not _module_present(m)]
    if missing:
        raise MedSAM2RuntimeError(
            MEDSAM2_REQUIRED,
            f"MedSAM2 runtime unavailable; missing modules: {missing}. "
            + MEDSAM2_NONCOMMERCIAL_NOTE,
        )


@dataclass
class MedSAM2Runtime:
    """A loaded MedSAM2 model + predictor (2D image path)."""

    checkpoint_path: str
    config_path: str
    device: str
    load_time_seconds: float
    _model: Any = field(default=None, repr=False)
    _predictor: Any = field(default=None, repr=False)

    def info(self) -> dict[str, Any]:
        return {
            "status": "loaded",
            "model_id": "medsam2",
            "runtime_type": "in_process_sam2",
            "device": self.device,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_detected": True,
            "checkpoint_validated": True,
            "load_time_seconds": self.load_time_seconds,
            "commercial_safe": False,
            "license_note": MEDSAM2_NONCOMMERCIAL_NOTE,
            "disclaimer": _NOT_FOR_DIAGNOSIS,
        }


def load_medsam2_runtime(
    checkpoint: str | Path,
    *,
    config: str | None = None,
    device: str = "cpu",
) -> MedSAM2Runtime:
    """Load a real MedSAM2 model. Raises a structured error on any failure."""
    _require_runtime()
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise MedSAM2RuntimeError(
            MEDSAM2_CHECKPOINT_REQUIRED,
            f"MedSAM2 checkpoint not found: {ckpt}. Download a research-only "
            "checkpoint from HF wanglab/MedSAM2 (e.g. MedSAM2_latest.pt).",
        )

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise MedSAM2RuntimeError(MEDSAM2_REQUIRED, f"sam2 not importable: {exc}") from exc

    configs = [config, *(_CANDIDATE_CONFIGS)] if config else list(_CANDIDATE_CONFIGS)
    t0 = time.perf_counter()
    model = None
    last_err = ""
    used_config = ""
    for cfg in configs:
        if cfg is None:
            continue
        try:
            model = build_sam2(cfg, str(ckpt), device=device)
            used_config = cfg
            break
        except Exception as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "oom" in msg:
                raise MedSAM2RuntimeError(
                    MEDSAM2_OOM, f"out of memory loading MedSAM2: {exc}"
                ) from exc
            # A config-resolution miss → try the next candidate; otherwise it is a
            # genuine checkpoint/state-dict incompatibility and we stop.
            if "config" in msg or "not found" in msg or "does not exist" in msg:
                last_err = f"{cfg}: {exc}"
                continue
            raise MedSAM2RuntimeError(
                MEDSAM2_CHECKPOINT_INVALID,
                f"MedSAM2 checkpoint failed to load into the model: {exc}",
            ) from exc
    if model is None:
        raise MedSAM2RuntimeError(
            MEDSAM2_CONFIG_REQUIRED,
            f"could not resolve a MedSAM2/SAM2 config. Tried {configs}. last: {last_err}",
        )
    try:
        model.eval()
        predictor = SAM2ImagePredictor(model)
    except Exception as exc:
        raise MedSAM2RuntimeError(MEDSAM2_RUNTIME_UNAVAILABLE, str(exc)) from exc

    rt = MedSAM2Runtime(
        checkpoint_path=str(ckpt),
        config_path=used_config,
        device=device,
        load_time_seconds=round(time.perf_counter() - t0, 3),
    )
    rt._model = model
    rt._predictor = predictor
    return rt


def load_2d_input(path: str | Path):
    """Load a 2D RGB array from PNG/JPEG, or one slice of a NIfTI volume.

    Rejects DICOM and unreadable inputs with structured errors. Returns a uint8
    HxWx3 numpy array. (3D/video are intentionally unsupported here.)
    """
    import numpy as np

    p = Path(path)
    if not p.exists():
        raise MedSAM2RuntimeError(
            MEDSAM2_UNSUPPORTED_INPUT, f"input not found: {p}", hint="check the path"
        )
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        from PIL import Image

        return np.array(Image.open(p).convert("RGB")).astype(np.uint8)
    if suffix.endswith(".dcm") or suffix.endswith(".dicom"):
        raise MedSAM2RuntimeError(
            MEDSAM2_UNSUPPORTED_INPUT,
            "DICOM input is not supported by this runtime yet. Convert to PNG/NIfTI "
            "or use a dedicated DICOM pipeline.",
        )
    if suffix.endswith((".nii", ".nii.gz")):
        if not _module_present("nibabel"):
            raise MedSAM2RuntimeError(
                MEDSAM2_UNSUPPORTED_INPUT,
                "NIfTI input requires nibabel (pip install 'visionservex[medical]').",
            )
        import nibabel as nib

        vol = nib.load(str(p)).get_fdata()
        if vol.ndim != 3:
            raise MedSAM2RuntimeError(
                MEDSAM2_UNSUPPORTED_INPUT, f"expected 3D NIfTI, got shape {vol.shape}"
            )
        sl = vol[:, :, vol.shape[2] // 2]  # middle slice — 2D path only
        sl = sl - sl.min()
        denom = sl.max() or 1.0
        sl = (sl / denom * 255.0).astype(np.uint8)
        return np.stack([sl, sl, sl], axis=-1)
    raise MedSAM2RuntimeError(
        MEDSAM2_UNSUPPORTED_INPUT,
        f"unsupported input type {suffix!r}. Supported: PNG/JPEG (2D) and NIfTI (middle slice).",
    )


def segment_2d(
    runtime: MedSAM2Runtime,
    image,
    *,
    boxes: list[list[float]] | None = None,
    points: list[list[float]] | None = None,
    point_labels: list[int] | None = None,
    slice_index: int | None = None,
):
    """Run real 2D MedSAM2 inference; return a VisionServeX ``SegmentationResult``.

    One segment per box prompt (or one for a points prompt). Never returns mock
    output: failures raise a structured :class:`MedSAM2RuntimeError`.
    """
    import numpy as np
    import torch

    from visionservex.core.results import Box, Segment, SegmentationResult

    img = np.asarray(image)
    if img.ndim != 3 or img.shape[2] != 3:
        raise MedSAM2RuntimeError(
            MEDSAM2_UNSUPPORTED_INPUT, f"expected HxWx3 image, got shape {img.shape}"
        )
    h, w = img.shape[:2]
    predictor = runtime._predictor

    np_boxes = np.array(boxes, dtype=float) if boxes else None
    np_points = np.array(points, dtype=float) if points else None
    np_labels = (
        np.array(point_labels, dtype=int)
        if point_labels is not None
        else (np.ones(len(points), dtype=int) if points else None)
    )

    segments: list[Segment] = []
    try:
        with torch.inference_mode():
            predictor.set_image(img)
            if np_boxes is not None:
                masks, scores, _ = predictor.predict(box=np_boxes, multimask_output=False)
            else:
                pc = np_points if np_points is not None else np.array([[w / 2, h / 2]])
                pl = np_labels if np_labels is not None else np.array([1])
                masks, scores, _ = predictor.predict(
                    point_coords=pc, point_labels=pl, multimask_output=False
                )
    except Exception as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            raise MedSAM2RuntimeError(MEDSAM2_OOM, f"MedSAM2 inference OOM: {exc}") from exc
        raise MedSAM2RuntimeError(MEDSAM2_RUNTIME_UNAVAILABLE, f"inference failed: {exc}") from exc

    masks = np.asarray(masks)
    if masks.ndim == 2:  # (H, W) -> (1, H, W)
        masks = masks[None, ...]
    scores_arr = np.asarray(scores).ravel()
    for i in range(masks.shape[0]):
        m = (masks[i] > 0).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs):
            box = Box(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        elif np_boxes is not None and i < len(np_boxes):
            b = np_boxes[i]
            box = Box(float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        else:
            box = Box(0, 0, float(w), float(h))
        score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
        segments.append(Segment(box=box, score=score, label="lesion", mask=m))

    result = SegmentationResult(
        kind="segmentation",
        model_id="medsam2",
        task="foundation_segment",
        image_size=(w, h),
        device=runtime.device,
        precision="fp32",
        backend="medsam2_runtime",
        segments=segments,
    )
    result.metadata["engine"] = "medsam2_runtime"
    result.metadata["input_mode"] = "2d_slice"
    result.metadata["prompt_type"] = "box" if boxes else "point"
    result.metadata["checkpoint_path"] = runtime.checkpoint_path
    result.metadata["commercial_safe"] = False
    result.metadata["research_only"] = True
    result.metadata["not_for_diagnosis"] = _NOT_FOR_DIAGNOSIS
    if slice_index is not None:
        result.metadata["slice_index"] = slice_index
    result.warnings.append(MEDSAM2_NONCOMMERCIAL_NOTE)
    return result


__all__ = [
    "MEDSAM2_CHECKPOINT_INVALID",
    "MEDSAM2_CHECKPOINT_REQUIRED",
    "MEDSAM2_CONFIG_REQUIRED",
    "MEDSAM2_LICENSE_RESTRICTED",
    "MEDSAM2_OOM",
    "MEDSAM2_REQUIRED",
    "MEDSAM2_RUNTIME_UNAVAILABLE",
    "MEDSAM2_UNSUPPORTED_INPUT",
    "MedSAM2Runtime",
    "MedSAM2RuntimeError",
    "load_2d_input",
    "load_medsam2_runtime",
    "medsam2_doctor",
    "segment_2d",
]

# SPDX-License-Identifier: Apache-2.0
"""v2.55.0: SAM optional-extra engines for efficientsam, hq-sam, mobilesam.

Packages (all Apache-2.0):
  efficientsam          - EfficientViT-SAM (pip install efficientsam)
  mobile-sam            - MobileSAM        (pip install mobile-sam)
  segment-anything-hq   - HQ-SAM          (pip install segment-anything-hq)

Checkpoints auto-downloaded to ~/.cache/visionservex/{mobilesam,hqsam,efficientsam}/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from visionservex.core.results import BaseResult, Box, Segment, SegmentationResult
from visionservex.engines.base import BaseEngine, MissingDependencyError
from visionservex.engines.registry import register_engine

MOBILESAM_CKPT = Path.home() / ".cache" / "visionservex" / "mobilesam" / "mobile_sam.pt"
HQSAM_CKPT = Path.home() / ".cache" / "visionservex" / "hqsam" / "sam_hq_vit_b.pth"
EFFICIENTSAM_CKPT = (
    Path.home() / ".cache" / "visionservex" / "efficientsam" / "efficientvit_sam_l0.pt"
)


def _to_rgb(img: Any) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3].copy()


def _make_result(
    model_id: str,
    masks: np.ndarray,
    scores: np.ndarray,
    img_np: np.ndarray,
    backend: str,
    device: str,
) -> SegmentationResult:
    h, w = img_np.shape[:2]
    segments = []
    for m, s in zip(masks, scores, strict=False):
        mask_u8 = (m > 0).astype(np.uint8) * 255
        ys, xs = np.where(m > 0)
        if xs.size:
            box = Box(
                x1=float(xs.min()), y1=float(ys.min()), x2=float(xs.max()), y2=float(ys.max())
            )
        else:
            box = Box(x1=0.0, y1=0.0, x2=float(w), y2=float(h))
        segments.append(Segment(box=box, score=float(s), label="segment", mask=mask_u8))
    return SegmentationResult(
        model_id=model_id,
        task="foundation_segment",
        image_size=(w, h),
        device=device,
        backend=backend,
        segments=segments,
    )


class MobileSAMEngine(BaseEngine):
    """MobileSAM (mobile-sam, Apache-2.0)."""

    def load(self, *, device: str = "cuda", precision: str = "fp32") -> None:
        try:
            from mobile_sam import SamPredictor, build_sam_vit_t
        except ImportError:
            raise MissingDependencyError("pip install mobile-sam") from None
        if not MOBILESAM_CKPT.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint missing: {MOBILESAM_CKPT}. "
                "Download from: https://github.com/ChaoningZhang/MobileSAM/releases"
            )
        model = build_sam_vit_t(checkpoint=str(MOBILESAM_CKPT))
        model.eval()
        try:
            model.to(device)
            self._device = device
        except Exception:
            model.to("cpu")
            self._device = "cpu"
        self._predictor = SamPredictor(model)
        self._loaded = True

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed  # done in preprocess + postprocess

    def preprocess(
        self, image: Any, *, points: Any = None, boxes: Any = None, **kwargs: Any
    ) -> Any:
        img_np = _to_rgb(image)
        self._predictor.set_image(img_np)
        return {"img_np": img_np, "points": points, "boxes": boxes}

    def postprocess(self, raw: Any, *, image: Any = None, **kwargs: Any) -> BaseResult:
        img_np = raw["img_np"]
        boxes = raw.get("boxes")
        points = raw.get("points")
        if boxes is not None:
            b = np.array(boxes, dtype=np.float32).flatten()
            masks, scores, _ = self._predictor.predict(box=b, multimask_output=False)
        else:
            pt = points if points is not None else [img_np.shape[1] // 2, img_np.shape[0] // 2]
            if isinstance(pt, (list, tuple)) and len(pt) == 2 and not hasattr(pt[0], "__len__"):
                pt = [pt]
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(pt, dtype=np.float32),
                point_labels=np.ones(len(pt), dtype=np.int32),
                multimask_output=False,
            )
        return _make_result(self.entry.id, masks, scores, img_np, "mobilesam", self._device)

    def unload(self) -> None:
        self._predictor = None
        self._loaded = False


class HQSAMEngine(BaseEngine):
    """HQ-SAM (segment-anything-hq, Apache-2.0)."""

    def load(self, *, device: str = "cuda", precision: str = "fp32") -> None:
        try:
            import torch
            from segment_anything_hq import SamPredictor, build_sam_vit_b
        except ImportError:
            raise MissingDependencyError("pip install segment-anything-hq") from None
        if not HQSAM_CKPT.exists():
            raise FileNotFoundError(f"HQ-SAM checkpoint missing: {HQSAM_CKPT}")
        # Build the architecture first, then load the state dict with an explicit
        # CPU map_location. The upstream ``build_sam_vit_b(checkpoint=...)`` path
        # calls a bare ``torch.load`` that fails on CPU-only hosts when the
        # checkpoint carries CUDA tensors ("Attempting to deserialize ... CUDA ...").
        model = build_sam_vit_b(checkpoint=None)
        state = torch.load(str(HQSAM_CKPT), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        try:
            model.to(device)
            self._device = device
        except Exception:
            model.to("cpu")
            self._device = "cpu"
        self._predictor = SamPredictor(model)
        self._loaded = True

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def preprocess(
        self, image: Any, *, points: Any = None, boxes: Any = None, **kwargs: Any
    ) -> Any:
        img_np = _to_rgb(image)
        self._predictor.set_image(img_np)
        return {"img_np": img_np, "points": points, "boxes": boxes}

    def postprocess(self, raw: Any, *, image: Any = None, **kwargs: Any) -> BaseResult:
        img_np = raw["img_np"]
        boxes = raw.get("boxes")
        points = raw.get("points")
        if boxes is not None:
            b = np.array(boxes, dtype=np.float32).flatten()
            masks, scores, _ = self._predictor.predict(box=b, multimask_output=False)
        else:
            pt = points if points is not None else [img_np.shape[1] // 2, img_np.shape[0] // 2]
            if isinstance(pt, (list, tuple)) and len(pt) == 2 and not hasattr(pt[0], "__len__"):
                pt = [pt]
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(pt, dtype=np.float32),
                point_labels=np.ones(len(pt), dtype=np.int32),
                multimask_output=False,
            )
        return _make_result(self.entry.id, masks, scores, img_np, "hq-sam", self._device)

    def unload(self) -> None:
        self._predictor = None
        self._loaded = False


class EfficientSAMEngine(BaseEngine):
    """EfficientViT-SAM L0 (efficientsam package, Apache-2.0)."""

    def load(self, *, device: str = "cuda", precision: str = "fp32") -> None:
        try:
            import efficientsam.sam_model_zoo as zoo
            from efficientsam.cached_sam_model import EfficientViTSamPredictor
        except ImportError:
            raise MissingDependencyError("pip install efficientsam") from None
        if not EFFICIENTSAM_CKPT.exists():
            raise FileNotFoundError(f"EfficientViT-SAM checkpoint missing: {EFFICIENTSAM_CKPT}")
        model = zoo.create_efficientvit_sam_model(
            "efficientvit-sam-l0", pretrained=True, weight_url=str(EFFICIENTSAM_CKPT)
        )
        model.eval()
        try:
            model.to(device)
            self._device = device
        except Exception:
            model.to("cpu")
            self._device = "cpu"
        self._predictor = EfficientViTSamPredictor(model)
        self._loaded = True

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def preprocess(
        self, image: Any, *, points: Any = None, boxes: Any = None, **kwargs: Any
    ) -> Any:
        img_np = _to_rgb(image)
        self._predictor.set_image(img_np)
        return {"img_np": img_np, "points": points, "boxes": boxes}

    def postprocess(self, raw: Any, *, image: Any = None, **kwargs: Any) -> BaseResult:
        img_np = raw["img_np"]
        boxes = raw.get("boxes")
        points = raw.get("points")
        if boxes is not None:
            b = np.array(boxes, dtype=np.float32).flatten()
            masks, scores, _ = self._predictor.predict(box=b, multimask_output=False)
        else:
            pt = points if points is not None else [img_np.shape[1] // 2, img_np.shape[0] // 2]
            if isinstance(pt, (list, tuple)) and len(pt) == 2 and not hasattr(pt[0], "__len__"):
                pt = [pt]
            masks, scores, _ = self._predictor.predict(
                point_coords=np.array(pt, dtype=np.float32),
                point_labels=np.ones(len(pt), dtype=np.int32),
                multimask_output=False,
            )
        return _make_result(self.entry.id, masks, scores, img_np, "efficientsam", self._device)

    def unload(self) -> None:
        self._predictor = None
        self._loaded = False


register_engine("mobilesam", lambda entry: MobileSAMEngine(entry))
register_engine("efficientsam", lambda entry: EfficientSAMEngine(entry))
register_engine("hq-sam", lambda entry: HQSAMEngine(entry))

__all__ = ["EfficientSAMEngine", "HQSAMEngine", "MobileSAMEngine"]

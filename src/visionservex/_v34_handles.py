"""VisionServeX v3.4 handle classes — SAMHandle, DINOHandle, PipelineHandle.

These are returned by the VSX instance methods ``vsx.sam()``, ``vsx.dino()``,
and ``vsx.pipeline()``.  They mirror the honesty-first contract already
established in vsx._SAMHandle / _DINOHandle / _PipelineHandle, but as public,
importable classes with stable v3.4 API signatures.

No fake inference ever — every method either calls a real engine or raises a
structured ValueError with code=GATED_HF_AUTH_REQUIRED for gated models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ALIAS: dict[str, str] = {
    "sam-vit-b": "sam-vit-base",
    "sam-vit-l": "sam-vit-large",
    "sam-vit-h": "sam-vit-huge",
}

_SAM_RUNNABLE = (
    "sam-vit-base sam-vit-large sam-vit-huge "
    "sam2-hiera-tiny sam2-hiera-small sam2-hiera-base-plus sam2-hiera-large "
    "sam2.1-hiera-tiny sam2.1-hiera-small sam2.1-hiera-base-plus sam2.1-hiera-large "
    "mobilesam efficientsam medsam"
)
_SAM_AUTH = "sam3-base"
_SAM_LEGAL = "hq-sam hq-sam2 light-hq-sam tinysam q-tinysam"
_SAM_SIDECAR = "medsam2"

_DINO_EMBED = "dinov2-small dinov2-base dinov2-large dinov2-giant"
_DINO_DETECT = (
    "grounding-dino-swin-t grounding-dino-swin-b grounding-dino-tiny "
    "grounding-dino-original-swin-t grounding-dino-original-swin-b"
)
_DINO_AUTH = "grounding-dino-1.5 grounding-dino-1.6"
_DINO_EXT_API = "grounding-dino-1.5-pro grounding-dino-1.6-pro dino-x-api"
_DINO_LEGAL = "dinov3-vits16 dinov3-vitb16 dinov3-vitl16 dinov3-vit7b16"


def _in(mid: str, csv: str) -> bool:
    return mid in csv.split()


def _load_image(path):
    from PIL import Image as _Image

    if isinstance(path, str):
        return _Image.open(path).convert("RGB")
    return path


class _GatedError(ValueError):
    """Raised for HF-gated models that require auth token."""

    code = "GATED_HF_AUTH_REQUIRED"

    def __init__(self, model_id: str, fix: str):
        super().__init__(
            f"[GATED_HF_AUTH_REQUIRED] {model_id} is a gated Hugging Face model. Fix: {fix}"
        )
        self.model_id = model_id
        self.fix = fix


# ---------------------------------------------------------------------------
# SAMHandle
# ---------------------------------------------------------------------------


class SAMHandle:
    """Handle for a SAM-family segmentation model.

    Parameters
    ----------
    model_id:
        One of the runnable SAM variants, e.g. ``'sam-vit-b'``, ``'mobilesam'``,
        ``'sam2.1-hiera-tiny'``.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = _ALIAS.get(model_id, model_id)

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _state(self) -> str:
        m = self.model_id
        if _in(m, _SAM_AUTH):
            return "auth_required"
        if _in(m, _SAM_LEGAL) or _in(m, _SAM_SIDECAR):
            return "legal_review_required"
        if _in(m, _SAM_RUNNABLE):
            return "benchmark_passed"
        return "legal_review_required"

    def _assert_runnable(self) -> None:
        state = self._state()
        if state == "auth_required":
            raise _GatedError(
                self.model_id,
                f"export HF_TOKEN=<your_token> && request gated access on "
                f"https://huggingface.co/{self.model_id}",
            )
        if state != "benchmark_passed":
            raise ValueError(
                f"{self.model_id} state={state!r} — cannot run inference directly. "
                f"Run `visionservex sam status {self.model_id}` for next steps."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, image, boxes=None, points=None) -> dict:
        """Segment *image* using this SAM model.

        Parameters
        ----------
        image:
            File path (str) or PIL Image.
        boxes:
            Optional bounding-box prompt(s) in ``[x1, y1, x2, y2]`` format.
        points:
            Optional point-prompt list of ``(x, y)`` tuples.

        Returns
        -------
        dict with keys ``segments`` (list of mask dicts), ``model_id``.
        """
        self._assert_runnable()
        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            raw = model.predict(_load_image(image), box=boxes, points=points)
        return {"model_id": self.model_id, "segments": raw}

    def track(self, frame_dir: str, out_dir: str) -> dict:
        """Run SAM2 video-object tracking over a directory of frames.

        Returns a dict with ``module_missing`` key if SAM2 is not installed.
        Only SAM2 / SAM2.1 variants support this.
        """
        if not (self.model_id.startswith("sam2") or "video" in self.model_id):
            return {
                "model_id": self.model_id,
                "error": "video tracking requires a SAM2/SAM2.1 model",
                "hint": "Use e.g. vsx.sam('sam2.1-hiera-small').track(frame_dir, out_dir)",
            }
        self._assert_runnable()
        try:
            from visionservex.sam2_runtime import track_video_dir

            return track_video_dir(self.model_id, frame_dir, out_dir)
        except ImportError as exc:
            return {
                "model_id": self.model_id,
                "module_missing": str(exc),
                "hint": "pip install visionservex[sam2] to enable SAM2 video tracking",
            }

    def export_onnx(self, out_path: str) -> dict:
        """Export the SAM mask-decoder to ONNX.

        Delegates to ``visionservex.onnx_export.export_sam_decoder_onnx``.
        """
        self._assert_runnable()
        from visionservex.onnx_export import export_sam_decoder_onnx, onnx_eligible

        if self.model_id not in onnx_eligible():
            raise ValueError(
                f"{self.model_id} is not ONNX-export-eligible. "
                f"Eligible: {list(onnx_eligible().keys())}"
            )
        return export_sam_decoder_onnx(self.model_id, out_path)

    def status(self) -> dict:
        """Return auth/checkpoint/onnx status for this model."""
        import os
        from pathlib import Path

        from visionservex.onnx_export import onnx_eligible

        state = self._state()
        ckpt_map = {
            "sam-vit-base": "~/.cache/visionservex/sam/sam_vit_b_01ec64.pth",
            "sam-vit-large": "~/.cache/visionservex/sam/sam_vit_l_0b3195.pth",
            "sam-vit-huge": "~/.cache/visionservex/sam/sam_vit_h_4b8939.pth",
            "mobilesam": "~/.cache/visionservex/mobilesam/mobile_sam.pt",
        }
        ckpt = ckpt_map.get(self.model_id)
        ckpt_cached = bool(ckpt and Path(ckpt).expanduser().exists())
        hf_token_set = bool(os.environ.get("HF_TOKEN"))
        onnx_ok = self.model_id in onnx_eligible()
        return {
            "model_id": self.model_id,
            "state": state,
            "auth_required": state == "auth_required",
            "hf_token_set": hf_token_set,
            "checkpoint_cached": ckpt_cached,
            "onnx_eligible": onnx_ok,
        }


# ---------------------------------------------------------------------------
# DINOHandle
# ---------------------------------------------------------------------------


class DINOHandle:
    """Handle for a DINOv2 / GroundingDINO model.

    Parameters
    ----------
    model_id:
        E.g. ``'dinov2-base'`` (embedding) or ``'grounding-dino-swin-t'`` (detection).
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def _task(self) -> str:
        m = self.model_id
        if _in(m, _DINO_DETECT):
            return "open_vocab_detect"
        if _in(m, _DINO_EMBED):
            return "embed"
        if _in(m, _DINO_EXT_API) or _in(m, _DINO_AUTH):
            return "open_vocab_detect"
        return "embed"

    def _state(self) -> str:
        m = self.model_id
        if _in(m, _DINO_EXT_API):
            return "external_api_only"
        if _in(m, _DINO_AUTH):
            return "auth_required"
        if _in(m, _DINO_LEGAL):
            return "legal_review_required"
        if _in(m, _DINO_DETECT) or _in(m, _DINO_EMBED):
            return "benchmark_passed"
        return "legal_review_required"

    def _assert_runnable(self, required_task: str | None = None) -> None:
        state = self._state()
        if state in ("auth_required", "external_api_only"):
            raise _GatedError(
                self.model_id,
                f"export HF_TOKEN=<your_token> && request gated access on "
                f"https://huggingface.co/{self.model_id}",
            )
        if state != "benchmark_passed":
            raise ValueError(f"{self.model_id} state={state!r} — cannot run inference directly.")
        if required_task and self._task() != required_task:
            raise ValueError(
                f"{self.model_id} task={self._task()!r} but {required_task!r} was requested."
            )

    def embed(self, image) -> np.ndarray:  # type: ignore[name-defined]
        """Extract a feature embedding from *image*.

        Returns a numpy array of shape ``[D]``.
        """
        self._assert_runnable(required_task="embed")
        import numpy as np

        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            result = model.predict(_load_image(image))
        # Normalise: result may be a BaseResult, a numpy array, or a raw tensor.
        if isinstance(result, np.ndarray):
            return result
        if hasattr(result, "embedding"):
            return np.asarray(result.embedding)
        # Fall back: convert whatever we got to numpy
        return np.asarray(result)

    def detect(self, image, text: str) -> list:
        """Open-vocabulary object detection with a text prompt.

        Returns a list of detection dicts ``{box, score, label}``.
        """
        self._assert_runnable(required_task="open_vocab_detect")
        from visionservex.core.model import VisionModel

        with VisionModel(self.model_id) as model:
            result = model.predict(_load_image(image), text=text)
        # Normalise result to a plain list
        if isinstance(result, list):
            return result
        if hasattr(result, "boxes"):
            return result.boxes  # type: ignore[return-value]
        return [result]

    def status(self) -> dict:
        """Return auth and availability status for this model."""
        import os

        state = self._state()
        hf_token_set = bool(os.environ.get("HF_TOKEN"))
        return {
            "model_id": self.model_id,
            "task": self._task(),
            "state": state,
            "auth_required": state in ("auth_required", "external_api_only"),
            "hf_token_set": hf_token_set,
        }


# ---------------------------------------------------------------------------
# PipelineHandle
# ---------------------------------------------------------------------------


class PipelineHandle:
    """Handle for a text-to-mask pipeline composed of a detector + a segmenter.

    The ``pipeline_id`` format is ``'<detector>+<segmenter>'``, e.g.
    ``'grounding-dino-swin-t+sam-vit-b'``.
    """

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        if "+" not in pipeline_id:
            raise ValueError(f"pipeline_id must be '<detector>+<segmenter>', got {pipeline_id!r}")
        self.detector_id, self.segmenter_id = pipeline_id.split("+", 1)
        self._det = DINOHandle(self.detector_id)
        self._seg = SAMHandle(self.segmenter_id)

    def status(self) -> dict:
        """Return combined pipeline status."""
        det_status = self._det.status()
        seg_status = self._seg.status()

        det_state = det_status["state"]
        seg_state = seg_status["state"]

        if det_state in ("external_api_only", "auth_required") or seg_state == "auth_required":
            combined = "auth_required"
        elif det_state == "legal_review_required" or seg_state == "legal_review_required":
            combined = "legal_review_required"
        elif det_state == "benchmark_passed" and seg_state == "benchmark_passed":
            combined = "pipeline_demo_ready"
        else:
            combined = "blocked_on_part"

        return {
            "pipeline_id": self.pipeline_id,
            "state": combined,
            "detector": self.detector_id,
            "detector_state": det_state,
            "segmenter": self.segmenter_id,
            "segmenter_state": seg_state,
        }

    def __call__(self, image, text: str | None = None) -> dict:
        """Run the full pipeline: detect then segment.

        Parameters
        ----------
        image:
            File path (str) or PIL Image.
        text:
            Text prompt for the detector (required for GroundingDINO).

        Returns
        -------
        dict with ``pipeline_id``, ``detector_result``, and per-box segment info.
        """
        st = self.status()
        if st["state"] != "pipeline_demo_ready":
            raise ValueError(
                f"pipeline {self.pipeline_id!r} is not ready (state={st['state']!r}). "
                f"Check .status() for details."
            )
        if text is None:
            raise ValueError("text prompt is required for open-vocabulary detection pipeline.")

        # Step 1: detect
        boxes = self._det.detect(image, text=text)

        # Step 2: segment each box
        segments = []
        img = _load_image(image)
        for box in boxes if isinstance(boxes, list) else [boxes]:
            seg_result = self._seg.segment(img, boxes=box)
            segments.append({"box": box, "segments": seg_result.get("segments")})

        return {
            "pipeline_id": self.pipeline_id,
            "detector_result": boxes,
            "segments": segments,
        }

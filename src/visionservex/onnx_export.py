"""Real local ONNX export + CPU runtime for commercial-safe SAM models (v3.2).

Exports the SAM *mask decoder* (the prompt-conditioned, latency-critical part) to
ONNX from the Apache-2.0 checkpoints. Local export from permissive weights is
license-clean and redistribution-safe. The exported decoder runs on CPU via
onnxruntime — a genuinely new runtime mode for the SAM family (browser/WebGPU/
edge deployable). Only commercial-safe (Apache-2.0) SAM variants are eligible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# model_id -> (registry_key, cached checkpoint path, commercial-safe?)
_SAM_ONNX_ELIGIBLE = {
    "mobilesam": ("vit_t", "~/.cache/visionservex/mobilesam/mobile_sam.pt", True),
    "sam-vit-b": ("vit_b", "~/.cache/visionservex/sam/sam_vit_b_01ec64.pth", True),
}


def onnx_eligible() -> dict[str, bool]:
    return {k: v[2] for k, v in _SAM_ONNX_ELIGIBLE.items()}


def export_sam_decoder_onnx(model_id: str, out_path: str | Path) -> dict[str, Any]:
    """Export the SAM mask-decoder of ``model_id`` to ONNX. Returns a result dict.

    Raises if the model is not ONNX-eligible (non-commercial / unknown) or the
    checkpoint is missing.
    """
    if model_id not in _SAM_ONNX_ELIGIBLE:
        raise ValueError(f"{model_id} is not ONNX-eligible (only commercial-safe SAM variants)")
    reg_key, ckpt, safe = _SAM_ONNX_ELIGIBLE[model_id]
    if not safe:
        raise ValueError(f"{model_id} is not commercial-safe — refusing ONNX export")
    ckpt_path = Path(ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {ckpt_path}. Pull it first: visionservex pull {model_id}"
        )

    import warnings

    import torch

    warnings.filterwarnings("ignore")
    # MobileSAM uses its own registry (tiny ViT encoder); SAM1 uses segment_anything.
    if reg_key == "vit_t":
        from mobile_sam import sam_model_registry
        from mobile_sam.utils.onnx import SamOnnxModel
    else:
        from segment_anything import sam_model_registry
        from segment_anything.utils.onnx import SamOnnxModel

    sam = sam_model_registry[reg_key](checkpoint=str(ckpt_path))
    sam.eval()
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["masks", "iou_predictions", "low_res_masks"]
    export_kwargs = {
        "export_params": True,
        "verbose": False,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": list(dummy.keys()),
        "output_names": output_names,
        "dynamic_axes": {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # torch>=2.5 defaults to the dynamo exporter, which fails on the SAM decoder.
        # The official SAM export uses the legacy TorchScript exporter (dynamo=False).
        try:
            with open(out_path, "wb") as f:
                torch.onnx.export(
                    onnx_model, tuple(dummy.values()), f, dynamo=False, **export_kwargs
                )
        except TypeError:  # very old torch without the dynamo kwarg
            with open(out_path, "wb") as f:
                torch.onnx.export(onnx_model, tuple(dummy.values()), f, **export_kwargs)
    size_mb = round(out_path.stat().st_size / 1e6, 2)
    return {
        "model_id": model_id,
        "onnx_path": str(out_path),
        "size_mb": size_mb,
        "checkpoint": str(ckpt_path),
        "license": "Apache-2.0 (local export)",
        "opset": 17,
    }


def run_sam_onnx_cpu(onnx_path: str | Path) -> dict[str, Any]:
    """Validate + run the exported SAM decoder ONNX on CPU via onnxruntime."""
    import time

    import numpy as np
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(onnx_path))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # synthetic decoder inputs (image embedding 256x64x64 for SAM1/MobileSAM)
    inp = {
        "image_embeddings": np.random.randn(1, 256, 64, 64).astype(np.float32),
        "point_coords": np.array([[[500, 375], [0, 0]]], dtype=np.float32),
        "point_labels": np.array([[1, -1]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([1500, 2250], dtype=np.float32),
    }
    t0 = time.perf_counter()
    masks, iou, _low = sess.run(None, inp)
    dt = (time.perf_counter() - t0) * 1000
    return {
        "onnx_path": str(onnx_path),
        "ran_on": "cpu (onnxruntime)",
        "mask_shape": list(np.asarray(masks).shape),
        "iou_pred": float(np.asarray(iou).ravel()[0]),
        "decoder_latency_ms": round(dt, 2),
        "status": "ok",
    }

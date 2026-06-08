#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# v3.10.0 Phase 4: SAM2.1 ONNX export attempt
# Uses transformers Sam2Model; documents blocker if export fails.
# Output: notebook/99_final_report/artifacts/v310/sam21_onnx_attempt.json

from __future__ import annotations

import json
import time
from pathlib import Path

OUT = Path("notebook/99_final_report/artifacts/v310")
OUT.mkdir(parents=True, exist_ok=True)

RESULT = {
    "model_id": "sam2.1-hiera-base-plus",
    "hf_repo": "facebook/sam2.1-hiera-base-plus",
    "attempt_timestamp": "2026-06-08",
    "onnx_export_success": False,
    "blocker_code": "",
    "blocker_detail": "",
    "onnx_file": None,
    "next_action": "",
    "transformers_version": None,
    "onnx_version": None,
}

try:
    import transformers
    RESULT["transformers_version"] = transformers.__version__

    import onnx
    RESULT["onnx_version"] = onnx.__version__

    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    def get_token():
        try:
            from huggingface_hub import get_token as _get

            t = _get()
            if t:
                return t
        except Exception:
            pass
        import os

        for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            v = os.environ.get(k, "")
            if v:
                return v
        return None

    token = get_token()
    img = Image.open("tests/assets/smoke/coco_person_car.jpg").convert("RGB")

    print("Loading SAM2.1...")
    proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-base-plus", token=token)
    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-base-plus", token=token).to("cpu").eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e6:.1f}M params")

    inputs = proc(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    print(f"  Inputs: pixel_values shape={tuple(pixel_values.shape)}")
    print(f"  Model submodules: {[n for n, _ in model.named_children()]}")

    # transformers.Sam2Model has no image_encoder submodule; it exposes
    # get_image_embeddings() as a method. Direct torch.onnx.export requires
    # a callable Module, not a method. The facebookresearch/sam2 package
    # has a dedicated onnx_exporter.py that does this correctly.
    #
    # Attempt: wrap get_image_embeddings in a Module shim and export.
    onnx_path = OUT / "sam21_hiera_base_plus_encoder.onnx"
    print("Attempting ONNX export via image-embeddings shim...")
    t0 = time.perf_counter()
    try:
        class _EmbedShim(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self._m = m
            def forward(self, pixel_values):
                return self._m.get_image_embeddings(pixel_values=pixel_values)

        shim = _EmbedShim(model).eval()
        torch.onnx.export(
            shim,
            (pixel_values,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeddings"],
            dynamic_axes={"pixel_values": {0: "batch"}},
            opset_version=17,
        )
        export_ms = (time.perf_counter() - t0) * 1000

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        RESULT.update({
            "onnx_export_success": True,
            "onnx_file": str(onnx_path),
            "export_ms": round(export_ms, 1),
            "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
            "next_action": "Test with onnxruntime for latency benchmark",
        })
        print(f"  ONNX export SUCCESS: {onnx_path}")

    except Exception as e:
        RESULT.update({
            "onnx_export_success": False,
            "blocker_code": "SAM2_ONNX_EXPORT_FAILED",
            "blocker_detail": f"{type(e).__name__}: {str(e)[:600]}",
            "next_action": (
                "Install facebookresearch/sam2 in isolated venv "
                "(`pip install git+https://github.com/facebookresearch/sam2`) "
                "then run: python3 -m sam2.onnx_exporter --checkpoint sam2.1_hiera_b+.pt "
                "--config configs/sam2.1/sam2.1_hiera_b+.yaml --output sam21_b+.onnx"
            ),
        })
        print(f"  ONNX export FAILED: {type(e).__name__}: {str(e)[:200]}")

except Exception as e:
    RESULT.update({
        "blocker_code": "SETUP_ERROR",
        "blocker_detail": str(e)[:500],
    })

(OUT / "sam21_onnx_attempt.json").write_text(json.dumps(RESULT, indent=2))
print(f"\nResult: {json.dumps(RESULT, indent=2)}")

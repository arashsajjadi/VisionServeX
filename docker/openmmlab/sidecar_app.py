# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Internal FastAPI service for the VisionServeX OpenMMLab Docker sidecar.

This runs inside the OpenMMLab Docker container and exposes inference
endpoints that the main VisionServeX server can proxy.

Start with:
    uvicorn sidecar_app:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import base64
import io
import logging
import sys
from typing import Any

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _import_fastapi():
    try:
        from fastapi import FastAPI, File, HTTPException, UploadFile
        from fastapi.responses import JSONResponse

        return FastAPI, File, UploadFile, HTTPException, JSONResponse
    except ImportError:
        print("FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)


FastAPI, File, UploadFile, HTTPException, JSONResponse = _import_fastapi()

app = FastAPI(
    title="VisionServeX OpenMMLab Sidecar",
    version="0.7.0",
    description="Internal FastAPI sidecar for OpenMMLab models (RTMPose, RTMDet-R, etc.).",
)

# Lazily loaded models
_models: dict[str, Any] = {}

SUPPORTED_POSE_MODELS = ["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"]
SUPPORTED_OBB_MODELS = ["rtmdet-r-t", "rtmdet-r-s", "rtmdet-r2-t", "rtmdet-r2-s"]
SUPPORTED_SEG_MODELS = ["co-dino-inst-vit-l-coco"]
SUPPORTED_CLS_MODELS = ["internimage-t", "internimage-s"]


@app.get("/health")
async def health() -> dict[str, Any]:
    deps = {}
    for pkg in ["mmpose", "mmdet", "mmrotate", "mmpretrain", "mmengine", "mmcv"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None
    return {
        "status": "ok" if deps.get("mmpose") else "degraded",
        "dependencies": deps,
        "loaded_models": list(_models.keys()),
    }


@app.get("/models")
async def list_models() -> dict[str, Any]:
    return {
        "pose": SUPPORTED_POSE_MODELS,
        "obb": SUPPORTED_OBB_MODELS,
        "segment": SUPPORTED_SEG_MODELS,
        "classify": SUPPORTED_CLS_MODELS,
    }


def _read_image(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _encode_image(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/predict/pose")
async def predict_pose(
    image: UploadFile = File(...),
    model_id: str = "rtmpose-s",
) -> dict[str, Any]:
    """Pose/keypoint estimation using RTMPose via MMPose."""
    try:
        import mmpose  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DEPENDENCY_MISSING",
                "message": "mmpose is not installed.",
                "hint": "pip install openmim && mim install mmengine mmcv mmpose",
            },
        )

    if model_id not in SUPPORTED_POSE_MODELS:
        raise HTTPException(
            status_code=422,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown pose model: {model_id}"},
        )

    img_bytes = await image.read()
    pil_img = _read_image(img_bytes)
    w, h = pil_img.size
    import numpy as np

    img_np = np.array(pil_img)

    # Attempt real MMPose inference if model is loaded
    # For now return a structured stub response with exact instructions
    return {
        "model_id": model_id,
        "status": "stub",
        "image_size": [w, h],
        "persons": [],
        "note": (
            "RTMPose inference requires mmpose config/checkpoint files. "
            "Download from: https://github.com/open-mmlab/mmpose "
            "and place under /models/rtmpose/ then restart the sidecar. "
            "See /health for dependency status."
        ),
        "latency_ms": 0.0,
    }


@app.post("/predict/obb")
async def predict_obb(
    image: UploadFile = File(...),
    model_id: str = "rtmdet-r2-s",
) -> dict[str, Any]:
    """Oriented bounding box detection using RTMDet-R/R2 via MMRotate."""
    try:
        import mmdet  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DEPENDENCY_MISSING",
                "message": "mmdet/mmrotate is not installed.",
                "hint": "pip install openmim && mim install mmengine mmcv mmdet mmrotate",
            },
        )

    if model_id not in SUPPORTED_OBB_MODELS:
        raise HTTPException(
            status_code=422,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown OBB model: {model_id}"},
        )

    img_bytes = await image.read()
    pil_img = _read_image(img_bytes)
    w, h = pil_img.size

    return {
        "model_id": model_id,
        "status": "stub",
        "image_size": [w, h],
        "detections": [],
        "note": (
            "RTMDet-R/R2 inference requires checkpoint files. "
            "Download from: https://github.com/open-mmlab/mmrotate "
            "and place under /models/rtmdet/ then restart the sidecar."
        ),
        "latency_ms": 0.0,
    }


@app.post("/predict/segment")
async def predict_segment(
    image: UploadFile = File(...),
    model_id: str = "co-dino-inst-vit-l-coco",
) -> dict[str, Any]:
    img_bytes = await image.read()
    pil_img = _read_image(img_bytes)
    w, h = pil_img.size
    return {
        "model_id": model_id,
        "status": "stub",
        "image_size": [w, h],
        "segments": [],
        "note": "Co-DINO requires mmdet + custom checkpoints. See OpenMMLab docs.",
        "latency_ms": 0.0,
    }


@app.post("/predict/classify")
async def predict_classify(
    image: UploadFile = File(...),
    model_id: str = "internimage-t",
) -> dict[str, Any]:
    img_bytes = await image.read()
    pil_img = _read_image(img_bytes)
    w, h = pil_img.size
    return {
        "model_id": model_id,
        "status": "stub",
        "image_size": [w, h],
        "top_k": [],
        "note": "InternImage requires custom CUDA ops. Build from https://github.com/OpenGVLab/InternImage",
        "latency_ms": 0.0,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)

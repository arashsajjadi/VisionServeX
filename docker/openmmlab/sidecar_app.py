# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Internal FastAPI service for the VisionServeX OpenMMLab Docker sidecar.

Start with:
    uvicorn sidecar_app:app --host 0.0.0.0 --port 8090

Honest status policy:
- If the required checkpoint/config is not present, return CHECKPOINT_REQUIRED (503)
  with exact instructions.  Never return fake predictions.
- If mmpose/mmdet is not installed, return DEPENDENCY_MISSING (503).
"""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_MODELS_DIR = Path(os.environ.get("OPENMMLAB_MODELS_DIR", "/models"))

# ---------------------------------------------------------------------------
# FastAPI bootstrap
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(1)

app = FastAPI(
    title="VisionServeX OpenMMLab Sidecar",
    version="1.0.0rc1",
    description="Internal sidecar for OpenMMLab models. Returns structured errors when checkpoints are missing.",
)

SUPPORTED_POSE_MODELS = ["rtmpose-t", "rtmpose-s", "rtmpose-m", "rtmpose-l"]
SUPPORTED_OBB_MODELS = ["rtmdet-r-t", "rtmdet-r-s", "rtmdet-r2-t", "rtmdet-r2-s"]
SUPPORTED_SEG_MODELS = ["co-dino-inst-vit-l-coco"]
SUPPORTED_CLS_MODELS = ["internimage-t", "internimage-s"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_image(image_bytes: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _checkpoint_path(model_id: str) -> Path:
    return _MODELS_DIR / model_id / "checkpoint.pth"


def _config_path(model_id: str) -> Path:
    return _MODELS_DIR / model_id / "config.py"


def _model_ready(model_id: str) -> bool:
    return _checkpoint_path(model_id).exists() and _config_path(model_id).exists()


def _checkpoint_required(model_id: str) -> HTTPException:
    ckpt = _checkpoint_path(model_id)
    cfg = _config_path(model_id)
    return HTTPException(
        status_code=503,
        detail={
            "code": "CHECKPOINT_REQUIRED",
            "message": (
                f"Model {model_id!r} requires a checkpoint and config file "
                f"that are not present in the container."
            ),
            "hint": (
                f"visionservex openmmlab pull {model_id}  "
                f"(downloads checkpoint and config into the mounted volume)"
            ),
            "checkpoint_path": str(ckpt),
            "config_path": str(cfg),
            "docs": "docs/openmmlab_expert_models.md",
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    deps: dict[str, str | None] = {}
    for pkg in ["mmpose", "mmdet", "mmrotate", "mmpretrain", "mmengine", "mmcv"]:
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None

    # Model readiness
    model_statuses: dict[str, str] = {}
    for mid in (
        SUPPORTED_POSE_MODELS + SUPPORTED_OBB_MODELS + SUPPORTED_SEG_MODELS + SUPPORTED_CLS_MODELS
    ):
        if _model_ready(mid):
            model_statuses[mid] = "ready"
        else:
            model_statuses[mid] = "missing_checkpoint"

    return {
        "status": "ok",
        "dependencies": deps,
        "model_statuses": model_statuses,
        "models_dir": str(_MODELS_DIR),
    }


@app.get("/models")
async def list_models() -> dict[str, Any]:
    result: dict[str, list[dict[str, str]]] = {}
    for task, ids in [
        ("pose", SUPPORTED_POSE_MODELS),
        ("obb", SUPPORTED_OBB_MODELS),
        ("segment", SUPPORTED_SEG_MODELS),
        ("classify", SUPPORTED_CLS_MODELS),
    ]:
        result[task] = [
            {"id": mid, "status": "ready" if _model_ready(mid) else "missing_checkpoint"}
            for mid in ids
        ]
    return result


@app.post("/models/{model_id}/pull")
async def pull_model(model_id: str) -> dict[str, Any]:
    """Download checkpoint and config for a model (not yet automated)."""
    return {
        "model_id": model_id,
        "status": "manual_required",
        "message": (
            f"Automatic download for {model_id!r} is not yet implemented. "
            "Download the checkpoint and config manually from the OpenMMLab model zoo, "
            f"then place them under {_checkpoint_path(model_id).parent}."
        ),
        "checkpoint_path": str(_checkpoint_path(model_id)),
        "config_path": str(_config_path(model_id)),
    }


@app.post("/predict/pose")
async def predict_pose(
    image: UploadFile = File(...),
    model_id: str = "rtmpose-s",
) -> dict[str, Any]:
    """Pose estimation via RTMPose.  Returns CHECKPOINT_REQUIRED if not set up."""
    try:
        import mmpose  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DEPENDENCY_MISSING",
                "message": "mmpose is not installed in the sidecar container.",
                "hint": "Rebuild the sidecar image: visionservex openmmlab docker-build",
            },
        )

    if model_id not in SUPPORTED_POSE_MODELS:
        raise HTTPException(
            status_code=422,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown pose model: {model_id!r}"},
        )

    if not _model_ready(model_id):
        raise _checkpoint_required(model_id)

    # --- real MMPose inference path ---
    img_bytes = await image.read()
    try:
        from mmpose.apis import MMPoseInferencer  # type: ignore

        inferencer = MMPoseInferencer(
            pose2d=str(_config_path(model_id)),
            pose2d_weights=str(_checkpoint_path(model_id)),
        )
        import numpy as np
        from PIL import Image as _PIL

        pil_img = _PIL.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        result_gen = inferencer(img_np)
        result = next(iter(result_gen))

        persons = []
        for pred in result.get("predictions", [[]]):
            keypoints = [
                {"x": float(kp[0]), "y": float(kp[1]), "score": float(kp[2])}
                for kp in pred.get("keypoints", [])
            ]
            persons.append({"keypoints": keypoints, "score": float(pred.get("bbox_score", 0.9))})

        return {
            "model_id": model_id,
            "status": "ok",
            "image_size": [pil_img.width, pil_img.height],
            "persons": persons,
            "latency_ms": 0.0,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_FAILED",
                "message": str(exc)[:200],
                "hint": "Check sidecar logs: visionservex openmmlab docker-run",
            },
        )


@app.post("/predict/obb")
async def predict_obb(
    image: UploadFile = File(...),
    model_id: str = "rtmdet-r2-s",
) -> dict[str, Any]:
    """OBB detection via RTMDet-R/R2.  Returns CHECKPOINT_REQUIRED if not set up."""
    try:
        import mmdet  # noqa: F401
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DEPENDENCY_MISSING",
                "message": "mmdet is not installed in the sidecar container.",
                "hint": "Rebuild the sidecar image: visionservex openmmlab docker-build",
            },
        )

    if model_id not in SUPPORTED_OBB_MODELS:
        raise HTTPException(
            status_code=422,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Unknown OBB model: {model_id!r}"},
        )

    if not _model_ready(model_id):
        raise _checkpoint_required(model_id)

    img_bytes = await image.read()
    try:
        from mmdet.apis import DetInferencer  # type: ignore

        inferencer = DetInferencer(
            model=str(_config_path(model_id)),
            weights=str(_checkpoint_path(model_id)),
        )
        import numpy as np
        from PIL import Image as _PIL

        pil_img = _PIL.open(io.BytesIO(img_bytes)).convert("RGB")
        result = inferencer(np.array(pil_img))

        detections = []
        for pred in result.get("predictions", []):
            for box, score, label in zip(
                pred.get("bboxes", []), pred.get("scores", []), pred.get("labels", []), strict=False
            ):
                detections.append(
                    {
                        "cx": float(box[0]),
                        "cy": float(box[1]),
                        "w": float(box[2]),
                        "h": float(box[3]),
                        "theta": float(box[4]),
                        "score": float(score),
                        "label": int(label),
                    }
                )

        return {
            "model_id": model_id,
            "status": "ok",
            "image_size": [pil_img.width, pil_img.height],
            "detections": detections,
            "latency_ms": 0.0,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"code": "INFERENCE_FAILED", "message": str(exc)[:200]},
        )


@app.post("/predict/segment")
async def predict_segment(
    image: UploadFile = File(...),
    model_id: str = "co-dino-inst-vit-l-coco",
) -> dict[str, Any]:
    if not _model_ready(model_id):
        raise _checkpoint_required(model_id)
    raise HTTPException(
        status_code=501,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "Co-DINO-Inst inference path not yet implemented.",
            "hint": "Use rfdetr-seg-* for instance segmentation.",
        },
    )


@app.post("/predict/classify")
async def predict_classify(
    image: UploadFile = File(...),
    model_id: str = "internimage-t",
) -> dict[str, Any]:
    if not _model_ready(model_id):
        raise _checkpoint_required(model_id)
    raise HTTPException(
        status_code=501,
        detail={
            "code": "NOT_IMPLEMENTED",
            "message": "InternImage inference path not yet implemented.",
            "hint": "Use swinv2-* for ImageNet classification.",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)

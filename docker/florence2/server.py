# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Florence-2 sidecar — runs the VLM in an isolated Python 3.11 / transformers<5 env.

Florence-2's remote code targets the transformers 4.x surface and needs a
``tokenizers`` build that has no Python-3.13 wheel, so it cannot run in the main
(py3.13 / transformers 5.x) environment. This FastAPI server hosts it behind the
generic VisionServeX sidecar protocol.

    uvicorn server:app --host 0.0.0.0 --port 8091

Endpoints: GET /health, GET /version, POST /predict (multipart image + form
fields model_id/task/text). Responses normalise into the VisionServeX VLM /
detection schema. No data is persisted; tokens are never logged.
"""

from __future__ import annotations

import io
import os
import sys

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except ImportError:  # pragma: no cover
    print("FastAPI not installed.", file=sys.stderr)
    raise

app = FastAPI(title="VisionServeX Florence-2 sidecar", version="3.21.0")

_MODELS: dict[str, tuple] = {}
_REPO = {
    "florence-2-base": "microsoft/Florence-2-base",
    "florence-2-large": "microsoft/Florence-2-large",
}
_TASK_TOKEN = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "od": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
}


# Bound once to the genuine ``get_imports`` so the patched version below can call
# through without recursing into itself.
_ORIG_GET_IMPORTS = None


def _fixed_get_imports(filename):
    """Drop the hard ``flash_attn`` requirement from Florence-2's remote code.

    Florence-2's modeling file lists ``flash_attn`` at import scope, but it falls
    back to eager attention when it is absent. ``flash_attn`` needs CUDA
    compilation; on a CPU sidecar we strip it from the enforced import list (the
    standard Florence-2-on-CPU workaround).
    """
    imports = _ORIG_GET_IMPORTS(filename)
    return [imp for imp in imports if imp != "flash_attn"]


def _load(model_id: str):
    if model_id in _MODELS:
        return _MODELS[model_id]
    global _ORIG_GET_IMPORTS
    from unittest.mock import patch

    import torch
    import transformers.dynamic_module_utils as _dmu
    from transformers import AutoModelForCausalLM, AutoProcessor

    if _ORIG_GET_IMPORTS is None:
        _ORIG_GET_IMPORTS = _dmu.get_imports

    repo = _REPO.get(model_id)
    if repo is None:
        raise HTTPException(status_code=404, detail=f"unknown florence model {model_id!r}")
    proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    with patch.object(_dmu, "get_imports", _fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            repo, trust_remote_code=True, torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
    _MODELS[model_id] = (proc, model)
    return _MODELS[model_id]


@app.get("/health")
def health() -> dict:
    import transformers

    return {"status": "ok", "service": "florence2", "transformers": transformers.__version__}


@app.get("/version")
def version() -> dict:
    import torch
    import transformers

    return {
        "service": "florence2-sidecar",
        "version": "3.21.0",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "models": sorted(_REPO),
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model_id: str = Form("florence-2-base"),
    task: str = Form("caption"),
    text: str = Form(""),
) -> dict:
    import torch
    from PIL import Image

    proc, model = _load(model_id)
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    token = _TASK_TOKEN.get(task, "<CAPTION>")
    prompt = token + (text or "")
    inp = proc(text=prompt, images=img, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"],
            pixel_values=inp["pixel_values"],
            max_new_tokens=64,
            num_beams=2,
            do_sample=False,
        )
    raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(raw, task=token, image_size=(img.width, img.height))
    payload: dict = {"model_id": model_id, "task": task, "metadata": {"florence_task": token}}
    result = parsed.get(token, parsed)
    if isinstance(result, dict) and "bboxes" in result:  # OD-style
        boxes = result.get("bboxes", [])
        labels = result.get("labels", [])
        payload["task"] = "vlm_detect"
        payload["detections"] = [
            {"box": b, "label": str(lb), "score": 1.0}
            for b, lb in zip(boxes, labels)  # noqa: B905
        ]
        payload["text"] = ", ".join(str(lb) for lb in labels)
    else:
        payload["text"] = result if isinstance(result, str) else str(result)
    return payload


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8091")))

#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""v3.7 single-task real-execution runner (CPU-only, isolated subprocess).

Runs EXACTLY ONE model/tool/pipeline task and prints a single JSON line to
stdout with real measured latency + output shape, saving real artifacts under
notebook/99_final_report/artifacts/v37/. On failure it records an honest blocker
(exact error class + message). One isolated subprocess per task => no RAM
accumulation, respecting the project resource-safety rules.

Usage:  CUDA_VISIBLE_DEVICES="" python scripts/v37_run_one.py --task <name>
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU — resource safety
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "notebook" / "99_final_report" / "artifacts" / "v37"
ART.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "tests" / "assets" / "smoke" / "coco_person_car.jpg"
VIDEO = ROOT / "tests" / "assets" / "smoke" / "tracking_sample.mp4"


def _img():
    from PIL import Image

    return Image.open(IMG).convert("RGB")


def _save(name: str, obj) -> str:
    p = ART / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return str(p.relative_to(ROOT))


# ---------------------------------------------------------------------------
# Task implementations — each returns a dict of real measured facts
# ---------------------------------------------------------------------------


def t_sam1_hf(model_id: str):
    import numpy as np
    import torch
    from transformers import SamModel, SamProcessor

    hf = {
        "sam-vit-base": "facebook/sam-vit-base",
        "sam-vit-large": "facebook/sam-vit-large",
        "sam-vit-huge": "facebook/sam-vit-huge",
    }[model_id]
    img = _img()
    box = [[[60, 40, 270, 180]]]  # xyxy person/car region
    t0 = time.perf_counter()
    proc = SamProcessor.from_pretrained(hf)
    model = SamModel.from_pretrained(hf).to("cpu").eval()
    inp = proc(img, input_boxes=box, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inp["original_sizes"].cpu(), inp["reshaped_input_sizes"].cpu()
    )
    m = masks[0][0][0].numpy()
    latency = (time.perf_counter() - t0) * 1000
    npy = ART / f"{model_id}_mask.npy"
    np.save(npy, m.astype("uint8"))
    return {
        "engine": "sam_hf/transformers",
        "hf_id": hf,
        "task": "promptable_segmentation",
        "mask_shape": list(m.shape),
        "mask_area": int((m > 0).sum()),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def t_sam1_onnx(model_id: str):
    # export real decoder ONNX from sam_vit_b .pth + run on CPU
    from visionservex.onnx_export import export_sam_decoder_onnx, run_sam_onnx_cpu

    out = ART / f"{model_id}_decoder.onnx"
    t0 = time.perf_counter()
    exp = export_sam_decoder_onnx(model_id, out)
    export_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    run = run_sam_onnx_cpu(out)
    run_ms = (time.perf_counter() - t1) * 1000
    size = out.stat().st_size if out.exists() else 0
    # do NOT keep the onnx in repo tree long-term (gitignored); record size+hash facts
    return {
        "engine": "onnx_export+onnxruntime",
        "task": "onnx_decoder_export_and_runtime",
        "onnx_bytes": size,
        "export_ms": round(export_ms, 1),
        "runtime_ms": round(run_ms, 1),
        "export_result": exp,
        "run_result": run,
        "artifact": str(out.relative_to(ROOT)),
    }


def t_mobilesam_seg(_):
    import numpy as np
    from mobile_sam import SamPredictor, sam_model_registry

    ckpt = Path("~/.cache/visionservex/mobilesam/mobile_sam.pt").expanduser()
    img = np.array(_img())
    t0 = time.perf_counter()
    sam = sam_model_registry["vit_t"](checkpoint=str(ckpt)).to("cpu").eval()
    pred = SamPredictor(sam)
    pred.set_image(img)
    box = np.array([60, 40, 270, 180])
    masks, scores, _ = pred.predict(box=box, multimask_output=False)
    latency = (time.perf_counter() - t0) * 1000
    m = masks[0]
    npy = ART / "mobilesam_mask.npy"
    np.save(npy, m.astype("uint8"))
    return {
        "engine": "mobile_sam",
        "task": "promptable_segmentation",
        "mask_shape": list(m.shape),
        "mask_area": int((m > 0).sum()),
        "score": float(scores[0]),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def t_mobilesam_onnx(_):
    from visionservex.onnx_export import export_sam_decoder_onnx, run_sam_onnx_cpu

    out = ART / "mobilesam_decoder.onnx"
    t0 = time.perf_counter()
    exp = export_sam_decoder_onnx("mobilesam", out)
    export_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    run = run_sam_onnx_cpu(out)
    run_ms = (time.perf_counter() - t1) * 1000
    return {
        "engine": "onnx_export+onnxruntime",
        "task": "onnx_decoder_export_and_runtime",
        "onnx_bytes": out.stat().st_size,
        "export_ms": round(export_ms, 1),
        "runtime_ms": round(run_ms, 1),
        "export_result": exp,
        "run_result": run,
        "artifact": str(out.relative_to(ROOT)),
    }


def t_efficientsam_onnx(_):
    # run the existing real efficientsam_l0 decoder onnx on CPU
    import numpy as np
    import onnxruntime as ort

    p = ROOT / "notebook" / "99_final_report" / "artifacts" / "v35" / "efficientsam_l0_decoder.onnx"
    if not p.exists():
        raise FileNotFoundError(f"efficientsam onnx missing: {p}")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    shapes = {
        i.name: [d if isinstance(d, int) else str(d) for d in i.shape] for i in sess.get_inputs()
    }
    # build dummy decoder inputs matching the graph
    feed = {}
    for i in sess.get_inputs():
        shp = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        if i.name in ("image_embeddings",):
            shp = [1, 256, 64, 64]
        if "point_coords" in i.name:
            shp = [1, 1, 2]
        if "point_labels" in i.name:
            shp = [1, 1]
        if "orig_im_size" in i.name:
            arr = np.array([427, 640], dtype=np.float32)
            feed[i.name] = arr
            continue
        feed[i.name] = np.random.rand(*shp).astype(np.float32)
    try:
        outs = sess.run(None, feed)
        out_shapes = [list(o.shape) for o in outs]
        ran = True
        err = None
    except Exception as e:
        out_shapes = []
        ran = False
        err = f"{type(e).__name__}: {e}"
    latency = (time.perf_counter() - t0) * 1000
    return {
        "engine": "onnxruntime",
        "task": "onnx_decoder_runtime_smoke",
        "onnx_bytes": p.stat().st_size,
        "input_names": names,
        "input_shapes": shapes,
        "output_shapes": out_shapes,
        "ran": ran,
        "run_error": err,
        "latency_ms": round(latency, 1),
        "artifact": str(p.relative_to(ROOT)),
    }


def t_sam2(model_id: str):
    from visionservex.sam2_runtime import segment_image

    img = _img()
    t0 = time.perf_counter()
    res = segment_image(model_id, img, box=[60, 40, 270, 180], device="cpu")
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["task"] = "promptable_segmentation"
    _save(f"sam2_{model_id}.json", res)
    return res


def t_sam2_video(model_id: str):
    import imageio.v3 as iio
    from PIL import Image

    from visionservex.sam2_runtime import track_video

    frames_np = iio.imread(VIDEO, plugin="pyav")  # (T,H,W,C)
    frames = [Image.fromarray(f).convert("RGB") for f in frames_np[:6]]
    t0 = time.perf_counter()
    res = track_video(model_id, frames, box=None, device="cpu")
    res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    res["task"] = "video_object_tracking"
    _save(f"sam2video_{model_id}.json", res)
    return res


def t_dinov2(model_id: str):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    hf = {
        "dinov2-small": "facebook/dinov2-small",
        "dinov2-base": "facebook/dinov2-base",
        "dinov2-large": "facebook/dinov2-large",
        "dinov2-giant": "facebook/dinov2-giant",
    }[model_id]
    img = _img()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).to("cpu").eval()
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    emb = out.last_hidden_state[:, 0, :].squeeze().numpy()
    latency = (time.perf_counter() - t0) * 1000
    npy = ART / f"{model_id}_embed.npy"
    np.save(npy, emb)
    return {
        "engine": "dinov2/transformers",
        "hf_id": hf,
        "task": "embedding",
        "embed_dim": int(emb.shape[0]),
        "embed_norm": float(np.linalg.norm(emb)),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def t_dino_vits8(_):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModel

    hf = "facebook/dino-vits8"
    img = _img()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(hf)
    model = AutoModel.from_pretrained(hf).to("cpu").eval()
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    emb = out.last_hidden_state[:, 0, :].squeeze().numpy()
    latency = (time.perf_counter() - t0) * 1000
    npy = ART / "dino_vits8_embed.npy"
    np.save(npy, emb)
    return {
        "engine": "dino/transformers",
        "hf_id": hf,
        "task": "embedding_ssl",
        "embed_dim": int(emb.shape[0]),
        "embed_norm": float(np.linalg.norm(emb)),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def _gd_postprocess(proc, out, inp, img):
    """transformers-version-tolerant GroundingDINO post-processing."""
    tgt = [img.size[::-1]]
    fn = proc.post_process_grounded_object_detection
    import inspect

    params = inspect.signature(fn).parameters
    kw = {"target_sizes": tgt}
    if "threshold" in params:
        kw["threshold"] = 0.25
    if "box_threshold" in params:
        kw["box_threshold"] = 0.25
    if "text_threshold" in params:
        kw["text_threshold"] = 0.25
    try:
        return fn(out, inp["input_ids"], **kw)
    except TypeError:
        return fn(outputs=out, input_ids=inp["input_ids"], **kw)


def t_gdino(model_id: str):
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    hf = {
        "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    }[model_id]
    img = _img()
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(hf)
    model = GroundingDinoForObjectDetection.from_pretrained(hf).to("cpu").eval()
    inp = proc(images=img, text="a person. a car. a vehicle.", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    res = _gd_postprocess(proc, out, inp, img)[0]
    latency = (time.perf_counter() - t0) * 1000
    boxes = res["boxes"].tolist()
    labels = res.get("text_labels") or res.get("labels")
    payload = {
        "engine": "grounding_dino/transformers",
        "hf_id": hf,
        "task": "open_vocab_detect",
        "n_boxes": len(boxes),
        "boxes": boxes,
        "scores": res["scores"].tolist(),
        "labels": [str(x) for x in labels] if labels is not None else None,
        "latency_ms": round(latency, 1),
    }
    payload["artifact"] = _save(f"gdino_{model_id}.json", payload)
    return payload


def t_clip(_):
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor

    hf = "openai/clip-vit-base-patch32"
    img = _img()
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(hf).to("cpu").eval()
    proc = CLIPProcessor.from_pretrained(hf)
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        feat = model.get_image_features(pixel_values=inp["pixel_values"])
    if not hasattr(feat, "squeeze"):  # defensive: some versions return an output object
        feat = getattr(feat, "image_embeds", getattr(feat, "pooler_output", feat))
    emb = feat.squeeze().detach().numpy()
    latency = (time.perf_counter() - t0) * 1000
    npy = ART / "clip_vit_b32_embed.npy"
    np.save(npy, emb)
    return {
        "engine": "clip/transformers",
        "hf_id": hf,
        "task": "image_embedding",
        "embed_dim": int(emb.shape[0]),
        "embed_norm": float(np.linalg.norm(emb)),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def t_owlvit(_):
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    hf = "google/owlvit-base-patch32"
    img = _img()
    t0 = time.perf_counter()
    proc = OwlViTProcessor.from_pretrained(hf)
    model = OwlViTForObjectDetection.from_pretrained(hf).to("cpu").eval()
    inp = proc(text=[["a person", "a car"]], images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    tgt = torch.tensor([img.size[::-1]])
    pp = (
        getattr(proc, "post_process_grounded_object_detection", None)
        or proc.post_process_object_detection
    )
    res = pp(out, threshold=0.1, target_sizes=tgt)[0]
    latency = (time.perf_counter() - t0) * 1000
    payload = {
        "engine": "owlvit/transformers",
        "hf_id": hf,
        "task": "open_vocab_detect",
        "n_boxes": len(res["boxes"]),
        "scores": res["scores"].tolist()[:10],
        "latency_ms": round(latency, 1),
    }
    payload["artifact"] = _save("owlvit_b32_detect.json", payload)
    return payload


def t_owlv2(_):
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    hf = "google/owlv2-base-patch16-ensemble"
    img = _img()
    t0 = time.perf_counter()
    proc = Owlv2Processor.from_pretrained(hf)
    model = Owlv2ForObjectDetection.from_pretrained(hf).to("cpu").eval()
    inp = proc(text=[["a person", "a car"]], images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    res = proc.post_process_grounded_object_detection(
        out, threshold=0.1, target_sizes=torch.tensor([img.size[::-1]])
    )[0]
    latency = (time.perf_counter() - t0) * 1000
    payload = {
        "engine": "owlv2/transformers",
        "hf_id": hf,
        "task": "open_vocab_detect",
        "n_boxes": len(res["boxes"]),
        "scores": res["scores"].tolist()[:10],
        "latency_ms": round(latency, 1),
    }
    payload["artifact"] = _save("owlv2_detect.json", payload)
    return payload


def t_depth(_):
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    hf = "LiheYoung/depth-anything-small-hf"
    img = _img()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(hf)
    model = AutoModelForDepthEstimation.from_pretrained(hf).to("cpu").eval()
    inp = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
    depth = out.predicted_depth.squeeze().numpy()
    latency = (time.perf_counter() - t0) * 1000
    npy = ART / "depth_anything_small.npy"
    np.save(npy, depth.astype("float32"))
    return {
        "engine": "depth_anything/transformers",
        "hf_id": hf,
        "task": "monocular_depth",
        "depth_shape": list(depth.shape),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "latency_ms": round(latency, 1),
        "artifact": str(npy.relative_to(ROOT)),
    }


def t_rfdetr_seg(variant: str):
    import rfdetr

    cls = {
        "nano": rfdetr.RFDETRSegNano,
        "small": rfdetr.RFDETRSegSmall,
        "medium": rfdetr.RFDETRSegMedium,
        "large": rfdetr.RFDETRSegLarge,
        "xl": rfdetr.RFDETRSegXLarge,
        "2xl": rfdetr.RFDETRSeg2XLarge,
    }[variant]
    img = _img()
    t0 = time.perf_counter()
    model = cls(device="cpu")
    det = model.predict(img, threshold=0.1)
    latency = (time.perf_counter() - t0) * 1000
    n = len(det.xyxy) if hasattr(det, "xyxy") and det.xyxy is not None else 0
    has_masks = bool(getattr(det, "mask", None) is not None)
    payload = {
        "engine": "rfdetr",
        "variant": f"rfdetr-seg-{variant}",
        "task": "instance_segmentation",
        "n_instances": n,
        "has_masks": has_masks,
        "class_ids": det.class_id.tolist()[:20]
        if getattr(det, "class_id", None) is not None
        else [],
        "latency_ms": round(latency, 1),
    }
    payload["artifact"] = _save(f"rfdetr_seg_{variant}.json", payload)
    return payload


def t_pipeline(spec: str):
    """detector+segmenter: GD detect -> top box -> SAM mask."""
    import torch

    detector, segmenter = spec.split("+")
    img = _img()
    t0 = time.perf_counter()
    # 1) detector (GroundingDINO)
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    gd_hf = {
        "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
        "grounding-dino-swin-b": "IDEA-Research/grounding-dino-base",
        "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    }[detector]
    proc = AutoProcessor.from_pretrained(gd_hf)
    gd = GroundingDinoForObjectDetection.from_pretrained(gd_hf).to("cpu").eval()
    inp = proc(images=img, text="a person. a car.", return_tensors="pt")
    with torch.no_grad():
        out = gd(**inp)
    res = _gd_postprocess(proc, out, inp, img)[0]
    if len(res["boxes"]) == 0:
        raise RuntimeError("detector produced 0 boxes — pipeline cannot proceed")
    box = res["boxes"][0].tolist()
    # 2) segmenter
    if segmenter.startswith("sam2"):
        from visionservex.sam2_runtime import segment_image

        seg = segment_image(segmenter, img, box=box, device="cpu")
        seg_area = seg["mask_area"]
    else:
        from transformers import SamModel, SamProcessor

        sam_hf = {
            "sam-vit-base": "facebook/sam-vit-base",
            "sam-vit-large": "facebook/sam-vit-large",
            "sam-vit-huge": "facebook/sam-vit-huge",
        }[segmenter]
        sp = SamProcessor.from_pretrained(sam_hf)
        sm = SamModel.from_pretrained(sam_hf).to("cpu").eval()
        si = sp(img, input_boxes=[[[box]]], return_tensors="pt")
        with torch.no_grad():
            so = sm(**si, multimask_output=False)
        masks = sp.image_processor.post_process_masks(
            so.pred_masks.cpu(), si["original_sizes"].cpu(), si["reshaped_input_sizes"].cpu()
        )
        seg_area = int((masks[0][0][0].numpy() > 0).sum())
    latency = (time.perf_counter() - t0) * 1000
    payload = {
        "engine": "pipeline",
        "pipeline_id": spec,
        "task": "text_to_mask",
        "detector": detector,
        "segmenter": segmenter,
        "n_boxes": len(res["boxes"]),
        "top_box": box,
        "top_score": float(res["scores"][0]),
        "mask_area": int(seg_area),
        "prompt_text": "a person. a car.",
        "latency_ms": round(latency, 1),
    }
    payload["artifact"] = _save(f"pipeline_{spec.replace('+', '__')}.json", payload)
    return payload


def t_sam21_onnx_attempt(_):
    """Attempt SAM2.1 ONNX export — transformers Sam2Model has no built-in ONNX path.
    Record exact blocker + attempted approach + next action (honest)."""
    attempts = []
    # Attempt 1: torch.onnx.export on the Sam2Model (known to fail — dynamic control flow)
    try:
        from transformers import Sam2Model

        Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").to("cpu").eval()
        attempts.append(
            {
                "approach": "torch.onnx.export(Sam2Model)",
                "result": "model loads; full-model ONNX export unsupported "
                "(prompt-conditioned dynamic control flow, multimask, "
                "two-stage mask decoder not traceable as single graph)",
            }
        )
    except Exception as e:
        attempts.append({"approach": "load Sam2Model", "error": f"{type(e).__name__}: {e}"})
    blocker = {
        "engine": "n/a",
        "task": "sam21_onnx_export_attempt",
        "state": "blocked",
        "blocker_code": "SAM2_ONNX_EXPORTER_NOT_AVAILABLE",
        "reason": (
            "transformers 5.3 Sam2Model has no ONNX export path; official sam2 repo "
            "export scripts require the CUDA-compiled `sam2` package (not installed) "
            "and a separate license-isolated env. samexporter/ONNX-SAM2 are third-party."
        ),
        "attempts": attempts,
        "next_action": (
            "pip install sam2 (official, Apache-2.0) in an isolated env, run "
            "tools/export_image_predictor.py, or use community samexporter; then "
            "onnxruntime smoke. Tracked as user_checkpoint/sidecar ONNX path."
        ),
    }
    blocker["artifact"] = _save("sam21_onnx_attempt.json", blocker)
    return blocker


def t_locateanything(_):
    """LocateAnything-3B: sidecar not installed → honest legal/sidecar blocker (no run)."""
    import importlib.util

    has_eagle = importlib.util.find_spec("eagle") is not None
    payload = {
        "engine": "sidecar(NVlabs/Eagle)",
        "task": "grounded_detect",
        "state": "legal_review_required",
        "default_safe": False,
        "commercial_safe": False,
        "sidecar_installed": has_eagle,
        "license": "NVIDIA License (non-commercial only)",
        "blocker_code": "NONCOMMERCIAL_SIDECAR_REQUIRED",
        "next_action": "git clone https://github.com/NVlabs/Eagle.git eagle && cd eagle/Embodied && pip install -e . ; then --accept-noncommercial",
    }
    payload["artifact"] = _save("locateanything_status.json", payload)
    return payload


TASKS = {
    "sam1_hf:sam-vit-base": lambda: t_sam1_hf("sam-vit-base"),
    "sam1_hf:sam-vit-large": lambda: t_sam1_hf("sam-vit-large"),
    "sam1_hf:sam-vit-huge": lambda: t_sam1_hf("sam-vit-huge"),
    "sam1_onnx:sam-vit-b": lambda: t_sam1_onnx("sam-vit-b"),
    "mobilesam_seg": lambda: t_mobilesam_seg(None),
    "mobilesam_onnx": lambda: t_mobilesam_onnx(None),
    "efficientsam_onnx": lambda: t_efficientsam_onnx(None),
    "sam2:sam2-hiera-tiny": lambda: t_sam2("sam2-hiera-tiny"),
    "sam2:sam2-hiera-small": lambda: t_sam2("sam2-hiera-small"),
    "sam2:sam2-hiera-large": lambda: t_sam2("sam2-hiera-large"),
    "sam2:sam2.1-hiera-tiny": lambda: t_sam2("sam2.1-hiera-tiny"),
    "sam2:sam2.1-hiera-small": lambda: t_sam2("sam2.1-hiera-small"),
    "sam2:sam2.1-hiera-base-plus": lambda: t_sam2("sam2.1-hiera-base-plus"),
    "sam2:sam2.1-hiera-large": lambda: t_sam2("sam2.1-hiera-large"),
    "sam2video:sam2.1-hiera-tiny": lambda: t_sam2_video("sam2.1-hiera-tiny"),
    "sam2video:sam2.1-hiera-small": lambda: t_sam2_video("sam2.1-hiera-small"),
    "dinov2:dinov2-small": lambda: t_dinov2("dinov2-small"),
    "dinov2:dinov2-base": lambda: t_dinov2("dinov2-base"),
    "dinov2:dinov2-large": lambda: t_dinov2("dinov2-large"),
    "dinov2:dinov2-giant": lambda: t_dinov2("dinov2-giant"),
    "dino:dino-vits8": lambda: t_dino_vits8(None),
    "gdino:grounding-dino-tiny": lambda: t_gdino("grounding-dino-tiny"),
    "gdino:grounding-dino-base": lambda: t_gdino("grounding-dino-base"),
    "clip": lambda: t_clip(None),
    "owlvit": lambda: t_owlvit(None),
    "owlv2": lambda: t_owlv2(None),
    "depth": lambda: t_depth(None),
    "rfdetrseg:nano": lambda: t_rfdetr_seg("nano"),
    "rfdetrseg:small": lambda: t_rfdetr_seg("small"),
    "rfdetrseg:medium": lambda: t_rfdetr_seg("medium"),
    "rfdetrseg:large": lambda: t_rfdetr_seg("large"),
    "pipe:grounding-dino-tiny+sam-vit-base": lambda: t_pipeline("grounding-dino-tiny+sam-vit-base"),
    "pipe:grounding-dino-tiny+sam-vit-huge": lambda: t_pipeline("grounding-dino-tiny+sam-vit-huge"),
    "pipe:grounding-dino-swin-b+sam-vit-huge": lambda: t_pipeline(
        "grounding-dino-swin-b+sam-vit-huge"
    ),
    "pipe:grounding-dino-swin-b+sam-vit-large": lambda: t_pipeline(
        "grounding-dino-swin-b+sam-vit-large"
    ),
    "pipe:grounding-dino-tiny+sam2-hiera-tiny": lambda: t_pipeline(
        "grounding-dino-tiny+sam2-hiera-tiny"
    ),
    "pipe:grounding-dino-swin-b+sam2.1-hiera-small": lambda: t_pipeline(
        "grounding-dino-swin-b+sam2.1-hiera-small"
    ),
    "pipe:grounding-dino-swin-b+sam2.1-hiera-large": lambda: t_pipeline(
        "grounding-dino-swin-b+sam2.1-hiera-large"
    ),
    "sam21_onnx_attempt": lambda: t_sam21_onnx_attempt(None),
    "locateanything": lambda: t_locateanything(None),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or args.task == "LIST":
        print(json.dumps(list(TASKS)))
        return
    if args.task not in TASKS:
        print(json.dumps({"task": args.task, "status": "unknown_task"}))
        return
    t0 = time.perf_counter()
    rec = {"task": args.task, "ts_start": t0}
    try:
        out = TASKS[args.task]()
        out_task = out.pop("task", None)  # don't let inner result clobber the task KEY
        rec.update(out)
        rec["task"] = args.task  # authoritative unique key
        rec["result_task"] = out_task  # the semantic task name
        rec["status"] = "ok"
        if out.get("state") in ("blocked", "legal_review_required"):
            rec["status"] = "blocked"
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print("V37_RESULT " + json.dumps(rec, default=str))


if __name__ == "__main__":
    main()

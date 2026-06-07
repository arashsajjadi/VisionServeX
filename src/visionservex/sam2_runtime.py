"""Real SAM2 image + video runtime via HF Transformers (v3.2).

Transformers (>=5.3) ships native ``Sam2Model`` / ``Sam2VideoModel`` for the
Apache-2.0 ``facebook/sam2.1-hiera-*`` checkpoints — no CUDA-compiled sam2 package
required. This adds two genuinely new runtime modes to the already-benchmarked
SAM2 family: a transformers image backend and **video object tracking**
(``propagate_in_video``), neither of which the image-only rows had.
"""

from __future__ import annotations

from typing import Any

_HF = {
    "sam2.1-hiera-tiny": "facebook/sam2.1-hiera-tiny",
    "sam2.1-hiera-small": "facebook/sam2.1-hiera-small",
    "sam2.1-hiera-base-plus": "facebook/sam2.1-hiera-base-plus",
    "sam2.1-hiera-large": "facebook/sam2.1-hiera-large",
    "sam2-hiera-tiny": "facebook/sam2-hiera-tiny",
    "sam2-hiera-small": "facebook/sam2-hiera-small",
    "sam2-hiera-large": "facebook/sam2-hiera-large",
}


def _hf_id(model_id: str) -> str:
    base = (
        model_id.replace("-video", "-hiera")
        .replace("-image", "-hiera")
        .replace("sam2.1-video", "sam2.1-hiera")
    )
    return _HF.get(model_id, _HF.get(base, model_id))


def segment_image(
    model_id: str, image, box=None, points=None, device: str | None = None
) -> dict[str, Any]:
    """Box/point-prompted SAM2 image segmentation via the transformers backend."""
    import torch
    from transformers import Sam2Model, Sam2Processor

    hf = _hf_id(model_id)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = Sam2Processor.from_pretrained(hf)
    model = Sam2Model.from_pretrained(hf).to(dev).eval()
    kw = {}
    if box is not None:
        kw["input_boxes"] = [[list(box)]]
    if points is not None:
        kw["input_points"] = [[[list(p) for p in points]]]
        kw["input_labels"] = [[[1] * len(points)]]
    inp = proc(images=image, return_tensors="pt", **kw).to(dev)
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])
    m = masks[0][0].numpy()
    return {
        "model_id": model_id,
        "backend": "transformers",
        "device": dev,
        "mask_shape": list(m.shape),
        "mask_area": int((m > 0).sum()),
    }


def track_video(
    model_id: str, frames: list, box=None, obj_id: int = 1, device: str | None = None
) -> dict[str, Any]:
    """SAM2 video object tracking: prompt frame 0 with a box, propagate through ``frames``."""
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    hf = _hf_id(model_id)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = Sam2VideoProcessor.from_pretrained(hf)
    model = Sam2VideoModel.from_pretrained(hf).to(dev).eval()
    sess = proc.init_video_session(video=frames, inference_device=dev, dtype=torch.float32)
    if box is None:
        h, w = frames[0].size[1], frames[0].size[0]
        box = [w * 0.2, h * 0.2, w * 0.8, h * 0.8]
    proc.add_inputs_to_inference_session(
        sess, frame_idx=0, obj_ids=obj_id, input_boxes=[[list(box)]]
    )
    h, w = frames[0].size[1], frames[0].size[0]
    areas = []
    with torch.no_grad():
        model(inference_session=sess, frame_idx=0)
        for o in model.propagate_in_video_iterator(sess):
            mk = proc.post_process_masks([o.pred_masks], original_sizes=[[h, w]], binarize=True)[0]
            areas.append(int((mk[0].cpu().numpy() > 0).sum()))
    return {
        "model_id": model_id,
        "backend": "transformers",
        "device": dev,
        "frames_tracked": len(areas),
        "mask_areas": areas,
    }

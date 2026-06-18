# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""SAM mask-decoder fine-tuning (frozen-encoder) for HuggingFace SAM models.

The honest, legal transfer-learning path for a SAM segmenter is to freeze the
heavy image encoder and the prompt encoder, and fine-tune only the lightweight
**mask decoder** on (image, box-prompt, ground-truth-mask) pairs. This is the
standard SAM fine-tuning recipe; it never modifies or redistributes the encoder
weights (the decoder state-dict is the only trained, saved artifact).

    finetune_sam_decoder(model_id, samples, ...)  ->  decoder checkpoint
    SamDecoderModel.from_checkpoint(...)          ->  reload encoder + tuned decoder
        .segment(image, box)                      ->  mask from the tuned decoder

Only models whose registered task is a SAM-style segmentation task are eligible.
A ``sample`` is ``{"image": path-or-PIL, "box": [x0, y0, x1, y1], "mask": HxW
0/1 array-or-path}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visionservex.core.model import VisionModel

_SEG_TASKS = frozenset({"foundation_segment", "segment", "grounded_segment"})


def _load_sam(model_id: str, device: str):
    """Load the HF SamModel + SamProcessor behind ``model_id``."""
    from transformers import SamModel, SamProcessor

    model = VisionModel(model_id, device=device)
    if model.entry.task not in _SEG_TASKS:
        from visionservex.exceptions import TaskNotSupportedError

        raise TaskNotSupportedError(
            model_id,
            "finetune_sam_decoder",
            model.entry.task,
            hint="SAM decoder fine-tune requires a SAM-style segmentation model.",
        )
    repo = model.entry.hf_repo_id
    if not repo:
        raise ValueError(f"WEIGHTS_MISSING: {model_id} has no hf_repo_id to fine-tune from")
    proc = SamProcessor.from_pretrained(repo)
    sam = SamModel.from_pretrained(repo).to(device)
    return sam, proc, repo


def _as_mask(mask: Any) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        from PIL import Image

        mask = np.array(Image.open(mask).convert("L"))
    arr = np.asarray(mask)
    return (arr > 0).astype(np.float32)


def _freeze_encoders(sam) -> int:
    """Freeze vision + prompt encoders; leave the mask decoder trainable. Returns trainable params."""
    for p in sam.parameters():
        p.requires_grad_(False)
    trainable = 0
    for p in sam.mask_decoder.parameters():
        p.requires_grad_(True)
        trainable += p.numel()
    return trainable


def _dice_bce_loss(logits, target):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum() + 1.0
    den = prob.sum() + target.sum() + 1.0
    dice = 1.0 - num / den
    return bce + dice


def finetune_sam_decoder(
    model_id: str,
    samples: list[dict],
    *,
    epochs: int = 12,
    lr: float = 1e-4,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune only the SAM mask decoder on box-prompted masks. Frozen encoders."""
    import torch

    if not samples:
        raise ValueError("DATASET_INVALID: need >=1 {image, box, mask} sample")
    sam, proc, repo = _load_sam(model_id, device)
    trainable = _freeze_encoders(sam)

    # Pre-encode the (frozen) inputs once; only the decoder sees gradients.
    batches = []
    for s in samples:
        from PIL import Image

        img = s["image"]
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        inp = proc(img, input_boxes=[[list(s["box"])]], return_tensors="pt").to(device)
        gt = _as_mask(s["mask"])
        batches.append((inp, torch.tensor(gt, dtype=torch.float32, device=device)))

    opt = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=1e-4)
    sam.train()
    last = 0.0
    for _ in range(max(1, epochs)):
        for inp, gt in batches:
            opt.zero_grad()
            out = sam(**inp, multimask_output=False)
            logits = out.pred_masks[0, 0, 0]  # (H, W) low-res mask logits
            target = torch.nn.functional.interpolate(
                gt[None, None], size=logits.shape[-2:], mode="nearest"
            )[0, 0]
            loss = _dice_bce_loss(logits, target)
            loss.backward()
            opt.step()
            last = float(loss.detach())

    out_dir = Path(output_dir or Path.cwd() / "sam_decoder_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_id.replace('/', '_')}_decoder.pt"
    torch.save(
        {
            "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
            "model_id": model_id,
            "hf_repo_id": repo,
            "trainable_params": trainable,
            "artifact": "mask_decoder_only",
        },
        ckpt,
    )
    return {
        "status": "OK",
        "method": "sam_decoder_finetune",
        "checkpoint": str(ckpt),
        "trainable_params": trainable,
        "final_loss": round(last, 4),
        "frozen": ["vision_encoder", "prompt_encoder"],
        "model_id": model_id,
    }


class SamDecoderModel:
    """A reloaded frozen SAM encoder + fine-tuned mask decoder."""

    def __init__(self, sam, proc, model_id: str, device: str) -> None:
        self._sam = sam
        self._proc = proc
        self.model_id = model_id
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, *, device: str = "cpu"
    ) -> SamDecoderModel:
        import torch
        from transformers import SamModel, SamProcessor

        blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        proc = SamProcessor.from_pretrained(blob["hf_repo_id"])
        sam = SamModel.from_pretrained(blob["hf_repo_id"]).to(device)
        sam.mask_decoder.load_state_dict(blob["mask_decoder_state_dict"])
        sam.eval()
        return cls(sam, proc, blob["model_id"], device)

    def segment(self, image: Any, box: list[float]) -> np.ndarray:
        """Predict a 0/1 mask for a box prompt using the fine-tuned decoder."""
        import torch
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        inp = self._proc(image, input_boxes=[[list(box)]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._sam(**inp, multimask_output=False)
        logits = out.pred_masks[0, 0, 0]
        return (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)


__all__ = ["SamDecoderModel", "finetune_sam_decoder"]

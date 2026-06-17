# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Embedding head fine-tuning (linear probe) for frozen-backbone embedders.

Foundation embedders (DINOv2, CLIP, SigLIP, …) are not "trained" end-to-end in
VisionServeX — but the standard, cheap, legal transfer-learning path *is*
supported here: freeze the backbone, extract embeddings, and fit a small linear
classification head on top. This produces a real, reloadable artifact and a full
lifecycle:

    finetune_embedding_head(...)  ->  head checkpoint + class map
    EmbeddingHeadModel.from_checkpoint(...)  ->  reload
        .classify(img)     # embed-via-backbone -> head -> softmax top-k
        .embed(img)        # backbone embedding (unchanged dimension)
        .similarity(a, b)  # cosine over backbone embeddings

It never modifies or redistributes the backbone weights; the head is the only
trained, saved artifact. Only models whose registered task is ``embed`` are
eligible — calling it on another task raises ``TaskNotSupportedError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from visionservex.core.model import VisionModel
from visionservex.core.results import ClassificationResult

_EMBED_TASKS = frozenset({"embed", "embedding"})


def _resolve_imagefolder(dataset: str | Path) -> Path:
    root = Path(dataset)
    if (root / "train").is_dir():
        root = root / "train"
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(
            f"DATASET_INVALID: ImageFolder at {root} needs >=2 class subfolders, found {len(classes)}"
        )
    return root


def _embed_array(model: VisionModel, image_path: str | Path) -> np.ndarray:
    res = model.embed(str(image_path))
    emb = getattr(res, "embedding", None)
    if emb is None:
        raise RuntimeError(f"{model.entry.id}: embed() returned no .embedding")
    return np.asarray(emb, dtype=np.float32).ravel()


def finetune_embedding_head(
    model_id: str,
    dataset: str | Path,
    *,
    epochs: int = 40,
    lr: float = 1e-2,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit a linear classification head on a frozen embedder. Returns a result dict.

    The backbone stays frozen; only the ``nn.Linear`` head is trained and saved.
    """
    import torch

    model = VisionModel(model_id, device=device)
    if model.entry.task not in _EMBED_TASKS:
        from visionservex.exceptions import TaskNotSupportedError

        raise TaskNotSupportedError(
            model_id,
            "finetune_embedding_head",
            model.entry.task,
            hint="embedding head fine-tune requires an `embed`-task model.",
        )

    root = _resolve_imagefolder(dataset)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    feats, labels = [], []
    for ci, cname in enumerate(classes):
        for img in sorted((root / cname).glob("*")):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                feats.append(_embed_array(model, img))
                labels.append(ci)
    if not feats:
        raise ValueError(f"DATASET_INVALID: no images under {root}")

    x = torch.tensor(np.stack(feats), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    dim = x.shape[1]
    head = torch.nn.Linear(dim, len(classes))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    head.train()
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        loss = loss_fn(head(x), y)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        train_acc = float((head(x).argmax(1) == y).float().mean())

    out_dir = Path(output_dir or Path.cwd() / "embedding_head_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{model_id.replace('/', '_')}_head.pt"
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "class_names": classes,
            "embed_dim": dim,
            "model_id": model_id,
            "normalized": True,
            "head_type": "linear_probe",
        },
        ckpt,
    )
    return {
        "status": "OK",
        "method": "head_train",
        "checkpoint": str(ckpt),
        "class_names": classes,
        "embed_dim": dim,
        "train_acc": round(train_acc, 4),
        "model_id": model_id,
    }


class EmbeddingHeadModel:
    """A reloaded frozen embedder + trained linear head."""

    def __init__(self, model_id: str, head: Any, classes: list[str], dim: int, device: str) -> None:
        self.model_id = model_id
        self._embedder = VisionModel(model_id, device=device)
        self._head = head
        self.class_names = classes
        self.embed_dim = dim
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, *, device: str = "cpu"
    ) -> EmbeddingHeadModel:
        import torch

        blob = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        classes = blob["class_names"]
        dim = int(blob["embed_dim"])
        head = torch.nn.Linear(dim, len(classes))
        head.load_state_dict(blob["head_state_dict"])
        head.eval()
        return cls(blob["model_id"], head, classes, dim, device)

    def embed(self, image: Any):
        """Backbone embedding (unchanged by the head fine-tune)."""
        return self._embedder.embed(image)

    def similarity(self, a: Any, b: Any) -> float:
        return self._embedder.similarity(a, b)

    def classify(self, image: Any, *, top_k: int = 5) -> ClassificationResult:
        import torch

        emb = (
            _embed_array(self._embedder, image)
            if not hasattr(image, "shape")
            else np.asarray(image)
        )
        with torch.no_grad():
            logits = self._head(torch.tensor(emb, dtype=torch.float32).unsqueeze(0))
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
        ranked = sorted(zip(self.class_names, probs, strict=False), key=lambda kv: -kv[1])[
            : max(1, top_k)
        ]
        return ClassificationResult(
            model_id=self.model_id,
            task="classify",
            top_k=[(c, float(p)) for c, p in ranked],
        )


__all__ = ["EmbeddingHeadModel", "finetune_embedding_head"]

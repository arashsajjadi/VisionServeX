# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""DINOv2 / SigLIP2 feature extraction engine via HF Transformers.

Wraps:
- facebook/dinov2-* (image-only embeddings)
- google/siglip2-* (image+text embeddings for retrieval)

Output: ``EmbeddingResult`` with the pooled image embedding as a numpy array.

DO NOT use this engine for detection AP or classification top-k. Use
embeddings only for retrieval, deduplication, dataset reports, similarity.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult
from visionservex.engines._stub import StubEngine
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)


class DINOv2Engine(StubEngine):
    """HF AutoModel-based feature extractor for DINOv2 and SigLIP2 (image side)."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_dinov2"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        self._embedding_dim: int = 0

    def _real_load(self, *, device: str, precision: str) -> None:
        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id",
                install_hint=self._install_hint(),
            )
        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import AutoImageProcessor, AutoModel  # type: ignore

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        _log.info("loading %s from %s on %s", self.entry.id, self.entry.hf_repo_id, device)
        self._processor = AutoImageProcessor.from_pretrained(self.entry.hf_repo_id, use_fast=True)
        self._model = AutoModel.from_pretrained(self.entry.hf_repo_id, **kwargs)
        self._model.to(device)
        self._model.eval()
        self._torch = torch
        try:
            self._embedding_dim = int(getattr(self._model.config, "hidden_size", 0))
        except Exception:
            self._embedding_dim = 0

    def unload(self) -> None:
        if self._model is not None:
            with contextlib.suppress(Exception):
                self._model.cpu()
            del self._model, self._processor
            self._model = None
            self._processor = None
        super().unload()

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, **kwargs)

        from visionservex.core.embedding_results import EmbeddingResult

        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype

        inputs = self._processor(images=[image], return_tensors="pt")
        inputs_dev: dict[str, Any] = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_dev[k] = v

        with self._torch.no_grad():
            # SigLIP / SigLIP2: the full SiglipModel requires both pixel_values AND
            # input_ids (text tokens). For image-only embedding we route through the
            # vision sub-model directly, which accepts only pixel_values.
            _vision_submodel = getattr(self._model, "vision_model", None)
            _pixel_only = {"pixel_values"} >= set(inputs_dev.keys())
            if _vision_submodel is not None and _pixel_only:
                out = _vision_submodel(pixel_values=inputs_dev.get("pixel_values"))
            else:
                out = self._model(**inputs_dev)

        # DINOv2 / SigLIP2 expose pooler_output or last_hidden_state.
        embedding = None
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            embedding = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # CLS token is at index 0 for ViT-style models
            embedding = out.last_hidden_state[:, 0, :]
        elif hasattr(out, "image_embeds") and out.image_embeds is not None:
            embedding = out.image_embeds
        else:
            # Last resort: average pool
            embedding = out[0].mean(dim=1) if isinstance(out, tuple) else None

        if embedding is None:
            raise RuntimeError(
                f"could not extract embedding from {self.entry.id} output keys: {list(out.keys()) if hasattr(out, 'keys') else type(out)}"
            )

        # L2-normalize
        embedding = self._torch.nn.functional.normalize(embedding, p=2, dim=-1)
        emb_np = embedding.cpu().float().numpy()[0]

        return EmbeddingResult(
            kind="embedding",
            model_id=self.entry.id,
            task=self.entry.task,
            embedding=emb_np,
            embedding_dim=int(emb_np.shape[0]),
            normalized=True,
            backend=self.backend_label,
            device=self.device,
            precision=self.precision,
        )

    def encode_text(self, prompts: Sequence[str]) -> Any:
        """Encode text prompts (SigLIP / CLIP-like models only). Returns numpy array."""
        if not self._real_ready:
            raise RuntimeError("model not loaded")
        # Only some models support text encoding
        family = self.entry.family
        if "siglip" not in family.lower() and "clip" not in family.lower():
            raise NotImplementedError(
                f"text encoding not supported for {family!r}. "
                "Use a SigLIP or CLIP model for text-image retrieval."
            )
        from transformers import AutoTokenizer  # type: ignore

        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.entry.hf_repo_id)

        inputs = self._tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True)
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with self._torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
        text_features = self._torch.nn.functional.normalize(text_features, p=2, dim=-1)
        return text_features.cpu().float().numpy()

    def infer(self, preprocessed: Any, **kwargs: Any) -> Any:
        return preprocessed

    def postprocess(self, raw: Any, *, image: Any, **kwargs: Any) -> BaseResult:
        return self._mock.postprocess(raw, image=image, **kwargs)


def _factory(entry: ModelEntry) -> DINOv2Engine:
    return DINOv2Engine(entry)


register_engine("dinov2", _factory)
register_engine("siglip2", _factory)  # same engine handles SigLIP2 image side


__all__ = ["DINOv2Engine"]

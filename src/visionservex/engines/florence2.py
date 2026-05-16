# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Florence-2 — multi-task vision-language model engine.

Real implementation via Hugging Face Transformers with ``trust_remote_code=True``:
``AutoProcessor`` + ``AutoModelForCausalLM``.

Reference:
- https://github.com/microsoft/Florence-2
- https://huggingface.co/microsoft/Florence-2-base
- https://huggingface.co/microsoft/Florence-2-large

Supported tasks (via the ``task`` kwarg or auto-mapping from prompt tokens):
- caption                       → "<CAPTION>"
- detailed_caption              → "<DETAILED_CAPTION>"
- more_detailed_caption         → "<MORE_DETAILED_CAPTION>"
- object_detection              → "<OD>"
- dense_region_caption          → "<DENSE_REGION_CAPTION>"
- phrase_grounding              → "<CAPTION_TO_PHRASE_GROUNDING>"
- ocr                           → "<OCR>"
- region_ocr / ocr_with_region  → "<OCR_WITH_REGION>"

Output: a single ``OpenVocabularyResult`` whose ``detections`` list is populated
for box-producing tasks; text-only tasks (caption, ocr) put the generated text
in ``metadata["text"]`` and leave ``detections`` empty.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PIL import Image

from visionservex.core.results import (
    BaseResult,
    Box,
    Detection,
    OpenVocabularyResult,
)
from visionservex.engines._stub import StubEngine, assert_modules
from visionservex.engines.base import MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Task → prompt token table.
_TASK_TOKEN: dict[str, str] = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
    "object_detection": "<OD>",
    "dense_caption": "<DENSE_REGION_CAPTION>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "ocr": "<OCR>",
    "region_ocr": "<OCR_WITH_REGION>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}


def _apply_florence2_compat_shims(hf_repo_id: str) -> None:
    """Apply all transformers >= 5.x compat patches for Florence-2.

    Patches are applied directly to the class objects in sys.modules so they
    survive the full from_pretrained lifecycle.  Each patch is idempotent.
    """

    # Pre-import the dynamic modules so their classes land in sys.modules.
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # type: ignore

        lang_cfg_cls = get_class_from_dynamic_module(
            "configuration_florence2.Florence2LanguageConfig", hf_repo_id, trust_remote_code=True
        )
        # Shim 1: forced_bos_token_id was removed from PretrainedConfig in transformers 5.x.
        # The Florence-2 custom config reads it in __init__; we add it as a simple fallback.
        if "forced_bos_token_id" not in lang_cfg_cls.__dict__:
            lang_cfg_cls.forced_bos_token_id = None  # type: ignore[attr-defined]
    except Exception as exc:
        _log.debug("Florence-2 config shim skipped: %s", exc)

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module  # type: ignore

        model_cls = get_class_from_dynamic_module(
            "modeling_florence2.Florence2ForConditionalGeneration",
            hf_repo_id,
            trust_remote_code=True,
        )
        # Shim 2: _supports_sdpa / _supports_flash_attn_2 are @property on
        # Florence2PreTrainedModel (parent class). They delegate to self.language_model
        # which is not yet set during from_pretrained.__init__, raising AttributeError.
        # Assigning plain False/True to the subclass __dict__ shadows the properties.
        model_cls._supports_sdpa = False  # type: ignore[attr-defined]
        model_cls._supports_flash_attn_2 = False  # type: ignore[attr-defined]
    except Exception as exc:
        _log.debug("Florence-2 model shim skipped: %s", exc)


def _apply_florence2_config_shim() -> bool:
    """Patch Florence2LanguageConfig for transformers >= 5.0 compatibility.

    In transformers 5.x, ``forced_bos_token_id`` was removed from
    ``PretrainedConfig``. Florence-2's custom ``configuration_florence2.py``
    still accesses ``self.forced_bos_token_id`` in its ``__init__``.
    We add a fallback property that returns ``None``.
    """

    try:
        # The config class lives in transformers_modules (cached custom code).
        cfg_mod = None
        for mod_name in list(__import__("sys").modules.keys()):
            if "configuration_florence2" in mod_name:
                cfg_mod = __import__("sys").modules[mod_name]
                break
        if cfg_mod is None:
            # Module not yet imported — cannot patch; the engine will retry.
            return False
        cls = getattr(cfg_mod, "Florence2LanguageConfig", None)
        if cls is None:
            return False
        if hasattr(cls, "forced_bos_token_id"):
            return False
        cls.forced_bos_token_id = property(lambda self: None)
        return True
    except Exception:
        return False


def _apply_florence2_tokenizer_shim() -> bool:
    """Patch TokenizersBackend.additional_special_tokens if missing.

    Florence-2's custom ``processing_florence2.py`` accesses
    ``tokenizer.additional_special_tokens`` but the fast ``TokenizersBackend``
    (tokenizers >= 0.21 / Python 3.13) doesn't expose this attribute.

    This shim adds it as a property that returns the list of special-token
    content strings from ``added_tokens_decoder``, matching what the slow
    tokenizer would return.

    Returns True if the shim was applied, False if it was already present.
    """
    try:
        from transformers.tokenization_utils_tokenizers import (  # type: ignore
            TokenizersBackend,
        )

        if hasattr(TokenizersBackend, "additional_special_tokens"):
            return False

        def _additional_special_tokens(self: Any) -> list[str]:
            # added_tokens_decoder: dict[int, AddedToken]
            return [
                tok_obj.content
                for tok_obj in self.added_tokens_decoder.values()
                if getattr(tok_obj, "special", False)
            ]

        TokenizersBackend.additional_special_tokens = property(  # type: ignore[attr-defined]
            _additional_special_tokens
        )
        return True
    except ImportError:
        return False  # Slow tokenizer path — shim not needed


def parse_florence2_generation(
    parsed: Any, *, task_token: str, image_size: tuple[int, int]
) -> tuple[list[Detection], dict[str, Any]]:
    """Convert Florence-2's parsed generation output into (detections, extra_metadata).

    The processor's ``post_process_generation`` returns a dict keyed by task token.
    For box-producing tasks the value is a dict with ``bboxes`` and ``labels``.
    For text-only tasks the value is a string.
    Phrase grounding and dense region caption produce both boxes and per-box labels.
    """
    detections: list[Detection] = []
    extra: dict[str, Any] = {"task_token": task_token}

    if not isinstance(parsed, dict):
        extra["raw"] = str(parsed)[:500]
        return detections, extra

    payload = parsed.get(task_token, parsed)

    # Text-only tasks
    if isinstance(payload, str):
        extra["text"] = payload
        return detections, extra

    if not isinstance(payload, dict):
        extra["raw"] = str(payload)[:500]
        return detections, extra

    # Box-producing tasks: keys we recognise across upstream versions
    bboxes = payload.get("bboxes") or payload.get("boxes") or payload.get("quad_boxes") or []
    labels = payload.get("labels") or payload.get("bboxes_labels") or []

    for idx, box in enumerate(bboxes):
        # Polygons (quad_boxes) → convert to axis-aligned bbox
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            if len(box) == 4:
                x1, y1, x2, y2 = (float(v) for v in box)
            else:
                xs = [float(box[i]) for i in range(0, len(box), 2)]
                ys = [float(box[i]) for i in range(1, len(box), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue

        label = str(labels[idx]) if idx < len(labels) else ""
        detections.append(
            Detection(
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                score=1.0,  # Florence-2 generative output has no per-box score
                label=label,
                class_id=None,
            )
        )

    if not detections and not labels:
        # Some upstream versions return only flat strings
        extra["raw"] = str(payload)[:500]
    return detections, extra


class Florence2Engine(StubEngine):
    """Florence-2 multi-task VLM engine."""

    real_install_extra = "hf"
    real_modules = ("transformers", "torch")
    backend_label = "huggingface_florence2"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    # ----- lifecycle -----

    def _real_load(self, *, device: str, precision: str) -> None:
        assert_modules(self.real_modules, install_hint=self._install_hint())

        # Version guard: Florence-2's custom code uses several transformers 4.x
        # internal APIs that were removed in transformers 5.x:
        #   - PretrainedConfig.forced_bos_token_id (config)
        #   - TokenizersBackend.additional_special_tokens (processor)
        #   - EncoderDecoderCache subscript protocol (generation)
        #   - meta-tensor-safe torch.linspace in DaViT init
        # All four issues are present in transformers 5.x simultaneously.
        try:
            import transformers as _tr  # type: ignore

            _tr_major = int(_tr.__version__.split(".")[0])
            if _tr_major >= 5:
                raise MissingDependencyError(
                    f"Florence-2 is incompatible with transformers {_tr.__version__}. "
                    "The model's custom code uses four internal APIs removed in transformers 5.x. "
                    "Install a compatible version: pip install 'transformers>=4.40,<5.0'",
                    install_hint=(
                        "pip install 'transformers>=4.40,<5.0'  "
                        "(or: pip install 'visionservex[hf]' after pinning transformers)"
                    ),
                )
        except MissingDependencyError:
            raise
        except Exception:
            pass  # If version parsing fails, proceed and let the real error surface.

        if not self.entry.hf_repo_id:
            raise MissingDependencyError(
                f"model {self.entry.id!r} has no hf_repo_id; cannot load from Hugging Face",
                install_hint=self._install_hint(),
            )

        from visionservex.runtime.downloads import download

        download(self.entry)

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        # Compatibility shims for Florence-2 on transformers >= 5.x / tokenizers >= 0.21.
        # 1. TokenizersBackend.additional_special_tokens: removed in tokenizers 0.21+.
        # 2. Florence2LanguageConfig.forced_bos_token_id: removed in transformers 5.x.
        _apply_florence2_tokenizer_shim()

        # Pre-load the custom modules so we can patch them before from_pretrained
        # triggers __init__. Two compatibility issues with transformers >= 5.x:
        # 1. Florence2LanguageConfig.__init__ reads self.forced_bos_token_id (removed in 5.x).
        # 2. Florence2ForConditionalGeneration._supports_sdpa is a @property that delegates
        #    to self.language_model._supports_sdpa — it raises AttributeError during
        #    from_pretrained when language_model is not yet assigned to the instance.
        _apply_florence2_compat_shims(self.entry.hf_repo_id)

        torch_dtype = torch.float32
        if precision == "fp16" and device != "cpu":
            torch_dtype = torch.float16
        elif precision == "bf16" and device != "cpu":
            torch_dtype = torch.bfloat16

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch_dtype is not torch.float32:
            kwargs["torch_dtype"] = torch_dtype

        self._processor = AutoProcessor.from_pretrained(
            self.entry.hf_repo_id, trust_remote_code=True
        )
        # Shim 3: In PyTorch >= 2.7, torch.linspace may return meta tensors when called
        # during DaViT vision-encoder initialization under device_map logic.
        # We temporarily override torch.linspace to force concrete CPU tensors.
        _orig_linspace = torch.linspace

        def _safe_linspace(*args: Any, **kw: Any) -> Any:
            result = _orig_linspace(*args, **kw)
            return result.to("cpu") if getattr(result, "is_meta", False) else result

        torch.linspace = _safe_linspace  # type: ignore[assignment]
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.entry.hf_repo_id, **kwargs)
        finally:
            torch.linspace = _orig_linspace  # type: ignore[assignment]
        self._model.to(device)
        self._model.eval()
        self._torch = torch

    def unload(self) -> None:
        if self._real_ready and self._model is not None:
            try:
                del self._model
                del self._processor
            except Exception:
                pass
            self._model = None
            self._processor = None
            self._real_ready = False
        super().unload()

    # ----- inference -----

    def predict(
        self,
        image: Image.Image,
        *,
        prompts: Sequence[str] | None = None,
        prompt: str | None = None,
        task: str | None = None,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, prompts=prompts, prompt=prompt, task=task, **kwargs)

        # Resolve task → token.
        task_key = (task or "caption").lower().strip()
        task_token = _TASK_TOKEN.get(task_key)
        if task_token is None:
            raise ValueError(
                f"Florence-2 task {task!r} unsupported. Choose from: {sorted(_TASK_TOKEN.keys())}"
            )

        # For phrase grounding, the user's prompt is appended after the token.
        user_text = ""
        if prompt:
            user_text = str(prompt).strip()
        elif prompts:
            user_text = ", ".join(str(p).strip() for p in prompts).strip()

        prompt_text = task_token
        if task_token == "<CAPTION_TO_PHRASE_GROUNDING>" and user_text:
            prompt_text = f"{task_token} {user_text}"

        inputs = self._processor(text=prompt_text, images=image, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        model_dtype = next(self._model.parameters()).dtype
        inputs_on_device = {}
        for k, v in inputs.items():
            v = v.to(device=model_device)
            if v.is_floating_point():
                v = v.to(dtype=model_dtype)
            inputs_on_device[k] = v

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs_on_device.get("input_ids"),
                pixel_values=inputs_on_device.get("pixel_values"),
                max_new_tokens=max_new_tokens,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        try:
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task_token,
                image_size=image.size,
            )
        except Exception as exc:
            _log.warning("Florence-2 post_process_generation failed: %s", exc)
            parsed = {"raw": generated_text}

        detections, extra = parse_florence2_generation(
            parsed, task_token=task_token, image_size=image.size
        )
        prompts_list = [user_text] if user_text else []

        metadata: dict[str, Any] = {
            "backend": self.backend_label,
            "precision": self.precision,
            "task": task_key,
            "task_token": task_token,
        }
        metadata.update(extra)

        return OpenVocabularyResult(
            kind="open_vocab",
            model_id=self.entry.id,
            task=task_key,
            image_size=image.size,
            device=self.device,
            prompts=prompts_list,
            detections=detections,
            metadata=metadata,
        )


def _factory(entry: ModelEntry) -> Florence2Engine:
    return Florence2Engine(entry)


register_engine("florence2", _factory)

__all__ = ["Florence2Engine", "parse_florence2_generation"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Torchvision image-classification engine (classic permissive backbones).

Wires ``torchvision.models`` (BSD-3-Clause; ImageNet-1K pretrained weights
distributed by PyTorch under the same permissive license) into VisionServeX as a
commercial-safe classification family — AlexNet, ResNet, ResNeXt, Wide-ResNet,
DenseNet, MobileNet, EfficientNet, ConvNeXt.

Full lifecycle (v3.15.0):
    pretrained inference -> fine-tune (ImageFolder) -> checkpoint -> reload ->
    predict -> ONNX export

No Ultralytics / AGPL / GPL runtime. Pretrained weights are pulled on demand by
torchvision (never bundled by VisionServeX).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from visionservex.core.results import BaseResult, ClassificationResult
from visionservex.engines._stub import StubEngine, assert_modules
from visionservex.engines.base import EngineError, MissingDependencyError
from visionservex.engines.registry import register_engine
from visionservex.registry import ModelEntry
from visionservex.utils.logging import get_logger

_log = get_logger(__name__)

# Curated, tested torchvision classifier ids -> torchvision arch name. Only these
# are exposed (no arbitrary torchvision model is runnable by default).
_TV_ARCHS: dict[str, str] = {
    "torchvision-alexnet": "alexnet",
    "torchvision-resnet18": "resnet18",
    "torchvision-resnet34": "resnet34",
    "torchvision-resnet50": "resnet50",
    "torchvision-resnet101": "resnet101",
    "torchvision-resnet152": "resnet152",
    "torchvision-wide-resnet50-2": "wide_resnet50_2",
    "torchvision-resnext50-32x4d": "resnext50_32x4d",
    "torchvision-densenet121": "densenet121",
    "torchvision-mobilenet-v2": "mobilenet_v2",
    "torchvision-mobilenet-v3-large": "mobilenet_v3_large",
    "torchvision-efficientnet-b0": "efficientnet_b0",
    "torchvision-convnext-tiny": "convnext_tiny",
}

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _replace_last_linear(model: Any, nc: int) -> Any:
    """Replace the model's final ``nn.Linear`` head with a fresh ``nc``-way one.

    Generic across torchvision classifier archs (resnet `.fc`, alexnet/mobilenet/
    efficientnet/convnext `.classifier[-1]`, densenet `.classifier`).
    """
    import torch.nn as nn

    last_name, last_mod = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_name, last_mod = name, mod
    if last_name is None or last_mod is None:
        raise EngineError("no nn.Linear classification head found to fine-tune")
    new = nn.Linear(last_mod.in_features, nc)
    parent = model
    *parents, attr = last_name.split(".")
    for p in parents:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if attr.isdigit():
        parent[int(attr)] = new
    else:
        setattr(parent, attr, new)
    return new


def _eval_transform(arch: str):
    """Eval/predict preprocess for *arch* (the default weights' own transform)."""
    from torchvision.models import get_model_weights

    return get_model_weights(arch).DEFAULT.transforms()


def _train_transform(imgsz: int):
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(imgsz),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TorchvisionClassifyEngine(StubEngine):
    """Classic torchvision image classifier (pretrained + fine-tune + export)."""

    real_install_extra = "torchvision"
    real_modules = ("torch", "torchvision")
    backend_label = "torchvision"

    def __init__(self, entry: ModelEntry) -> None:
        super().__init__(entry)
        self._model: Any = None
        self._preprocess: Any = None
        self._categories: list[str] = []
        self._torch: Any = None
        # When set (via load_checkpoint), a fine-tuned checkpoint overrides the
        # ImageNet pretrained weights as the sole weight source.
        self._checkpoint_override: Path | None = None

    def _arch(self) -> str:
        arch = _TV_ARCHS.get(self.entry.id)
        if arch is None:
            raise MissingDependencyError(
                f"{self.entry.id!r} is not a known torchvision classifier id "
                f"(supported: {sorted(_TV_ARCHS)})",
                install_hint="check `visionservex list-models --family torchvision-classify`",
            )
        return arch

    # ------ lifecycle ------

    def _real_load(self, *, device: str, precision: str) -> None:
        assert_modules(self.real_modules, install_hint=self._install_hint())
        import torch  # type: ignore
        from torchvision.models import get_model, get_model_weights  # type: ignore

        arch = self._arch()
        if self._checkpoint_override is not None:
            ckpt_path = self._checkpoint_override
            if not Path(ckpt_path).is_file():
                raise MissingDependencyError(
                    f"trained checkpoint not found: {ckpt_path}",
                    install_hint="pass a best.pt/last.pt produced by train()",
                )
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            classes = list(ckpt.get("classes") or [])
            nc = int(ckpt.get("nc") or len(classes) or 1000)
            model = get_model(arch, weights=None, num_classes=1000)
            if nc != 1000:
                _replace_last_linear(model, nc)
            model.load_state_dict(ckpt["state_dict"])
            self._categories = classes or [str(i) for i in range(nc)]
            _log.info("loaded fine-tuned %s from %s (%d classes)", arch, ckpt_path, nc)
        else:
            weights = get_model_weights(arch).DEFAULT
            model = get_model(arch, weights=weights)
            self._categories = list(weights.meta.get("categories") or [])
            _log.info("loaded torchvision %s (ImageNet-1K pretrained)", arch)

        self._preprocess = _eval_transform(arch)
        model.eval().to(device)
        self._model = model
        self._torch = torch

    def unload(self) -> None:
        import contextlib

        if self._model is not None:
            with contextlib.suppress(Exception):
                del self._model
            self._model = None
        super().unload()

    # ------ inference ------

    def predict(self, image: Image.Image, *, top_k: int = 5, **kwargs: Any) -> BaseResult:
        if not self._real_ready:
            return super().predict(image, **kwargs)

        dev = next(self._model.parameters()).device
        x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(dev)
        with self._torch.no_grad():
            logits = self._model(x)[0].float()
        probs = self._torch.nn.functional.softmax(logits, dim=-1)
        k = min(top_k, probs.numel())
        topk_probs, topk_idx = probs.topk(k)
        pairs = [
            (
                self._categories[int(i)] if int(i) < len(self._categories) else str(int(i)),
                float(p),
            )
            for i, p in zip(topk_idx.cpu(), topk_probs.cpu(), strict=False)
        ]
        return ClassificationResult(
            kind="classification",
            model_id=self.entry.id,
            task="classify",
            image_size=image.size,
            device=self.device,
            top_k=pairs,
            metadata={"backend": self.backend_label, "precision": self.precision},
        )

    # ------ training / fine-tuning ------

    def train(
        self,
        dataset: str | Path,
        *,
        epochs: int = 5,
        batch: int = 16,
        imgsz: int = 224,
        lr: float = 1e-3,
        device: str | None = None,
        workers: int = 0,
        project: str | None = None,
        name: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Fine-tune this classifier on an ImageFolder dataset.

        ``dataset`` is an ImageFolder root (``<root>/<class>/*.jpg``) or a dir with
        a ``train/`` subfolder. Returns the normalized training-result contract;
        the checkpoint records the dataset's class names so reload rebuilds the
        right head.
        """
        assert_modules(self.real_modules, install_hint=self._install_hint())
        import torch  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
        from torchvision.datasets import ImageFolder  # type: ignore
        from torchvision.models import get_model, get_model_weights  # type: ignore

        arch = self._arch()
        root = _resolve_imagefolder(dataset)
        dev = _resolve_device(device)

        ds = ImageFolder(str(root), transform=_train_transform(imgsz))
        classes = list(ds.classes)
        nc = len(classes)
        if nc < 2:
            raise EngineError(
                f"DATASET_INVALID: ImageFolder at {root} needs >=2 class subfolders, found {nc}"
            )
        loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers)

        weights = get_model_weights(arch).DEFAULT
        model = get_model(arch, weights=weights)  # ImageNet-pretrained backbone
        _replace_last_linear(model, nc)
        model.train().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        crit = torch.nn.CrossEntropyLoss()

        _log.info("fine-tuning %s on %s (%d classes) for %d epochs", arch, root, nc, epochs)
        t0 = time.time()
        epoch_losses: list[float] = []
        for _epoch in range(epochs):
            running, n = 0.0, 0
            for imgs, labels in loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                opt.zero_grad()
                loss = crit(model(imgs), labels)
                loss.backward()
                opt.step()
                running += float(loss.item()) * imgs.size(0)
                n += imgs.size(0)
            epoch_losses.append(running / max(1, n))
        hours = (time.time() - t0) / 3600.0

        save_dir = Path(project or "runs/classify") / (name or self.entry.id)
        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "arch": arch,
            "state_dict": model.state_dict(),
            "classes": classes,
            "nc": nc,
            "imgsz": imgsz,
            "model_family": "torchvision-classify",
        }
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"
        torch.save(ckpt, best)
        torch.save(ckpt, last)

        return {
            "status": "ok",
            "model_id": self.entry.id,
            "family": "torchvision-classify",
            "variant": arch,
            "dataset_format": "imagefolder",
            "dataset": str(root),
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "save_dir": str(save_dir),
            "metrics": {
                "final_loss": epoch_losses[-1] if epoch_losses else None,
                "epoch_losses": epoch_losses,
                "epochs_completed": len(epoch_losses),
                "num_classes": nc,
                "training_time_hours": round(hours, 4),
            },
            "artifacts": {"weights_dir": str(weights_dir), "classes": classes},
        }

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        precision: str = "fp32",
    ) -> TorchvisionClassifyEngine:
        """Load a fine-tuned classifier checkpoint for inference (no base fallback)."""
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise MissingDependencyError(
                f"trained checkpoint not found: {ckpt}",
                install_hint="train one first: VisionModel('torchvision-resnet18').train(imagefolder)",
            )
        if self._real_ready or self._model is not None:
            self.unload()
        self._real_ready = False
        self._checkpoint_override = ckpt
        self.load(device=device or self.device or "cpu", precision=precision)
        return self

    def export(self, format: str, output_path: str | Path) -> Path:
        """Export the loaded classifier to ONNX."""
        if format.lower() != "onnx":
            raise EngineError(
                f"EXPORT_UNSUPPORTED: torchvision classifier export supports 'onnx', not {format!r}"
            )
        if not self._real_ready:
            self.load(device=self.device or "cpu", precision="fp32")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        dev = next(self._model.parameters()).device
        dummy = self._torch.randn(1, 3, 224, 224, device=dev)
        self._torch.onnx.export(
            self._model,
            dummy,
            str(out),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )
        return out


def _resolve_device(device: str | None) -> str:
    if device in (None, "auto", ""):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device)


def _resolve_imagefolder(dataset: str | Path) -> Path:
    """Resolve an ImageFolder root: a dir of class subdirs, or one with train/."""
    p = Path(dataset)
    if not p.exists():
        raise MissingDependencyError(
            f"dataset path not found: {p}",
            install_hint="pass an ImageFolder root (<root>/<class>/*.jpg) or a dir with train/",
        )
    if (p / "train").is_dir():
        return p / "train"
    return p


def _factory(entry: ModelEntry) -> TorchvisionClassifyEngine:
    return TorchvisionClassifyEngine(entry)


register_engine("torchvision_classify", _factory)

__all__ = ["TorchvisionClassifyEngine"]

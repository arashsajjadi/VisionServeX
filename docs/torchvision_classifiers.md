# Torchvision Classifiers (v3.15.0)

VisionServeX v3.15.0 adds a **commercial-safe classic classifier family** backed by
[`torchvision.models`](https://github.com/pytorch/vision). torchvision code **and**
its ImageNet-1K pretrained weights are distributed by PyTorch under **BSD-3-Clause**
— permissive and commercial-safe. **No Ultralytics / AGPL / GPL** runtime. Weights
are pulled on demand by torchvision and **never bundled** by VisionServeX.

> The ImageNet *dataset* terms are research-only, but the trained weights that
> torchvision redistributes are BSD-3-Clause (the standard commercial position).
> The policy row records `dataset_risk=imagenet_weights_permissive`.

## Model IDs (full lifecycle: pretrained → fine-tune → reload → predict → export)

| Model ID | Arch | Params class |
|---|---|---|
| `torchvision-alexnet` | AlexNet | tiny/legacy |
| `torchvision-resnet18` | ResNet-18 | light |
| `torchvision-resnet34` | ResNet-34 | light |
| `torchvision-resnet50` | ResNet-50 | standard backbone |
| `torchvision-resnet101` | ResNet-101 | larger |
| `torchvision-resnet152` | ResNet-152 | larger |
| `torchvision-wide-resnet50-2` | Wide ResNet-50-2 | wide |
| `torchvision-resnext50-32x4d` | ResNeXt-50 32x4d | grouped |
| `torchvision-densenet121` | DenseNet-121 | dense |
| `torchvision-mobilenet-v2` | MobileNet-V2 | edge |
| `torchvision-mobilenet-v3-large` | MobileNet-V3-Large | edge |
| `torchvision-efficientnet-b0` | EfficientNet-B0 | efficient |
| `torchvision-convnext-tiny` | ConvNeXt-Tiny | modern conv |

Install: `pip install 'visionservex[torchvision]'` (torch + torchvision).

## Pretrained inference

```python
from visionservex import VisionModel

m = VisionModel("torchvision-resnet50")
res = m.predict("image.jpg", top_k=5)
for label, score in res.top_k:
    print(label, round(score, 3))
```

Output is a normalized `ClassificationResult` (`kind="classification"`,
`top_k=[(label, score), ...]` over the 1000 ImageNet classes).

## Fine-tune (ImageFolder)

```python
res = VisionModel("torchvision-resnet18").train(
    "path/to/imagefolder",   # <root>/<class>/*.jpg  (or a dir with train/)
    epochs=5, batch=16, imgsz=224, lr=1e-3, device="cuda",
)
print(res["best_checkpoint"])         # runs/classify/<name>/weights/best.pt
print(res["artifacts"]["classes"])    # ['cat', 'dog', ...]
```

Dataset format: **ImageFolder** (`<root>/<class>/*.jpg`). Fine-tuning starts from
the ImageNet-pretrained backbone and swaps the final `Linear` head to the dataset's
class count. The checkpoint records the class names so reload rebuilds the right head.

## Reload a fine-tuned checkpoint + predict

```python
m = VisionModel.from_checkpoint(res["best_checkpoint"],
                                model_id="torchvision-resnet18", device="cuda")
pred = m.predict("image.jpg", top_k=2)   # predicts YOUR classes, no base fallback
```

A missing checkpoint raises a clean error; there is no silent fall back to the
ImageNet weights.

## Export (ONNX)

```python
path = VisionModel("torchvision-resnet50").export(format="onnx", output_path="out/r50.onnx")
```

ONNX (opset 18, dynamic batch) is the supported, tested export. TorchScript is
`backend_supported_but_not_integrated`; TensorRT is unsupported (not overclaimed).

## Capability truth

`VisionModel("torchvision-resnet50").capabilities()` →
`readiness="train-ready"`, `commercial_safe=True`, `legal_status="commercial_safe_core"`,
`train_supported=True`, `trained_checkpoint_predict_supported=True`,
`export_supported=["onnx"]`, `supported_dataset_formats=["imagefolder"]`.

The full lifecycle is validated live (CPU) for `torchvision-resnet18` (the training
loop is arch-generic). See `tests/live/test_v315_classifier_finetune_live.py`
(gated `VSX_LIVE_TRAIN=1`).

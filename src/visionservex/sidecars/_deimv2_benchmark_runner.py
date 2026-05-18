"""v2.26.0: DEIMv2-S sidecar benchmark runner script.

Runs inside the ``visionservex-deimv2-blackwell-nightly-sidecar`` conda env.
Loads the DEIMv2 model once, then iterates over a directory of images and
emits one JSON line per image with timing + per-image canonical detections.

Output schema (one JSON line per image, NDJSON):
    {
      "image": "<basename>",
      "image_size_wh": [W, H],
      "forward_seconds": float,
      "device": "cuda" | "cpu",
      "n_predictions": int,
      "predictions": [
        {"xyxy": [...], "score": float, "class_id": int, "class_name": str}
      ]
    }

The final stdout line is a JSON summary with totals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

COCO80_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def build_deimv2_class():
    from engine.backbone import DINOv3STAs, HGNetv2
    from engine.deim import (
        DEIMTransformer,
        DFINETransformer,
        HybridEncoder,
        LiteEncoder,
    )
    from engine.deim.postprocessor import PostProcessor
    from huggingface_hub import PyTorchModelHubMixin

    class DEIMv2(nn.Module, PyTorchModelHubMixin):
        def __init__(self, config):
            super().__init__()
            if "DINOv3STAs" in config:
                self.backbone = DINOv3STAs(**config["DINOv3STAs"])
            else:
                self.backbone = HGNetv2(**config["HGNetv2"])
            if "LiteEncoder" in config:
                self.encoder = LiteEncoder(**config["LiteEncoder"])
            else:
                self.encoder = HybridEncoder(**config["HybridEncoder"])
            if "DEIMTransformer" in config:
                self.decoder = DEIMTransformer(**config["DEIMTransformer"])
            else:
                self.decoder = DFINETransformer(**config["DFINETransformer"])
            self.postprocessor = PostProcessor(**config["PostProcessor"])

        def forward(self, x, orig_target_sizes):
            x = self.backbone(x)
            x = self.encoder(x)
            x = self.decoder(x)
            return self.postprocessor(x, orig_target_sizes)

    return DEIMv2


def _unpack_output(o):
    if isinstance(o, tuple) and len(o) == 3:
        return o
    if isinstance(o, list) and len(o) == 1:
        return _unpack_output(o[0])
    if isinstance(o, list) and len(o) == 3 and all(hasattr(v, "shape") for v in o):
        return tuple(o)
    if isinstance(o, dict) and {"labels", "boxes", "scores"}.issubset(o):
        return o["labels"], o["boxes"], o["scores"]
    raise TypeError(f"unknown out shape: {type(o).__name__}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--model-id", default="deimv2-s")
    ap.add_argument("--hf-repo", default="Intellindust/DEIMv2_DINOv3_S_COCO")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--score-threshold", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-ndjson", required=True)
    ap.add_argument("--summary-json", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo_root))
    DEIMv2 = build_deimv2_class()
    device = args.device
    m = DEIMv2.from_pretrained(args.hf_repo).to(device).eval()
    n_params = sum(p.numel() for p in m.parameters())
    tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

    image_dir = Path(args.image_dir)
    images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[: args.max_images]

    # Warm-up on first image.
    if images:
        with torch.no_grad():
            warm_img = Image.open(images[0]).convert("RGB")
            W, H = warm_img.size
            x = tf(warm_img).unsqueeze(0).to(device)
            sizes = torch.tensor([[W, H]], device=device)
            _ = m(x, sizes)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    total_pred = 0
    total_time = 0.0
    per_image_times: list[float] = []

    out_path = Path(args.output_ndjson)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out_fh:
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            x = tf(img).unsqueeze(0).to(device)
            sizes = torch.tensor([[W, H]], device=device)
            t0 = time.time()
            with torch.no_grad():
                out = m(x, sizes)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            per_image_times.append(elapsed)
            total_time += elapsed

            labels, boxes, scores = _unpack_output(out)
            if labels.dim() == 2:
                lab0, box0, scr0 = labels[0], boxes[0], scores[0]
            else:
                lab0, box0, scr0 = labels, boxes, scores
            preds = []
            n_total = scr0.shape[0]
            for i in range(n_total):
                score_f = float(scr0[i].item())
                if score_f < args.score_threshold:
                    continue
                cid = int(lab0[i].item())
                name = COCO80_LABELS[cid] if 0 <= cid < len(COCO80_LABELS) else "unknown"
                preds.append(
                    {
                        "xyxy": [float(v) for v in box0[i].tolist()],
                        "score": score_f,
                        "class_id": cid,
                        "class_name": name,
                    }
                )

            row = {
                "image": img_path.name,
                "image_size_wh": [W, H],
                "forward_seconds": round(elapsed, 4),
                "device": device,
                "n_predictions_pre_threshold": int(n_total),
                "n_predictions": len(preds),
                "predictions": preds,
            }
            out_fh.write(json.dumps(row) + "\n")
            total_pred += len(preds)

    per_image_times.sort()
    n = len(per_image_times)
    p50 = per_image_times[n // 2] if n else 0.0
    p95 = per_image_times[min(int(n * 0.95), n - 1)] if n else 0.0
    summary = {
        "status": "ok",
        "code": "OK",
        "model_id": args.model_id,
        "device": device,
        "n_params": n_params,
        "n_images": n,
        "score_threshold": args.score_threshold,
        "total_predictions": total_pred,
        "total_inference_seconds": round(total_time, 3),
        "mean_forward_seconds": round(total_time / n, 4) if n else 0.0,
        "p50_forward_seconds": round(p50, 4),
        "p95_forward_seconds": round(p95, 4),
        "images_per_second": round(n / total_time, 2) if total_time > 0 else 0.0,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.startswith("cuda") and torch.cuda.is_available()
            else ""
        ),
        "torch_version": torch.__version__,
        "output_ndjson": str(out_path.resolve()),
        "image_dir": str(image_dir.resolve()),
        "note": (
            "This is a runtime/latency probe with predictions but NOT a scientific "
            "AP benchmark (no matching COCO annotation file supplied)."
        ),
    }
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

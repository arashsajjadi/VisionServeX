"""Shared path resolution for VisionServeX notebooks."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent
NB_ROOT = _HERE.parent
REPO_ROOT = NB_ROOT.parent


def load_config():
    raw = (NB_ROOT / "shared/config.yaml").read_text()
    cfg = {}
    for line in raw.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()
    return cfg


CFG = load_config()
COCO_400_ANN = Path(CFG.get("coco_400_ann", ""))
COCO_400_IMAGES = Path(CFG.get("coco_400_images", ""))
SMOKE_IMG = Path(CFG.get("smoke_img", ""))
SMOKE_ANN = Path(CFG.get("smoke_ann", ""))
SMOKE_VIDEO = Path(CFG.get("smoke_video", ""))

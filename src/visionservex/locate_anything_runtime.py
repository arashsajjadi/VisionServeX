# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Runtime shim for NVIDIA LocateAnything-3B (non-commercial, BYOT, sidecar).

VisionServeX does NOT ship or mirror LocateAnything-3B weights. This module
bridges to the NVlabs/Eagle sidecar (installed via `visionservex locate-anything install`).

WARNING: LocateAnything-3B pretrained weights are released under the NVIDIA License
for non-commercial use only. Do not use this model for commercial products, paid SaaS,
client work, production annotation, or redistribution unless you have written commercial
permission from NVIDIA. VisionServeX does not ship or mirror the weights.
Use is BYOT/user-local-cache only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SIDECAR_MODULE = "eagle.Embodied"
_SIDECAR_INSTALL = (
    "git clone https://github.com/NVlabs/Eagle.git eagle && cd eagle/Embodied && pip install -e ."
)

# Model IDs → HuggingFace repo IDs (user must have local cache or HF access; BYOT).
_MODEL_HF_IDS: dict[str, str] = {
    "locate-anything-3b": "nvidia/locate-anything-3b",
    "locate-anything-3b-v2": "nvidia/locate-anything-3b-v2",
    "locate-anything-3b-grounded": "nvidia/locate-anything-3b-grounded",
    "locate-anything-3b-coco": "nvidia/locate-anything-3b-coco",
    "locate-anything-3b-lvis": "nvidia/locate-anything-3b-lvis",
    "locate-anything-3b-objects365": "nvidia/locate-anything-3b-objects365",
    "locate-anything-3b-open-vocab": "nvidia/locate-anything-3b-open-vocab",
    "locate-anything-3b-caption": "nvidia/locate-anything-3b-caption",
    "locate-anything-3b-video": "nvidia/locate-anything-3b-video",
    "locate-anything-3b-ft": "nvidia/locate-anything-3b-ft",
}


def _check_sidecar() -> None:
    try:
        import importlib

        importlib.import_module("eagle")
    except ImportError as exc:
        raise RuntimeError(
            f"LocateAnything-3B sidecar not installed. Install with: {_SIDECAR_INSTALL}"
        ) from exc


def run_locate_anything(
    model_id: str,
    image,
    *,
    text: str,
    cache_dir: str | Path | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Run LocateAnything-3B grounded detection via the NVlabs/Eagle sidecar.

    Raises RuntimeError if the sidecar is not installed.
    Raises ValueError if model_id is not recognised.
    """
    _check_sidecar()

    hf_id = _MODEL_HF_IDS.get(model_id)
    if hf_id is None:
        raise ValueError(
            f"Unknown LocateAnything model ID: {model_id!r}. Known IDs: {sorted(_MODEL_HF_IDS)}"
        )

    resolved_cache = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / ".cache" / "visionservex" / "locate_anything"
    )

    from eagle.Embodied.locate_anything import LocateAnythingModel  # type: ignore[import]

    model = LocateAnythingModel.from_pretrained(hf_id, cache_dir=str(resolved_cache))

    import numpy as np

    img_arr = np.asarray(image)
    result = model.locate(img_arr, text=text, **kw)
    return {
        "model_id": model_id,
        "hf_id": hf_id,
        "text": text,
        "boxes": result.get("boxes", []),
        "scores": result.get("scores", []),
        "labels": result.get("labels", []),
    }

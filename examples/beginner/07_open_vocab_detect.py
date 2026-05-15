# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 07 — open-vocabulary detection with text prompts.

Defaults to the deterministic ``mock-open-vocab`` model. To use the real
Grounding DINO backend:

    pip install 'visionservex[grounding]'
    python examples/beginner/07_open_vocab_detect.py grounding-dino-tiny
"""

from __future__ import annotations

import sys
from pathlib import Path

from visionservex import VisionModel

DEFAULT_MODEL = "mock-open-vocab"
IMAGE = Path("examples/images/street.jpg")
OUT = Path("outputs/07_open_vocab.jpg")


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_MODEL
    prompts = ["a red car", "person", "wheel"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = VisionModel(model_id, auto_pull=False)
    result = model.predict(IMAGE, prompts=prompts)
    print(result.summary())
    for det in result.detections:
        print(f"  {det.label:30s} {det.score:.2f}")
    result.save(OUT)
    print(f"Saved: {OUT}")
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

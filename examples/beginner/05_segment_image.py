# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 05 — instance segmentation on a sample image."""

from __future__ import annotations

from pathlib import Path

from visionservex import VisionModel

MODEL_ID = "mock-segment"
IMAGE = Path("examples/images/simple_shapes.jpg")
OUT = Path("outputs/05_segment.jpg")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE.exists():
        print(f"Sample image not found: {IMAGE}")
        return

    model = VisionModel(MODEL_ID)
    result = model.predict(IMAGE)
    print(result.summary())
    result.save(OUT)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

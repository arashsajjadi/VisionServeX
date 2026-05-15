# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 06 — classify an image."""

from __future__ import annotations

from pathlib import Path

from visionservex import VisionModel

MODEL_ID = "mock-classify"
IMAGE = Path("examples/images/dog.jpg")


def main() -> None:
    if not IMAGE.exists():
        print(f"Sample image not found: {IMAGE}")
        return

    model = VisionModel(MODEL_ID)
    result = model.predict(IMAGE)
    print(result.summary())
    for label, score in result.top_k:
        print(f"  {label:14s} {score:.2f}")


if __name__ == "__main__":
    main()

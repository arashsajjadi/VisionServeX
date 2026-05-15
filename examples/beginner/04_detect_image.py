# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Beginner example 04 — detect objects in an image.

Uses the deterministic `mock-detect` model so it works without any heavy
dependency. Replace ``MODEL_ID`` with a real id (e.g. ``rfdetr-small``)
once you have installed the corresponding optional extra and pulled weights.
"""

from __future__ import annotations

from pathlib import Path

from visionservex import VisionModel

MODEL_ID = "mock-detect"
IMAGE = Path("examples/images/street.jpg")
OUT = Path("outputs/04_detect.jpg")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE.exists():
        print(f"Sample image not found: {IMAGE}")
        return

    print(f"Loading model: {MODEL_ID}")
    model = VisionModel(MODEL_ID)
    print(f"  device = {model.device}, precision = {model.precision}")

    print(f"Predicting on {IMAGE} ...")
    result = model.predict(IMAGE)
    print(result.summary())

    if hasattr(result, "detections"):
        for det in result.detections:
            print(f"  {det.label:14s} score={det.score:.2f} box={det.box.to_xyxy()}")
    result.save(OUT)
    print(f"\nAnnotated image saved to: {OUT}")


if __name__ == "__main__":
    main()

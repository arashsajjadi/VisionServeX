"""Grounded segmentation example with text prompts.

Uses `grounded-sam2` which composes Grounding DINO and SAM 2. Requires the
optional ``grounding`` and ``sam2`` extras; without them, the engine falls
back to MockEngine output (a single colored mask) and emits a warning so
the pipeline keeps running for plumbing tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

from visionservex import VisionModel


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python examples/grounded_segment.py <image_path> <prompt1[,prompt2,...]>")
        return 2

    image = Path(sys.argv[1])
    prompts = [p.strip() for p in sys.argv[2].split(",") if p.strip()]

    model = VisionModel("grounded-sam2")
    result = model.predict(image, prompts=prompts)
    print(result.summary())
    print(f"warnings: {result.warnings}")
    result.save(image.with_suffix(".annotated.jpg"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

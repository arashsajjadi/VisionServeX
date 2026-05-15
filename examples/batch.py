"""Batch prediction example.

Usage:
    python examples/batch.py <model_id> <image1> [<image2> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

from visionservex import VisionModel


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    model_id = sys.argv[1]
    paths = [Path(p) for p in sys.argv[2:]]

    model = VisionModel(model_id)
    for path, result in zip(paths, model.batch_predict(paths), strict=False):
        print(path.name, result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

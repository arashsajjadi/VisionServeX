"""Run a single prediction with the Python API.

Usage:
    python examples/predict.py <model_id> <image_path> [output_path]

The mock model `mock-detect` works without optional dependencies.
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
    image_path = Path(sys.argv[2])
    output = Path(sys.argv[3]) if len(sys.argv) >= 4 else None

    model = VisionModel(model_id)
    result = model.predict(image_path)
    print(result.summary())
    print(result.to_json(indent=2))
    if output:
        path = result.save(output)
        print(f"saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

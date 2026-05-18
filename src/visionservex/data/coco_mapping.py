# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""COCO category id ↔ contiguous index mapping.

Background
----------
COCO categories use two different id schemes:

- **Official COCO category_id (1..90, with gaps)**: the raw ids that appear
  in the COCO annotation JSON. Index 88 is "teddy bear"; ids 12, 26, 29, 30,
  45, 66, 68, 69, 71, 83 are not used.
- **Contiguous index (0..79)**: the 80-class index used by most modern
  detectors (Ultralytics, D-FINE, evaluators, …).

If an engine returns official ids but downstream code looks them up in a
contiguous 80-class list, the labels get scrambled (this is the v17
RF-DETR bug — class_id=88 reported as the contiguous-index-88 label,
or class_id=47 ("cup") reported as the contiguous-47 label ("apple")).

This module ships a verified bidirectional map plus the canonical 80-class
contiguous label list. RF-DETR (and any future engine that returns official
ids) routes its labels through :func:`remap_official_to_contiguous`.
"""

from __future__ import annotations

__all__ = [
    "COCO80_CONTIGUOUS_LABELS",
    "COCO_CONTIGUOUS_TO_OFFICIAL",
    "COCO_OFFICIAL_TO_CONTIGUOUS",
    "is_official_id_set",
    "remap_official_to_contiguous",
]


# Standard COCO 80-class names in contiguous 0..79 order.
COCO80_CONTIGUOUS_LABELS: tuple[str, ...] = (
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
)
assert len(COCO80_CONTIGUOUS_LABELS) == 80


# Official COCO category id (1..90) → contiguous index (0..79).
# Source: the official COCO annotation JSON; ids 12, 26, 29, 30, 45, 66,
# 68, 69, 71, 83 are intentionally absent (deprecated supercategories).
COCO_OFFICIAL_TO_CONTIGUOUS: dict[int, int] = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}
assert len(COCO_OFFICIAL_TO_CONTIGUOUS) == 80


# Reverse map (contiguous → official).
COCO_CONTIGUOUS_TO_OFFICIAL: dict[int, int] = {v: k for k, v in COCO_OFFICIAL_TO_CONTIGUOUS.items()}


def is_official_id_set(ids: list[int] | tuple[int, ...]) -> bool:
    """Heuristic: does this iterable of class ids look like official-COCO numbering?

    Returns True if ANY id is > 79 — that's only possible under official
    numbering (contiguous max is 79). All ids 0..79 are ambiguous; we
    conservatively call them contiguous in that case.
    """
    for raw in ids:
        try:
            cid = int(raw)
        except (TypeError, ValueError):
            continue
        if cid > 79:
            return True
    return False


def remap_official_to_contiguous(
    class_id: int,
) -> tuple[int, str, str]:
    """Map one official COCO id to (contiguous_id, label, mapping_source).

    ``mapping_source`` is one of:
    - ``"coco_official_to_contiguous"`` if the id was in the official map.
    - ``"already_contiguous"`` if the id was a valid contiguous index (0..79).
    - ``"unknown"`` if neither matches; returned class_id is -1.
    """
    if class_id in COCO_OFFICIAL_TO_CONTIGUOUS:
        contiguous = COCO_OFFICIAL_TO_CONTIGUOUS[class_id]
        return contiguous, COCO80_CONTIGUOUS_LABELS[contiguous], "coco_official_to_contiguous"
    if 0 <= class_id < 80:
        return class_id, COCO80_CONTIGUOUS_LABELS[class_id], "already_contiguous"
    return -1, f"unknown_class_{class_id}", "unknown"

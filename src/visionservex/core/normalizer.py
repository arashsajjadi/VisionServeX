# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Unified detection output normalizer.

Accepts every common serialization of bounding-box detections and converts
them to VisionServeX's canonical ``Detection`` / ``Box`` objects.

Supported box encodings
------------------------
1. List xyxy:          {"xyxy": [x1, y1, x2, y2]}
2. List box:           {"box": [x1, y1, x2, y2]}
3. List bbox:          {"bbox": [x1, y1, x2, y2]}
4. xywh bbox:          {"bbox": [x, y, w, h], "bbox_format": "xywh"}
5. Dict box x1/y1...:  {"box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
6. Dict xyxy dict:     {"xyxy": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
7. Dict xmin/ymin...:  {"box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}}
8. coordinates dict:   {"coordinates": {"left": x1, "top": y1, "right": x2, "bottom": y2}}
9. VisionServeX native: {"box": {"x1": x1, ...}} (asdict(Box) output)

Supported score keys:  score, confidence, conf, probability, prob
Supported label keys:  class_name, label, category, name, phrase, class_id,
                       category_id, label_id, cls
COCO category ID mapping: official 1-90 to contiguous 0-79 (COCO80).
"""

from __future__ import annotations

from typing import Any

from visionservex.core.results import Box, Detection

# ---------------------------------------------------------------------------
# COCO official category ID → contiguous 0-indexed ID
# ---------------------------------------------------------------------------

_COCO_OFFICIAL_TO_CONTIGUOUS: dict[int, int] = {
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

_COCO80_NAMES: list[str] = [
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
]

COCO80_NAMES: list[str] = _COCO80_NAMES  # public alias

_SCORE_KEYS = ("score", "confidence", "conf", "probability", "prob")
_LABEL_KEYS = ("class_name", "label", "category", "name", "phrase")
_CLASS_ID_KEYS = ("class_id", "category_id", "label_id", "cls")


class NormalizerError(ValueError):
    """Raised when a raw detection payload cannot be parsed."""

    code: str = "OUTPUT_SCHEMA_UNRECOGNIZED"


class AllPredictionsDroppedWarning(UserWarning):
    """Emitted when normalization drops all predictions from a non-empty payload."""


def _extract_box(raw: dict[str, Any]) -> Box | None:
    """Extract a Box from a raw detection dict. Returns None if not found."""

    def _list4_to_box(v: list, *, fmt: str = "xyxy") -> Box:
        v = [float(x) for x in v[:4]]
        if fmt == "xywh":
            return Box(x1=v[0], y1=v[1], x2=v[0] + v[2], y2=v[1] + v[3])
        return Box(x1=v[0], y1=v[1], x2=v[2], y2=v[3])

    def _dict_to_box(d: dict) -> Box | None:
        # x1/y1/x2/y2
        if all(k in d for k in ("x1", "y1", "x2", "y2")):
            return Box(x1=float(d["x1"]), y1=float(d["y1"]), x2=float(d["x2"]), y2=float(d["y2"]))
        # xmin/ymin/xmax/ymax
        if all(k in d for k in ("xmin", "ymin", "xmax", "ymax")):
            return Box(
                x1=float(d["xmin"]), y1=float(d["ymin"]), x2=float(d["xmax"]), y2=float(d["ymax"])
            )
        # left/top/right/bottom
        if all(k in d for k in ("left", "top", "right", "bottom")):
            return Box(
                x1=float(d["left"]), y1=float(d["top"]), x2=float(d["right"]), y2=float(d["bottom"])
            )
        return None

    # Check "coordinates" wrapper
    if "coordinates" in raw:
        cd = raw["coordinates"]
        if isinstance(cd, dict):
            b = _dict_to_box(cd)
            if b is not None:
                return b

    # Check list forms: xyxy, box, bbox
    for key in ("xyxy", "box", "bbox"):
        val = raw.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            fmt = "xywh" if raw.get("bbox_format") == "xywh" and key == "bbox" else "xyxy"
            return _list4_to_box(val, fmt=fmt)
        if isinstance(val, dict):
            b = _dict_to_box(val)
            if b is not None:
                return b

    return None


def _extract_score(raw: dict[str, Any]) -> float:
    for key in _SCORE_KEYS:
        if key in raw:
            try:
                return float(raw[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _extract_label_and_id(raw: dict[str, Any]) -> tuple[str, int | None]:
    """Returns (class_name, class_id). class_name may be 'class_N' if unmapped."""
    class_id: int | None = None
    class_name: str | None = None

    # Try to get class_id from ID keys
    for key in _CLASS_ID_KEYS:
        if key in raw:
            try:
                class_id = int(raw[key])
                break
            except (TypeError, ValueError):
                pass

    # Try to get class_name from label keys
    for key in _LABEL_KEYS:
        if key in raw and raw[key] is not None:
            class_name = str(raw[key]).strip()
            break

    # Resolve via COCO official ID mapping
    if class_id is not None:
        # If ID is in the COCO official range and there's a mapping, use it
        if class_id in _COCO_OFFICIAL_TO_CONTIGUOUS:
            contiguous = _COCO_OFFICIAL_TO_CONTIGUOUS[class_id]
            if class_name is None:
                class_name = _COCO80_NAMES[contiguous]
            class_id = contiguous
        elif 0 <= (class_id or 0) < len(_COCO80_NAMES) and class_name is None:
            class_name = _COCO80_NAMES[class_id]

    if class_name is None and class_id is not None:
        class_name = f"class_{class_id}"
    elif class_name is None:
        class_name = "unknown"

    return class_name, class_id


def normalize_detection(raw: dict[str, Any]) -> Detection:
    """Normalize a raw detection dict to a ``Detection`` object.

    Accepts all supported box, score, and label encodings.
    Raises ``NormalizerError`` if no box can be extracted.
    """
    box = _extract_box(raw)
    if box is None:
        msg = (
            "OUTPUT_SCHEMA_UNRECOGNIZED: Cannot extract box from raw detection. "
            f"Tried keys: xyxy, box, bbox, coordinates. "
            f"Raw keys present: {sorted(raw.keys())}. "
            "Hint: ensure the dict has one of: "
            "'xyxy':[x1,y1,x2,y2], 'box':[...], 'bbox':[...], "
            "'box':{'x1':...,'y1':...,'x2':...,'y2':...}."
        )
        e = NormalizerError(msg)
        e.code = "OUTPUT_SCHEMA_UNRECOGNIZED"
        raise e

    score = _extract_score(raw)
    label, class_id = _extract_label_and_id(raw)
    return Detection(box=box, score=score, label=label, class_id=class_id)


def normalize_detections(
    raw_list: list[dict[str, Any]],
    *,
    warn_on_empty: bool = True,
) -> list[Detection]:
    """Normalize a list of raw detection dicts.

    Skips entries that cannot be parsed and logs warnings.
    If all entries are skipped and the input was non-empty, emits a strong warning.
    """
    import warnings

    results: list[Detection] = []
    failed = 0

    for raw in raw_list:
        try:
            results.append(normalize_detection(raw))
        except NormalizerError:
            failed += 1

    if warn_on_empty and failed > 0 and not results and raw_list:
        warnings.warn(
            f"OUTPUT_NORMALIZATION_DROPPED_ALL_PREDICTIONS: "
            f"All {len(raw_list)} predictions were dropped during normalization. "
            f"This likely indicates a schema mismatch. "
            f"Raw sample keys: {sorted(raw_list[0].keys())}. "
            f"Run 'visionservex debug-output MODEL IMAGE --threshold 0.01' to diagnose.",
            AllPredictionsDroppedWarning,
            stacklevel=2,
        )

    return results


def parse_api_response(response: dict[str, Any]) -> list[Detection]:
    """Parse a VisionServeX HTTP API response or serialized DetectionResult.

    Handles both the HTTP API format::

        {"detections": [{"box": {"x1": ..., ...}, "score": 0.9, "label": "cat"}]}

    and the Python dict format::

        {"detections": [{"box": {"x1": ..., ...}, ...}], "kind": "detection"}

    Returns an empty list for non-detection responses.
    """
    if response.get("kind") not in (None, "detection", "open_vocab"):
        return []

    detections = response.get("detections", [])
    if not isinstance(detections, list):
        return []

    return normalize_detections(detections)


__all__ = [
    "COCO80_NAMES",
    "AllPredictionsDroppedWarning",
    "NormalizerError",
    "normalize_detection",
    "normalize_detections",
    "parse_api_response",
]

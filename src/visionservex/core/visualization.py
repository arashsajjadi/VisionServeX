# SPDX-License-Identifier: Apache-2.0
"""Package-level visualization utilities.

Provides draw functions for every result family without requiring users to
re-implement annotation in notebooks. All functions operate on PIL.Image and
take normalized result dicts so they can be driven by the JSON contract.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ─── Color helpers ──────────────────────────────────────────────────────────


def _deterministic_color(key: str | int, *, saturation: float = 0.85, value: float = 0.95) -> tuple[int, int, int]:
    """Deterministic RGB color from a string/int key (HSV → RGB).

    Same key always returns the same color so repeated calls across frames
    keep class/track colors stable.
    """
    digest = hashlib.md5(str(key).encode("utf-8")).digest()
    h = digest[0] / 255.0
    s = saturation
    v = value
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    r, g, b = {
        0: (v, t, p), 1: (q, v, p), 2: (p, v, t),
        3: (p, q, v), 4: (t, p, v), 5: (v, p, q),
    }[i]
    return (int(r * 255), int(g * 255), int(b * 255))


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def _ensure_image(image: Image.Image | str | Path) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    return image.copy().convert("RGB")


def _draw_label_box(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    text: str,
    color: tuple[int, int, int],
    *,
    font: ImageFont.ImageFont,
) -> None:
    """Draw a filled background rectangle behind the label."""
    if not text:
        return
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * 7, 12
    pad = 2
    bg_y1 = max(0, y1 - th - 2 * pad)
    bg_y2 = bg_y1 + th + 2 * pad
    draw.rectangle([x1, bg_y1, x1 + tw + 2 * pad, bg_y2], fill=color)
    draw.text((x1 + pad, bg_y1 + pad), text, fill="white", font=font)


# ─── Detection drawing ──────────────────────────────────────────────────────


def draw_detections(
    image: Image.Image | str | Path,
    detections: list[dict] | None,
    *,
    line_width: int = 2,
    font_size: int = 14,
    hide_labels: bool = False,
    hide_conf: bool = False,
    color_by: str = "class",
    no_predictions_label: str = "no detections",
    show_no_predictions: bool = True,
) -> Image.Image:
    """Draw detection boxes on an image.

    ``detections`` is a list of dicts with at least ``box`` (xyxy dict or list)
    and optionally ``score``, ``class_name``, ``class_id``, ``label``.
    """
    img = _ensure_image(image)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    items = detections or []
    if not items and show_no_predictions:
        _draw_label_box(draw, 5, 18, no_predictions_label, (180, 60, 60), font=font)
        return img

    for det in items:
        box = _extract_xyxy(det.get("box") or det.get("xyxy") or det.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        label = det.get("class_name") or det.get("label") or det.get("phrase") or ""
        class_id = det.get("class_id")
        score = det.get("score") or det.get("confidence") or det.get("conf")
        key = label or class_id or "obj"
        color = _deterministic_color(key)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if not hide_labels:
            parts: list[str] = []
            if label:
                parts.append(str(label))
            elif class_id is not None:
                parts.append(f"#{class_id}")
            if score is not None and not hide_conf:
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(f"{float(score):.2f}")
            _draw_label_box(draw, x1, y1, " ".join(parts), color, font=font)
    return img


def _extract_xyxy(box: Any) -> tuple[float, float, float, float] | None:
    """Extract xyxy tuple from various box formats."""
    if box is None:
        return None
    if isinstance(box, dict):
        if "x1" in box and "y1" in box:
            try:
                return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            except (TypeError, KeyError, ValueError):
                return None
        if "xmin" in box and "ymin" in box:
            try:
                return float(box["xmin"]), float(box["ymin"]), float(box["xmax"]), float(box["ymax"])
            except (TypeError, KeyError, ValueError):
                return None
        if "left" in box and "top" in box:
            try:
                return float(box["left"]), float(box["top"]), float(box["right"]), float(box["bottom"])
            except (TypeError, KeyError, ValueError):
                return None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except (TypeError, ValueError):
            return None
    return None


# ─── Ground truth ───────────────────────────────────────────────────────────


def draw_ground_truth(
    image: Image.Image | str | Path,
    ground_truth: list[dict] | None,
    *,
    line_width: int = 2,
    font_size: int = 14,
    color: tuple[int, int, int] = (0, 200, 0),
) -> Image.Image:
    """Draw ground-truth boxes (default green, fixed color, no score).

    ``ground_truth`` items must have ``box`` and optionally ``class_name``/``label``.
    """
    img = _ensure_image(image)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)
    for gt in (ground_truth or []):
        box = _extract_xyxy(gt.get("box") or gt.get("xyxy") or gt.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        label = gt.get("class_name") or gt.get("label") or ""
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        if label:
            _draw_label_box(draw, x1, y1, str(label), color, font=font)
    return img


def draw_prediction_comparison(
    image: Image.Image | str | Path,
    predictions: list[dict] | None,
    ground_truth: list[dict] | None,
    *,
    line_width: int = 2,
    font_size: int = 14,
) -> Image.Image:
    """Draw ground truth (green) and predictions (per-class colors) overlaid."""
    img = _ensure_image(image)
    img = draw_ground_truth(img, ground_truth, line_width=line_width, font_size=font_size)
    img = draw_detections(img, predictions, line_width=line_width, font_size=font_size, show_no_predictions=False)
    return img


# ─── Segmentation ──────────────────────────────────────────────────────────


def draw_segmentation_masks(
    image: Image.Image | str | Path,
    masks: list[dict] | None,
    *,
    alpha: float = 0.45,
    draw_boxes: bool = True,
    font_size: int = 14,
) -> Image.Image:
    """Alpha-blend segmentation masks on top of the image.

    Each item must have a ``mask`` (numpy array) OR ``mask_path`` (path to PNG),
    plus optional ``box`` and ``class_name``.
    """
    try:
        import numpy as np
    except ImportError:
        return _ensure_image(image)

    img = _ensure_image(image).convert("RGBA")
    width, height = img.size
    items = masks or []

    for i, m in enumerate(items):
        mask = m.get("mask")
        if mask is None and m.get("mask_path"):
            try:
                mask_img = Image.open(m["mask_path"]).convert("L")
                mask = np.array(mask_img) > 0
            except Exception:
                continue
        if mask is None:
            continue
        try:
            mask_arr = np.asarray(mask, dtype=bool)
        except Exception:
            continue
        if mask_arr.shape != (height, width):
            try:
                mask_img = Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L")
                mask_img = mask_img.resize((width, height), Image.NEAREST)
                mask_arr = np.asarray(mask_img) > 0
            except Exception:
                continue

        label = m.get("class_name") or m.get("label") or ""
        color = _deterministic_color(label or i)
        overlay_color = (*color, int(alpha * 255))
        overlay_arr = np.zeros((height, width, 4), dtype=np.uint8)
        overlay_arr[mask_arr] = overlay_color
        overlay_img = Image.fromarray(overlay_arr, mode="RGBA")
        img = Image.alpha_composite(img, overlay_img)

    img_rgb = img.convert("RGB")
    if draw_boxes:
        det_list = [
            {
                "box": m.get("box"),
                "class_name": m.get("class_name") or m.get("label"),
                "score": m.get("score") or m.get("iou_score"),
            }
            for m in items
            if m.get("box")
        ]
        if det_list:
            img_rgb = draw_detections(img_rgb, det_list, font_size=font_size, show_no_predictions=False)
    return img_rgb


# ─── Pose ───────────────────────────────────────────────────────────────────


_COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),  # torso
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
]


def draw_pose(
    image: Image.Image | str | Path,
    persons: list[dict] | None,
    *,
    kpt_radius: int = 3,
    line_width: int = 2,
    kpt_threshold: float = 0.3,
    skeleton: list[tuple[int, int]] | None = None,
) -> Image.Image:
    """Draw pose skeleton + keypoints.

    Each person dict has ``keypoints`` as a list of either:
    - ``[{"x", "y", "score"}, ...]``
    - or 2D ndarray-like ``[[x, y], ...]`` with optional ``keypoint_scores``.
    """
    img = _ensure_image(image)
    draw = ImageDraw.Draw(img)
    sk = skeleton or _COCO_SKELETON

    for i, person in enumerate(persons or []):
        color = _deterministic_color(f"person_{i}")
        kps_raw = person.get("keypoints") or []
        scores = person.get("keypoint_scores") or []

        # Normalize to list of (x, y, score)
        kps: list[tuple[float, float, float]] = []
        for k_idx, k in enumerate(kps_raw):
            if isinstance(k, dict):
                try:
                    x = float(k.get("x", 0))
                    y = float(k.get("y", 0))
                    s = float(k.get("score", 1.0))
                    kps.append((x, y, s))
                except (TypeError, ValueError):
                    continue
            elif isinstance(k, (list, tuple)) and len(k) >= 2:
                try:
                    x = float(k[0])
                    y = float(k[1])
                    s = float(scores[k_idx]) if k_idx < len(scores) else 1.0
                    kps.append((x, y, s))
                except (TypeError, ValueError, IndexError):
                    continue

        # Draw bones
        for a, b in sk:
            if a < len(kps) and b < len(kps):
                xa, ya, sa = kps[a]
                xb, yb, sb = kps[b]
                if sa >= kpt_threshold and sb >= kpt_threshold:
                    draw.line([(xa, ya), (xb, yb)], fill=color, width=line_width)
        # Draw keypoints
        for x, y, s in kps:
            if s >= kpt_threshold:
                draw.ellipse([x - kpt_radius, y - kpt_radius, x + kpt_radius, y + kpt_radius],
                              fill=color, outline=color)

        # Optional bbox
        box = _extract_xyxy(person.get("bbox") or person.get("box"))
        if box:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
    return img


# ─── OBB ────────────────────────────────────────────────────────────────────


def draw_obb(
    image: Image.Image | str | Path,
    oriented_boxes: list[dict] | None,
    *,
    line_width: int = 2,
    font_size: int = 14,
    hide_labels: bool = False,
    hide_conf: bool = False,
) -> Image.Image:
    """Draw rotated rectangles.

    Each item must contain ``x_center, y_center, width, height, theta`` (radians).
    """
    img = _ensure_image(image)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    for obb in (oriented_boxes or []):
        try:
            xc = float(obb["x_center"])
            yc = float(obb["y_center"])
            w = float(obb["width"])
            h = float(obb["height"])
            theta = float(obb["theta"])
        except (KeyError, TypeError, ValueError):
            continue

        label = obb.get("class_name") or obb.get("label") or ""
        score = obb.get("score")
        color = _deterministic_color(label or "obb")

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        dx, dy = w / 2, h / 2
        corners = [
            (xc + cos_t * dx - sin_t * dy, yc + sin_t * dx + cos_t * dy),
            (xc - cos_t * dx - sin_t * dy, yc - sin_t * dx + cos_t * dy),
            (xc - cos_t * dx + sin_t * dy, yc - sin_t * dx - cos_t * dy),
            (xc + cos_t * dx + sin_t * dy, yc + sin_t * dx - cos_t * dy),
        ]
        polygon = [(int(x), int(y)) for x, y in corners]
        draw.polygon(polygon, outline=color, width=line_width)

        if not hide_labels:
            parts: list[str] = []
            if label:
                parts.append(str(label))
            if score is not None and not hide_conf:
                with contextlib.suppress(TypeError, ValueError):
                    parts.append(f"{float(score):.2f}")
            if parts:
                _draw_label_box(
                    draw, polygon[0][0], polygon[0][1], " ".join(parts), color, font=font
                )
    return img


# ─── Tracks ─────────────────────────────────────────────────────────────────


def draw_tracks(
    image: Image.Image | str | Path,
    tracks: list[dict] | None,
    *,
    line_width: int = 2,
    font_size: int = 14,
    show_id: bool = True,
    trails: dict[int, list[tuple[float, float]]] | None = None,
    trail_length: int = 30,
) -> Image.Image:
    """Draw per-track boxes with track_id colors.

    Optionally draws trails (list of recent center points per track_id).
    """
    img = _ensure_image(image)
    draw = ImageDraw.Draw(img)
    font = _load_font(font_size)

    for tb in (tracks or []):
        box = _extract_xyxy(tb.get("box") or tb.get("xyxy"))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        tid = tb.get("track_id")
        label = tb.get("class_name") or tb.get("label") or ""
        score = tb.get("score")
        color = _deterministic_color(tid if tid is not None else label)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        parts: list[str] = []
        if show_id and tid is not None:
            parts.append(f"ID:{tid}")
        if label:
            parts.append(str(label))
        if score is not None:
            with contextlib.suppress(TypeError, ValueError):
                parts.append(f"{float(score):.2f}")
        _draw_label_box(draw, x1, y1, " ".join(parts), color, font=font)

        if trails is not None and tid is not None:
            recent = trails.get(tid, [])[-trail_length:]
            if len(recent) >= 2:
                for (xa, ya), (xb, yb) in itertools.pairwise(recent):
                    draw.line([(xa, ya), (xb, yb)], fill=color, width=max(1, line_width - 1))
    return img


def draw_video_frame(
    image: Image.Image | str | Path,
    payload: dict[str, Any],
    *,
    show_fps: bool = False,
    fps_average: float | None = None,
    fps_instant: float | None = None,
    font_size: int = 14,
    line_width: int = 2,
) -> Image.Image:
    """Compose all overlays for a single video frame payload.

    ``payload`` mirrors the per-frame JSONL schema: detections, masks, tracks,
    pose, oriented_boxes.
    """
    img = _ensure_image(image)
    if payload.get("masks"):
        img = draw_segmentation_masks(img, payload["masks"], font_size=font_size)
    if payload.get("detections"):
        img = draw_detections(img, payload["detections"], font_size=font_size, line_width=line_width,
                              show_no_predictions=False)
    if payload.get("tracks"):
        img = draw_tracks(img, payload["tracks"], font_size=font_size, line_width=line_width)
    if payload.get("pose"):
        img = draw_pose(img, payload["pose"], line_width=line_width)
    if payload.get("oriented_boxes"):
        img = draw_obb(img, payload["oriented_boxes"], font_size=font_size, line_width=line_width)
    if show_fps and (fps_average is not None or fps_instant is not None):
        draw = ImageDraw.Draw(img)
        font = _load_font(font_size)
        bits: list[str] = []
        if fps_instant is not None:
            bits.append(f"FPS: {fps_instant:.1f}")
        if fps_average is not None:
            bits.append(f"avg: {fps_average:.1f}")
        _draw_label_box(draw, 5, 18, " ".join(bits), (40, 40, 40), font=font)
    return img


def annotate_image(
    image: Image.Image | str | Path,
    payload: dict[str, Any],
    *,
    font_size: int = 14,
    line_width: int = 2,
    mask_alpha: float = 0.45,
    hide_labels: bool = False,
    hide_conf: bool = False,
) -> Image.Image:
    """One-call image annotation from a canonical envelope payload.

    Routes to the right draw function based on payload['task'] or the keys
    present in the payload. Extra styling kwargs (``mask_alpha``,
    ``hide_labels``, ``hide_conf``) are accepted at this entrypoint so the
    CLI/notebook can call ``annotate_image`` without dispatching themselves;
    they are forwarded to the underlying drawer where applicable.
    """
    task = (payload.get("task") or "").lower()
    if task in ("classify", "classification"):
        img = _ensure_image(image)
        draw = ImageDraw.Draw(img)
        font = _load_font(font_size)
        topk = payload.get("topk") or []
        lines = [
            f"{i + 1}. {item.get('class_name') or item.get('label', '')} "
            f"({float(item.get('score', 0)):.2f})"
            for i, item in enumerate(topk[:5])
        ]
        for i, line in enumerate(lines):
            _draw_label_box(draw, 5, 20 + i * (font_size + 6), line, (40, 80, 200), font=font)
        return img
    if task == "obb" or payload.get("oriented_boxes"):
        return draw_obb(
            image, payload.get("oriented_boxes"),
            font_size=font_size, line_width=line_width,
            hide_labels=hide_labels, hide_conf=hide_conf,
        )
    if task == "pose" or payload.get("persons"):
        return draw_pose(image, payload.get("persons"), line_width=line_width)
    if task in ("segment", "foundation_segment") or payload.get("masks"):
        return draw_segmentation_masks(
            image, payload.get("masks"), font_size=font_size, alpha=mask_alpha,
        )
    if payload.get("tracks"):
        return draw_tracks(image, payload["tracks"], font_size=font_size, line_width=line_width)
    return draw_detections(
        image, payload.get("detections") or [], font_size=font_size, line_width=line_width,
        hide_labels=hide_labels, hide_conf=hide_conf,
    )


__all__ = [
    "annotate_image",
    "draw_detections",
    "draw_ground_truth",
    "draw_obb",
    "draw_pose",
    "draw_prediction_comparison",
    "draw_segmentation_masks",
    "draw_tracks",
    "draw_video_frame",
]

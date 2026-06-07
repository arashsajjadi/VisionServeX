"""Segmentation metrics for the smart-annotation benchmark.

mean_iou, boundary_iou, and success-rate-at-IoU thresholds. Pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np


def _as_bool(m: np.ndarray) -> np.ndarray:
    return np.asarray(m).astype(bool)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two binary masks."""
    p, g = _as_bool(pred), _as_bool(gt)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def _boundary(mask: np.ndarray, dilation: int) -> np.ndarray:
    """Boundary region of a mask, dilated by ``dilation`` px (scipy binary ops)."""
    from scipy import ndimage

    m = _as_bool(mask)
    if m.sum() == 0:
        return np.zeros_like(m)
    eroded = ndimage.binary_erosion(m, iterations=1)
    boundary = m & ~eroded
    if dilation > 0:
        boundary = ndimage.binary_dilation(boundary, iterations=dilation)
    return boundary


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to a thin boundary band.

    ``dilation_ratio`` is relative to the image diagonal.
    """
    p, g = _as_bool(pred), _as_bool(gt)
    h, w = g.shape[:2]
    diag = (h**2 + w**2) ** 0.5
    d = max(1, round(float(dilation_ratio * diag)))
    pb = _boundary(p, d)
    gb = _boundary(g, d)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def success_rate_at_iou(ious: list[float], threshold: float) -> float:
    """Fraction of samples whose IoU >= threshold."""
    if not ious:
        return 0.0
    return float(np.mean([1.0 if v >= threshold else 0.0 for v in ious]))


def summarize(ious: list[float], boundary_ious: list[float], latencies_ms: list[float]) -> dict:
    """Aggregate per-sample metrics into a benchmark summary row."""
    arr = np.asarray(ious, dtype=float) if ious else np.asarray([0.0])
    barr = np.asarray(boundary_ious, dtype=float) if boundary_ious else np.asarray([0.0])
    lat = np.asarray(latencies_ms, dtype=float) if latencies_ms else np.asarray([0.0])
    return {
        "n": len(ious),
        "mean_iou": round(float(arr.mean()), 4),
        "boundary_iou": round(float(barr.mean()), 4),
        "success_rate_at_iou_50": round(success_rate_at_iou(ious, 0.50), 4),
        "success_rate_at_iou_75": round(success_rate_at_iou(ious, 0.75), 4),
        "latency_ms_mean": round(float(lat.mean()), 3),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "cpu_only_ok": True,
    }

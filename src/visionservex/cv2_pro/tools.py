"""CV2-Pro tool implementations + registry.

Every tool returns a JSON-serialisable dict with a stable output contract:
    {"tool": str, "kind": "proposals|mask|contours|onnx", "device": "cpu",
     "license_safe": True, "latency_ms": float, ...tool-specific...}
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

# tool_name -> (callable, kind, requires_contrib, license_str)
_REGISTRY: dict[str, tuple[Callable, str, bool, str]] = {}

_APACHE = "OpenCV Apache-2.0"


def register(name: str, kind: str, requires_contrib: bool, lic: str = _APACHE):
    def _wrap(fn: Callable) -> Callable:
        _REGISTRY[name] = (fn, kind, requires_contrib, lic)
        return fn

    return _wrap


def list_tools() -> list[str]:
    return sorted(_REGISTRY)


def TOOL_LICENSE() -> dict[str, str]:
    return {n: lic for n, (_, _, _, lic) in _REGISTRY.items()}


def _has_ximgproc() -> bool:
    import cv2

    return hasattr(cv2, "ximgproc")


def tool_available(name: str) -> tuple[bool, str]:
    if name not in _REGISTRY:
        return False, f"unknown tool {name!r}"
    _, _, needs_contrib, _ = _REGISTRY[name]
    if needs_contrib and not _has_ximgproc():
        return (
            False,
            "requires opencv-contrib-python-headless (pip install 'visionservex[cv2-pro]')",
        )
    return True, "ok"


def _img(image) -> np.ndarray:
    import cv2

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr[..., :3].astype("uint8")


def run_tool(name: str, image, **params) -> dict[str, Any]:
    ok, why = tool_available(name)
    if not ok:
        raise RuntimeError(f"cv2-pro tool {name!r} unavailable: {why}")
    fn, kind, _, lic = _REGISTRY[name]
    t0 = time.perf_counter()
    out = fn(_img(image), **params)
    out.update(
        {
            "tool": name,
            "kind": kind,
            "device": "cpu",
            "license_safe": True,
            "license": lic,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Region proposals
# ---------------------------------------------------------------------------
def _selective_search(image, strategy: str, fast: bool, max_boxes: int) -> dict:
    import cv2

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if fast:
        ss.switchToSelectiveSearchFast()
    else:
        ss.switchToSelectiveSearchQuality()
    rects = ss.process()[:max_boxes]
    boxes = [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in rects]
    return {"n_proposals": len(boxes), "boxes_xyxy": boxes}


@register("opencv-selective-search-fast", "proposals", True)
def selective_search_fast(image, max_boxes: int = 500) -> dict:
    return _selective_search(image, "all", True, max_boxes)


@register("opencv-selective-search-quality", "proposals", True)
def selective_search_quality(image, max_boxes: int = 500) -> dict:
    return _selective_search(image, "all", False, max_boxes)


@register("opencv-mser-proposals", "proposals", False)
def mser_proposals(image, max_boxes: int = 500) -> dict:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    boxes = []
    for pts in regions[:max_boxes]:
        x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
        boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return {"n_proposals": len(boxes), "boxes_xyxy": boxes}


# ---------------------------------------------------------------------------
# Segmentation refinement
# ---------------------------------------------------------------------------
@register("opencv-grabcut-plus", "mask", False)
def grabcut_plus(image, box: list | None = None, iters: int = 5) -> dict:
    import cv2

    h, w = image.shape[:2]
    if box is None:
        box = [w // 8, h // 8, w - w // 8, h - h // 8]
    x1, y1, x2, y2 = (int(v) for v in box)
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image,
        mask,
        (x1, y1, max(1, x2 - x1), max(1, y2 - y1)),
        bgd,
        fgd,
        iters,
        cv2.GC_INIT_WITH_RECT,
    )
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype("uint8")
    return {"mask_shape": list(out.shape), "mask_area": int(out.sum()), "bbox": [x1, y1, x2, y2]}


@register("opencv-watershed-plus", "mask", False)
def watershed_plus(image, fg_threshold: float = 0.5) -> dict:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, fg_threshold * dist.max(), 255, 0)
    sure_fg = sure_fg.astype("uint8")
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    return {"n_regions": int(n), "boundary_pixels": int((markers == -1).sum())}


@register("opencv-connected-components-refine", "mask", False)
def connected_components_refine(image, mask=None, min_area: int = 50) -> dict:
    import cv2

    if mask is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (np.asarray(mask) > 0).astype("uint8")
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    kept = [int(i) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return {"n_components": int(n - 1), "n_kept": len(kept), "min_area": min_area}


@register("opencv-contour-snap", "contours", False)
def contour_snap(image, mask=None, epsilon_ratio: float = 0.002) -> dict:
    import cv2

    if mask is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (np.asarray(mask) > 0).astype("uint8")
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) <= 0:
            continue
        eps = epsilon_ratio * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        polys.append([[int(x), int(y)] for x, y in approx])
    return {"n_polygons": len(polys), "polygons": polys[:50]}


@register("opencv-intelligent-scissors", "contours", False)
def intelligent_scissors(image, polyline: list | None = None) -> dict:
    import cv2

    h, w = image.shape[:2]
    tool = cv2.segmentation.IntelligentScissorsMB()
    tool.setEdgeFeatureCannyParameters(32, 100)
    tool.applyImage(image)
    if not polyline or len(polyline) < 2:
        polyline = [
            [w // 4, h // 4],
            [3 * w // 4, h // 4],
            [3 * w // 4, 3 * h // 4],
            [w // 4, 3 * h // 4],
        ]
    contour: list = []
    pts = [(int(p[0]), int(p[1])) for p in polyline]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    for i in range(len(pts) - 1):
        tool.buildMap(pts[i])
        seg = tool.getContour(pts[i + 1])
        contour.extend(seg.reshape(-1, 2).tolist())
    return {"n_contour_points": len(contour)}


# ---------------------------------------------------------------------------
# Background subtraction (video)
# ---------------------------------------------------------------------------
def _bg_subtract(image, algo: str) -> dict:
    import cv2

    sub = (
        cv2.createBackgroundSubtractorMOG2()
        if algo == "mog2"
        else cv2.createBackgroundSubtractorKNN()
    )
    # single-frame demo: prime + apply
    sub.apply(image)
    fg = sub.apply(image)
    return {"algo": algo, "fg_pixels": int((fg > 0).sum()), "frame_shape": list(image.shape[:2])}


@register("opencv-video-fg-bg-mog2", "mask", False)
def video_fgbg_mog2(image) -> dict:
    return _bg_subtract(image, "mog2")


@register("opencv-video-fg-bg-knn", "mask", False)
def video_fgbg_knn(image) -> dict:
    return _bg_subtract(image, "knn")


# ---------------------------------------------------------------------------
# DNN ONNX runner
# ---------------------------------------------------------------------------
@register("opencv-kmeans-color-segment", "mask", False)
def kmeans_color_segment(image, k: int = 4) -> dict:
    """K-means colour quantisation → flat colour regions (annotation pre-segment)."""
    import cv2

    z = image.reshape(-1, 3).astype("float32")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, _centers = cv2.kmeans(z, int(k), None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(image.shape[:2])
    sizes = [int((labels == i).sum()) for i in range(int(k))]
    return {"n_regions": int(k), "region_sizes": sizes}


@register("opencv-distance-transform-markers", "mask", False)
def distance_transform_markers(image, fg_threshold: float = 0.4) -> dict:
    """Distance-transform peaks → instance markers (seeds for watershed/SAM)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thr, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, fg_threshold * dist.max(), 255, 0)
    n, _ = cv2.connectedComponents(peaks.astype("uint8"))
    return {"n_markers": int(n - 1), "max_distance": round(float(dist.max()), 2)}


@register("opencv-dnn-onnx-runner", "onnx", False)
def dnn_onnx_runner(image, onnx: str | None = None, size: int = 640) -> dict:
    import cv2

    if not onnx:
        return {
            "status": "no_onnx_provided",
            "hint": "pass onnx=<path>.onnx",
            "dnn_backends": int(cv2.dnn.DNN_BACKEND_OPENCV),
        }
    net = cv2.dnn.readNetFromONNX(onnx)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()
    return {"status": "ok", "onnx": onnx, "output_shape": list(np.asarray(out).shape)}

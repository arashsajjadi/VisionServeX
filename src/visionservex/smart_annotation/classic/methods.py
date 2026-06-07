"""Eight classic, weight-free interactive refiners.

All run on CPU with permissive dependencies only (OpenCV Apache-2.0;
scikit-image / scikit-learn / scipy / numpy BSD-3). No GPL dependency is used
(in particular NO PyMaxflow); the "graph-cut" methods use OpenCV GrabCut, whose
min-cut solver ships under Apache-2.0.
"""

from __future__ import annotations

import time

import numpy as np

from ..contracts import Prompt, RefineResult
from .base import (
    box_from_prompt,
    finalize,
    rasterize_points,
    register,
)

_CV2_LIC = "OpenCV Apache-2.0"
_SKIMAGE_LIC = "scikit-image BSD-3"
_SKLEARN_LIC = "scikit-learn BSD-3"


def _ensure_image(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim == 2:
        import cv2

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype("uint8")
    return img[..., :3]


def _grabcut_mask(image: np.ndarray, prompt: Prompt, iters: int = 5) -> np.ndarray:
    """Core GrabCut: rect init from the prompt box, refined by point seeds."""
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_from_prompt(prompt, image.shape)
    gc = np.full((h, w), cv2.GC_PR_BGD, dtype="uint8")
    gc[y1:y2, x1:x2] = cv2.GC_PR_FGD
    used_seeds = False
    if prompt.positive_points:
        pos = rasterize_points(prompt.positive_points, image.shape, radius=3)
        gc[pos > 0] = cv2.GC_FGD
        used_seeds = True
    if prompt.scribble is not None:
        scr = rasterize_points(prompt.scribble, image.shape, radius=3)
        gc[scr > 0] = cv2.GC_FGD
        used_seeds = True
    if prompt.negative_points:
        neg = rasterize_points(prompt.negative_points, image.shape, radius=3)
        gc[neg > 0] = cv2.GC_BGD
        used_seeds = True
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    mode = cv2.GC_INIT_WITH_MASK if used_seeds else cv2.GC_INIT_WITH_RECT
    rect = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
    try:
        cv2.grabCut(image, gc, rect, bgd, fgd, iters, mode)
    except cv2.error:
        # Degenerate seeds: fall back to a pure rect init.
        gc = np.zeros((h, w), dtype="uint8")
        cv2.grabCut(image, gc, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype("uint8")


@register("classic-grabcut", _CV2_LIC)
def grabcut(image: np.ndarray, prompt: Prompt) -> RefineResult:
    t0 = time.perf_counter()
    img = _ensure_image(image)
    mask = _grabcut_mask(img, prompt)
    return finalize(mask, "classic-grabcut", t0, confidence=None)


def _fg_bg_seeds(image: np.ndarray, prompt: Prompt):
    """Build foreground/background seed maps from the prompt for label methods."""
    h, w = image.shape[:2]
    fg = np.zeros((h, w), dtype=bool)
    bg = np.zeros((h, w), dtype=bool)
    if prompt.positive_points:
        fg |= rasterize_points(prompt.positive_points, image.shape, radius=4).astype(bool)
    if prompt.scribble is not None:
        fg |= rasterize_points(prompt.scribble, image.shape, radius=4).astype(bool)
    if prompt.negative_points:
        bg |= rasterize_points(prompt.negative_points, image.shape, radius=4).astype(bool)
    if not fg.any():
        # Derive seeds from the box: inner region = fg, outer frame = bg.
        x1, y1, x2, y2 = box_from_prompt(prompt, image.shape)
        cx1 = x1 + (x2 - x1) // 4
        cy1 = y1 + (y2 - y1) // 4
        cx2 = x2 - (x2 - x1) // 4
        cy2 = y2 - (y2 - y1) // 4
        fg[max(cy1, 0) : cy2, max(cx1, 0) : cx2] = True
    if not bg.any():
        x1, y1, x2, y2 = box_from_prompt(prompt, image.shape)
        frame = np.ones((h, w), dtype=bool)
        frame[max(y1 - 2, 0) : min(y2 + 2, h), max(x1 - 2, 0) : min(x2 + 2, w)] = False
        bg = frame
    return fg, bg


@register("classic-random-walker", _SKIMAGE_LIC)
def random_walker(image: np.ndarray, prompt: Prompt) -> RefineResult:
    from skimage.segmentation import random_walker as rw

    t0 = time.perf_counter()
    img = _ensure_image(image)
    fg, bg = _fg_bg_seeds(img, prompt)
    markers = np.zeros(img.shape[:2], dtype="int32")
    markers[bg] = 1
    markers[fg] = 2
    if not (fg.any() and bg.any()):
        return finalize(fg.astype("uint8"), "classic-random-walker", t0)
    gray = img.astype("float64").mean(axis=2)
    gray = (gray - gray.min()) / (float(np.ptp(gray)) + 1e-6)
    labels = rw(gray, markers, beta=130, mode="bf")
    return finalize((labels == 2).astype("uint8"), "classic-random-walker", t0)


@register("classic-marker-watershed", _CV2_LIC)
def marker_watershed(image: np.ndarray, prompt: Prompt) -> RefineResult:
    import cv2

    t0 = time.perf_counter()
    img = _ensure_image(image)
    fg, bg = _fg_bg_seeds(img, prompt)
    markers = np.zeros(img.shape[:2], dtype="int32")
    markers[bg] = 1
    markers[fg] = 2
    unknown = ~(fg | bg)
    markers[unknown] = 0
    cv2.watershed(img, markers)
    return finalize((markers == 2).astype("uint8"), "classic-marker-watershed", t0)


@register("classic-slic-graphcut", _SKIMAGE_LIC + " + " + _CV2_LIC)
def slic_graphcut(image: np.ndarray, prompt: Prompt) -> RefineResult:
    """SLIC superpixels snapped onto a GrabCut (graph-cut) foreground estimate."""
    from skimage.segmentation import slic

    t0 = time.perf_counter()
    img = _ensure_image(image)
    base = _grabcut_mask(img, prompt)
    n_seg = int(np.clip((img.shape[0] * img.shape[1]) / 600, 80, 600))
    sp = slic(img, n_segments=n_seg, compactness=12, start_label=0, channel_axis=-1)
    out = np.zeros_like(base)
    for lab in np.unique(sp):
        region = sp == lab
        if base[region].mean() >= 0.5:
            out[region] = 1
    if out.sum() == 0:  # fall back to raw grabcut if snapping erased everything
        out = base
    return finalize(out, "classic-slic-graphcut", t0)


def _pixel_features(image: np.ndarray) -> np.ndarray:
    """Per-pixel colour + smoothed + gradient features for the RF classifier."""
    import cv2

    img = image.astype("float32") / 255.0
    lab = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_BGR2LAB).astype("float32") / 255.0
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    grad = (grad / (grad.max() + 1e-6))[..., None]
    feats = np.concatenate([img, lab, blur, grad], axis=2)
    return feats.reshape(-1, feats.shape[2])


@register("classic-interactive-rf", _SKLEARN_LIC)
def interactive_rf(image: np.ndarray, prompt: Prompt) -> RefineResult:
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.perf_counter()
    img = _ensure_image(image)
    h, w = img.shape[:2]
    fg, bg = _fg_bg_seeds(img, prompt)
    feats = _pixel_features(img)
    yfg = fg.reshape(-1)
    ybg = bg.reshape(-1)
    if yfg.sum() == 0 or ybg.sum() == 0:
        return finalize(fg.astype("uint8"), "classic-interactive-rf", t0)
    X = np.vstack([feats[yfg], feats[ybg]])
    y = np.concatenate([np.ones(int(yfg.sum())), np.zeros(int(ybg.sum()))])
    clf = RandomForestClassifier(n_estimators=40, max_depth=12, n_jobs=1, random_state=0)
    clf.fit(X, y)
    prob = clf.predict_proba(feats)[:, 1].reshape(h, w)
    mask = (prob >= 0.5).astype("uint8")
    conf = float(prob[mask > 0].mean()) if mask.any() else 0.0
    return finalize(mask, "classic-interactive-rf", t0, confidence=round(conf, 4))


@register("classic-slic-rf-smooth", _SKIMAGE_LIC + " + " + _SKLEARN_LIC)
def slic_rf_smooth(image: np.ndarray, prompt: Prompt) -> RefineResult:
    """RandomForest over SLIC superpixels — smoother, boundary-snapped masks."""
    from skimage.segmentation import slic
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.perf_counter()
    img = _ensure_image(image)
    h, w = img.shape[:2]
    fg, bg = _fg_bg_seeds(img, prompt)
    n_seg = int(np.clip((h * w) / 500, 80, 700))
    sp = slic(img, n_segments=n_seg, compactness=12, start_label=0, channel_axis=-1)
    feats_px = _pixel_features(img)
    labels = np.unique(sp)
    # superpixel feature = mean pixel feature; label by seed majority
    sp_feat, sp_y, sp_ids = [], [], []
    flat_sp = sp.reshape(-1)
    fg_f, bg_f = fg.reshape(-1), bg.reshape(-1)
    for lab in labels:
        m = flat_sp == lab
        sp_feat.append(feats_px[m].mean(axis=0))
        f, b = fg_f[m].sum(), bg_f[m].sum()
        sp_y.append(1 if f > b else (0 if b > 0 else -1))
        sp_ids.append(lab)
    sp_feat = np.asarray(sp_feat)
    sp_y = np.asarray(sp_y)
    train = sp_y >= 0
    if train.sum() < 2 or len(np.unique(sp_y[train])) < 2:
        # fall back to raw seed majority
        out = np.zeros((h, w), dtype="uint8")
        for lab, yy in zip(sp_ids, sp_y, strict=False):
            if yy == 1:
                out[sp == lab] = 1
        return finalize(out, "classic-slic-rf-smooth", t0)
    clf = RandomForestClassifier(n_estimators=60, max_depth=10, n_jobs=1, random_state=0)
    clf.fit(sp_feat[train], sp_y[train])
    pred = clf.predict(sp_feat)
    out = np.zeros((h, w), dtype="uint8")
    for lab, yy in zip(sp_ids, pred, strict=False):
        if yy == 1:
            out[sp == lab] = 1
    return finalize(out, "classic-slic-rf-smooth", t0)


@register("classic-intelligent-scissors", _CV2_LIC)
def intelligent_scissors(image: np.ndarray, prompt: Prompt) -> RefineResult:
    """Minimal-cost contour through ordered polyline/polygon points (live-wire)."""
    import cv2

    t0 = time.perf_counter()
    img = _ensure_image(image)
    h, w = img.shape[:2]
    pts = prompt.polyline or prompt.polygon
    if not pts or len(pts) < 2:
        # Degenerate: no contour points — fall back to grabcut box.
        return finalize(_grabcut_mask(img, prompt), "classic-intelligent-scissors", t0)
    tool = cv2.segmentation.IntelligentScissorsMB()
    tool.setEdgeFeatureCannyParameters(32, 100)
    tool.setGradientMagnitudeMaxLimit(200)
    tool.applyImage(img)
    contour: list[tuple[int, int]] = []
    ordered = [(round(float(x)), round(float(y))) for x, y in pts]
    if ordered[0] != ordered[-1]:
        ordered.append(ordered[0])  # close the loop
    for i in range(len(ordered) - 1):
        sx, sy = ordered[i]
        ex, ey = ordered[i + 1]
        tool.buildMap((int(np.clip(sx, 0, w - 1)), int(np.clip(sy, 0, h - 1))))
        seg = tool.getContour((int(np.clip(ex, 0, w - 1)), int(np.clip(ey, 0, h - 1))))
        contour.extend([tuple(p) for p in seg.reshape(-1, 2)])
    mask = np.zeros((h, w), dtype="uint8")
    if len(contour) >= 3:
        cv2.fillPoly(mask, [np.asarray(contour, dtype="int32")], 1)
    if mask.sum() == 0:
        mask = _grabcut_mask(img, prompt)
    return finalize(mask, "classic-intelligent-scissors", t0)


@register("classic-edge-plus", _CV2_LIC + " + " + _SKIMAGE_LIC)
def edge_plus(image: np.ndarray, prompt: Prompt) -> RefineResult:
    """GrabCut estimate snapped to strong image edges + morphological cleanup."""
    import cv2
    from scipy import ndimage

    t0 = time.perf_counter()
    img = _ensure_image(image)
    base = _grabcut_mask(img, prompt)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150) > 0
    # Cut the mask along strong edges, then keep the component containing the seed.
    cut = base.astype(bool) & ~edges
    lbl, n = ndimage.label(cut)
    if n >= 1:
        x1, y1, x2, y2 = box_from_prompt(prompt, img.shape)
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        seed_lab = lbl[min(cy, img.shape[0] - 1), min(cx, img.shape[1] - 1)]
        if seed_lab == 0:  # seed fell on an edge — pick the largest component
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
            seed_lab = int(np.argmax(sizes)) + 1
        keep = lbl == seed_lab
    else:
        keep = base.astype(bool)
    keep = ndimage.binary_closing(keep, iterations=2)
    keep = ndimage.binary_fill_holes(keep)
    return finalize(keep.astype("uint8"), "classic-edge-plus", t0)

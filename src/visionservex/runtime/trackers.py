# SPDX-License-Identifier: Apache-2.0
"""Tracker adapter registry for VisionServeX video-search.

All tracker adapters share one interface so callers can swap freely::

    tracker.update(
        detections,
        frame_idx=...,
        timestamp_s=...,
    ) -> list[TrackBox]

`detections` is a list of ``((x1, y1, x2, y2), score, label)`` tuples.
Only the simple-iou tracker is built-in; ByteTrack and OC-SORT are
optional extras that load lazily and raise a structured
``TrackerUnavailableError`` if their package is missing.

Licensing:
- simple-iou : built-in (Apache-2.0)
- bytetrack  : MIT — optional extra (pip install bytetracker)
- ocsort     : MIT — optional extra (pip install ocsort)
- deepsort   : GPL-3.0 — NOT routed to permissive core
- bot-sort   : Apache-2.0 — git-only install, expert sidecar
"""

from __future__ import annotations

from typing import Any

_TRACKER_PACKAGES: dict[str, str] = {
    "bytetrack": "bytetracker",
    "bot-sort": "botsort",
    "ocsort": "ocsort",
}

# Structured error codes that callers can dispatch on.
_TRACKER_BLOCKER_CODES: dict[str, str] = {
    "bytetrack": "BYTETRACK_REQUIRED",
    "ocsort": "OCSORT_REQUIRED",
    "bot-sort": "BOTSORT_REQUIRED",
    "deepsort": "DEEPSORT_GPL_BLOCKED",
}

_INSTALL_HINTS: dict[str, str] = {
    "bytetrack": "pip install bytetracker  # or git clone https://github.com/ifzhang/ByteTrack",
    "ocsort": "pip install ocsort  # or git clone https://github.com/noahcao/OC_SORT",
    "bot-sort": "git clone https://github.com/NirAharon/BoT-SORT && pip install -e .",
    "deepsort": (
        "deepsort is GPL-3.0 and excluded from VisionServeX's permissive core. "
        "If you accept GPL-3.0, install it manually."
    ),
}


class TrackerUnavailableError(Exception):
    """Raised when a tracker name resolves to a package that is missing."""

    def __init__(self, name: str, code: str, install: str) -> None:
        super().__init__(f"{code}: {name}")
        self.name = name
        self.code = code
        self.install = install

    def to_dict(self) -> dict:
        return {"code": self.code, "tracker": self.name, "install": self.install}


class TrackerAPIUnsupportedError(Exception):
    """Raised when a tracker package is installed but its API has shifted."""

    def __init__(
        self,
        name: str,
        code: str,
        installed_version: str | None,
        available_attrs: list[str],
    ) -> None:
        super().__init__(f"{code}: {name} (installed_version={installed_version})")
        self.name = name
        self.code = code
        self.installed_version = installed_version
        self.available_attrs = available_attrs

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "tracker": self.name,
            "installed_version": self.installed_version,
            "available_attrs": self.available_attrs,
        }


def _probe(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def list_trackers() -> dict[str, dict[str, Any]]:
    """Return registry of known tracker adapters with install status."""
    out: dict[str, dict[str, Any]] = {
        "simple-iou": {
            "installed": True,
            "license": "Apache-2.0",
            "core_safe": True,
            "install": None,
            "description": "Built-in IoU tracker (no extra install).",
        }
    }
    for name, pkg in _TRACKER_PACKAGES.items():
        out[name] = {
            "installed": _probe(pkg),
            "license": {
                "bytetrack": "MIT",
                "ocsort": "MIT",
                "bot-sort": "Apache-2.0",
            }.get(name, "unknown"),
            "core_safe": True,
            "install": _INSTALL_HINTS.get(name),
            "code": _TRACKER_BLOCKER_CODES.get(name),
        }
    # GPL trackers — always excluded from permissive core.
    out["deepsort"] = {
        "installed": False,
        "license": "GPL-3.0",
        "core_safe": False,
        "install": _INSTALL_HINTS["deepsort"],
        "code": _TRACKER_BLOCKER_CODES["deepsort"],
        "description": "DeepSORT is GPL-3.0 — excluded from permissive core.",
    }
    return out


def build_tracker(name: str, **kwargs: Any) -> Any:
    """Build a tracker adapter instance.

    Returns ``None`` for simple-iou so callers can use SimpleIoUTracker
    directly. Raises TrackerUnavailableError when an optional package is
    missing.
    """
    if name in ("simple-iou", "simple_iou", ""):
        return None
    if name == "deepsort":
        raise TrackerUnavailableError(
            "deepsort",
            _TRACKER_BLOCKER_CODES["deepsort"],
            _INSTALL_HINTS["deepsort"],
        )

    pkg = _TRACKER_PACKAGES.get(name)
    if not pkg:
        raise TrackerUnavailableError(
            name,
            "TRACKER_UNKNOWN",
            f"Unknown tracker {name!r}. Available: "
            f"{', '.join(sorted({'simple-iou', *_TRACKER_PACKAGES}))}",
        )

    if not _probe(pkg):
        raise TrackerUnavailableError(
            name,
            _TRACKER_BLOCKER_CODES.get(name, "TRACKER_REQUIRED"),
            _INSTALL_HINTS.get(name, f"pip install {pkg}"),
        )

    if name == "bytetrack":
        return _ByteTrackAdapter(**kwargs)
    if name == "ocsort":
        return _OCSortAdapter(**kwargs)
    # bot-sort placeholder until adapter ships.
    raise TrackerUnavailableError(
        name,
        _TRACKER_BLOCKER_CODES.get(name, "TRACKER_REQUIRED"),
        _INSTALL_HINTS.get(name, f"pip install {pkg}"),
    )


# ---------------------------------------------------------------------------
# ByteTrack
# ---------------------------------------------------------------------------


class _ByteTrackAdapter:
    """Wrap the upstream bytetracker package to a uniform interface."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            from bytetracker import BYTETracker  # type: ignore
        except ImportError as exc:  # pragma: no cover - probed earlier
            raise TrackerUnavailableError(
                "bytetrack",
                _TRACKER_BLOCKER_CODES["bytetrack"],
                _INSTALL_HINTS["bytetrack"],
            ) from exc

        class _Args:
            track_thresh = kwargs.get("track_thresh", 0.5)
            track_buffer = kwargs.get("track_buffer", 30)
            match_thresh = kwargs.get("match_thresh", 0.8)
            mot20 = False

        try:
            self._tracker = BYTETracker(_Args(), frame_rate=kwargs.get("frame_rate", 30))
        except TypeError:
            try:
                self._tracker = BYTETracker(_Args())
            except Exception as exc:  # pragma: no cover - hard to reach
                import bytetracker as _bt  # type: ignore

                raise TrackerAPIUnsupportedError(
                    "bytetrack",
                    "BYTETRACK_API_UNSUPPORTED",
                    getattr(_bt, "__version__", None),
                    [a for a in dir(_bt) if not a.startswith("_")][:20],
                ) from exc
        self._frame_count = 0

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], float, str]],
        *,
        frame_idx: int = 0,
        timestamp_s: float = 0.0,
        img_size: tuple[int, int] = (1080, 1920),
    ) -> list[Any]:
        import numpy as np

        from visionservex.runtime.simple_tracker import TrackBox

        if not detections:
            self._frame_count += 1
            return []

        dets_arr = np.asarray(
            [[b[0], b[1], b[2], b[3], s] for (b, s, _lbl) in detections],
            dtype=np.float32,
        )
        try:
            tracked = self._tracker.update(
                dets_arr,
                img_size=img_size,
                img_info=img_size,
            )
        except TypeError:
            tracked = self._tracker.update(dets_arr)
        self._frame_count += 1

        results: list[Any] = []
        for t, det in zip(tracked, detections, strict=False):
            try:
                tid = int(t.track_id)
                tlbr = t.tlbr
                score = float(getattr(t, "score", det[1]))
            except AttributeError:
                continue
            results.append(
                TrackBox(
                    track_id=tid,
                    box=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                    score=score,
                    label=det[2] if len(det) > 2 else "object",
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
        return results


# ---------------------------------------------------------------------------
# OC-SORT
# ---------------------------------------------------------------------------


class _OCSortAdapter:
    """Wrap the upstream ocsort package to a uniform interface.

    OC-SORT's ``update`` signature is roughly::

        update(dets, img_info=(H, W), img_size=(H, W))

    where ``dets`` is an ndarray ``[[x1, y1, x2, y2, score], ...]``.
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
            # Both ``from ocsort.ocsort import OCSort`` and
            # ``from ocsort import OCSort`` appear in the wild. Try both.
            try:
                from ocsort.ocsort import OCSort  # type: ignore
            except ImportError:
                from ocsort import OCSort  # type: ignore
        except ImportError as exc:  # pragma: no cover - probed earlier
            raise TrackerUnavailableError(
                "ocsort",
                _TRACKER_BLOCKER_CODES["ocsort"],
                _INSTALL_HINTS["ocsort"],
            ) from exc

        try:
            self._tracker = OCSort(
                det_thresh=kwargs.get("det_thresh", 0.6),
                max_age=kwargs.get("max_age", 30),
                min_hits=kwargs.get("min_hits", 3),
                iou_threshold=kwargs.get("iou_threshold", 0.3),
            )
        except TypeError as exc:
            import ocsort as _oc  # type: ignore

            raise TrackerAPIUnsupportedError(
                "ocsort",
                "OCSORT_API_UNSUPPORTED",
                getattr(_oc, "__version__", None),
                [a for a in dir(_oc) if not a.startswith("_")][:20],
            ) from exc
        self._frame_count = 0

    def update(
        self,
        detections: list[tuple[tuple[float, float, float, float], float, str]],
        *,
        frame_idx: int = 0,
        timestamp_s: float = 0.0,
        img_size: tuple[int, int] = (1080, 1920),
    ) -> list[Any]:
        import numpy as np

        from visionservex.runtime.simple_tracker import TrackBox

        if not detections:
            self._frame_count += 1
            return []

        dets_arr = np.asarray(
            [[b[0], b[1], b[2], b[3], s] for (b, s, _lbl) in detections],
            dtype=np.float32,
        )
        try:
            tracked = self._tracker.update(
                dets_arr,
                img_info=img_size,
                img_size=img_size,
            )
        except TypeError:
            try:
                tracked = self._tracker.update(dets_arr)
            except Exception as exc:
                import ocsort as _oc  # type: ignore

                raise TrackerAPIUnsupportedError(
                    "ocsort",
                    "OCSORT_API_UNSUPPORTED",
                    getattr(_oc, "__version__", None),
                    [a for a in dir(_oc) if not a.startswith("_")][:20],
                ) from exc
        self._frame_count += 1

        results: list[Any] = []
        # OC-SORT returns ndarray rows [x1, y1, x2, y2, track_id, ...].
        try:
            rows = list(tracked)
        except TypeError:
            rows = []
        for row, det in zip(rows, detections, strict=False):
            try:
                x1, y1, x2, y2, tid = (
                    float(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    int(row[4]),
                )
            except (IndexError, ValueError, TypeError):
                continue
            score = float(det[1])
            label = det[2] if len(det) > 2 else "object"
            results.append(
                TrackBox(
                    track_id=tid,
                    box=(x1, y1, x2, y2),
                    score=score,
                    label=label,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
        return results


__all__ = [
    "TrackerAPIUnsupportedError",
    "TrackerUnavailableError",
    "build_tracker",
    "list_trackers",
]

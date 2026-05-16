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


def _detections_to_tensor_or_array(
    detections: list[tuple[tuple[float, float, float, float], float, str]],
    *,
    n_cols: int,
    as_torch: bool,
):
    """Convert detections to the form an upstream tracker expects.

    bytetracker 0.3.x and ocsort 0.0.x both require a torch.Tensor with
    columns ``[x1, y1, x2, y2, score, class_id]`` (6). Newer versions may
    accept a 5-column ndarray. ``n_cols`` and ``as_torch`` let callers
    pick.
    """
    import numpy as np

    if n_cols == 6:
        rows = [[b[0], b[1], b[2], b[3], s, 0.0] for (b, s, _lbl) in detections]
    else:
        rows = [[b[0], b[1], b[2], b[3], s] for (b, s, _lbl) in detections]
    arr = np.asarray(rows, dtype=np.float32)
    if as_torch:
        try:
            import torch  # type: ignore

            return torch.from_numpy(arr)
        except ImportError:
            return arr
    return arr


def _parse_tracker_output(
    tracked: Any,
    detections: list[tuple[tuple[float, float, float, float], float, str]],
    *,
    frame_idx: int,
    timestamp_s: float,
):
    """Normalize a tracker's return value to a list of TrackBox.

    Handles two real-world shapes:

    1. ``ndarray`` rows ``[x1, y1, x2, y2, track_id, class_id, score]`` —
       used by bytetracker 0.3.x and ocsort 0.0.x.
    2. List of objects with ``.track_id`` and ``.tlbr`` — used by some
       newer ByteTrack forks.
    """
    from visionservex.runtime.simple_tracker import TrackBox

    results: list[Any] = []

    # Case A: ndarray-like with rows.
    try:
        rows = list(tracked)
    except TypeError:
        return results
    if not rows:
        return results

    label_lookup = [det[2] if len(det) > 2 else "object" for det in detections]
    default_label = label_lookup[0] if label_lookup else "object"

    for row in rows:
        # Branch 1: ndarray / list-of-floats row.
        try:
            iter_len = len(row)
        except TypeError:
            iter_len = 0
        if iter_len >= 5 and not hasattr(row, "track_id"):
            try:
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                tid = int(row[4])
                score = float(row[6]) if iter_len >= 7 else 0.9
            except (IndexError, TypeError, ValueError):
                continue
            results.append(
                TrackBox(
                    track_id=tid,
                    box=(x1, y1, x2, y2),
                    score=score,
                    label=default_label,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
            continue
        # Branch 2: object with .track_id / .tlbr.
        if hasattr(row, "track_id") and hasattr(row, "tlbr"):
            try:
                tid = int(row.track_id)
                tlbr = row.tlbr
                score = float(getattr(row, "score", 0.9))
            except (AttributeError, TypeError, ValueError):
                continue
            results.append(
                TrackBox(
                    track_id=tid,
                    box=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                    score=score,
                    label=default_label,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            )
    return results


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

        self._call_style = "frame_idx"  # bytetracker 0.3.x: update(dets, frame_idx)
        try:
            # Real installed bytetracker 0.3.2 uses BYTETracker() no-arg ctor.
            self._tracker = BYTETracker()
        except TypeError:
            try:
                self._tracker = BYTETracker(_Args(), frame_rate=kwargs.get("frame_rate", 30))
                self._call_style = "img_size"
            except TypeError:
                try:
                    self._tracker = BYTETracker(_Args())
                    self._call_style = "raw"
                except Exception as exc:
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
        if not detections:
            self._frame_count += 1
            return []

        # bytetracker 0.3.x: 6-col torch.Tensor + frame_idx.
        for n_cols, as_torch, call_kind in (
            (6, True, "frame_idx"),
            (5, False, "img_size"),
            (5, False, "raw"),
        ):
            dets = _detections_to_tensor_or_array(detections, n_cols=n_cols, as_torch=as_torch)
            try:
                if call_kind == "frame_idx":
                    tracked = self._tracker.update(dets, frame_idx)
                elif call_kind == "img_size":
                    tracked = self._tracker.update(dets, img_size=img_size, img_info=img_size)
                else:
                    tracked = self._tracker.update(dets)
                self._frame_count += 1
                return _parse_tracker_output(
                    tracked,
                    detections,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            except (TypeError, IndexError, AttributeError):
                continue

        # All call styles failed — surface the API mismatch.
        import bytetracker as _bt  # type: ignore

        raise TrackerAPIUnsupportedError(
            "bytetrack",
            "BYTETRACK_API_UNSUPPORTED",
            getattr(_bt, "__version__", None),
            [a for a in dir(_bt) if not a.startswith("_")][:20],
        )


# ---------------------------------------------------------------------------
# OC-SORT
# ---------------------------------------------------------------------------


class _OCSortAdapter:
    """Wrap the upstream ocsort package to a uniform interface.

    Real ocsort 0.0.2 wants ``update(torch_tensor_6cols, frame_idx)`` and
    returns ``ndarray[ [x1, y1, x2, y2, track_id, class_id, score], ... ]``.
    Some newer forks expose ``update(dets, img_info, img_size)`` — both are
    tried below.
    """

    def __init__(self, **kwargs: Any) -> None:
        try:
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

        # ocsort 0.0.2 silently fails on min_hits=3 with one detection;
        # fall back to a lower min_hits if the caller didn't override.
        ctor_kwargs = {
            "det_thresh": kwargs.get("det_thresh", 0.3),
            "max_age": kwargs.get("max_age", 30),
            "min_hits": kwargs.get("min_hits", 1),
            "iou_threshold": kwargs.get("iou_threshold", 0.3),
        }
        try:
            self._tracker = OCSort(**ctor_kwargs)
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
        if not detections:
            self._frame_count += 1
            return []

        for n_cols, as_torch, call_kind in (
            (6, True, "frame_idx"),
            (5, False, "img_size"),
            (5, False, "raw"),
        ):
            dets = _detections_to_tensor_or_array(detections, n_cols=n_cols, as_torch=as_torch)
            try:
                if call_kind == "frame_idx":
                    tracked = self._tracker.update(dets, frame_idx)
                elif call_kind == "img_size":
                    tracked = self._tracker.update(dets, img_info=img_size, img_size=img_size)
                else:
                    tracked = self._tracker.update(dets)
                self._frame_count += 1
                return _parse_tracker_output(
                    tracked,
                    detections,
                    frame_idx=frame_idx,
                    timestamp_s=timestamp_s,
                )
            except (TypeError, IndexError, AttributeError):
                continue

        import ocsort as _oc  # type: ignore

        raise TrackerAPIUnsupportedError(
            "ocsort",
            "OCSORT_API_UNSUPPORTED",
            getattr(_oc, "__version__", None),
            [a for a in dir(_oc) if not a.startswith("_")][:20],
        )


__all__ = [
    "TrackerAPIUnsupportedError",
    "TrackerUnavailableError",
    "build_tracker",
    "list_trackers",
]

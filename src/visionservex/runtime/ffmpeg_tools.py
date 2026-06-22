# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Worker-side ffmpeg / ffprobe video toolkit (v3.22.0).

Provides container/codec diagnostics, **lossless** faststart remux, safe
browser-compatible H.264 transcode (NVENC when available), and hardware-path
detection — so users no longer run ffmpeg by hand.

Security model
--------------
* ffmpeg/ffprobe argument vectors are built ENTIRELY from controlled presets and
  numeric parameters that are range-validated here. **No string from a request is
  ever interpolated into an ffmpeg argument.** Input/output are filesystem paths
  passed as argv elements (never a shell string; ``shell=False`` always).
* Hard caps on input size / duration / resolution / fps are enforced before any
  heavy operation (:func:`enforce_limits`).
* Callers are expected to sandbox temp files (see ``runtime.temp_manager``).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# Hard safety caps (overridable by callers/config).
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
MAX_DURATION_S = 60 * 60  # 1 hour
MAX_WIDTH = 7680
MAX_HEIGHT = 4320
MAX_FPS = 240

# Allowed transcode presets (height → settings). Preset-only: no free-form args.
BROWSER_PRESETS: dict[str, dict[str, Any]] = {
    "480p": {"height": 480, "crf": 23, "nvenc_cq": 25},
    "720p": {"height": 720, "crf": 23, "nvenc_cq": 25},
    "1080p": {"height": 1080, "crf": 21, "nvenc_cq": 23},
    "source": {"height": 0, "crf": 21, "nvenc_cq": 23},  # keep source resolution
}


class FFmpegUnavailableError(RuntimeError):
    """Raised when ffmpeg/ffprobe is not installed."""


class VideoLimitError(ValueError):
    """Raised when an input video exceeds a configured hard cap."""


@dataclass
class VideoProbe:
    path: str
    exists: bool = False
    ok: bool = False
    size_bytes: int = 0
    container: str | None = None
    video_codec: str | None = None
    profile: str | None = None
    level: int | None = None
    pix_fmt: str | None = None
    bit_depth: int | None = None
    chroma_subsampling: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    nb_frames: int | None = None
    overall_bitrate: int | None = None
    has_audio: bool = False
    audio_codec: str | None = None
    faststart: bool | None = None  # True = moov before mdat (web-streamable)
    top_level_atoms: list[str] = field(default_factory=list)
    browser_compatible: bool = False  # codec+pixfmt+faststart all web-safe
    recommended_action: str = "unknown"  # none | remux_faststart | transcode | unsupported
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_tools() -> None:
    if not FFMPEG or not FFPROBE:
        raise FFmpegUnavailableError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg (apt: ffmpeg) to use "
            "the worker video pipeline."
        )


def _pixfmt_depth_chroma(pix_fmt: str | None) -> tuple[int | None, str | None]:
    """Best-effort 8/10-bit + chroma classification from a pix_fmt name."""
    if not pix_fmt:
        return None, None
    depth = 10 if ("10" in pix_fmt or "p10" in pix_fmt) else (12 if "12" in pix_fmt else 8)
    if "420" in pix_fmt:
        chroma = "4:2:0"
    elif "422" in pix_fmt:
        chroma = "4:2:2"
    elif "444" in pix_fmt:
        chroma = "4:4:4"
    else:
        chroma = None
    return depth, chroma


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    try:
        if "/" in rate:
            num, den = rate.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(rate)
    except Exception:
        return None


def _moov_before_mdat(path: str) -> tuple[bool | None, list[str]]:
    """Scan top-level MP4 atoms; return (moov_before_mdat, atom_order)."""
    order: list[str] = []
    try:
        with open(path, "rb") as f:
            while len(order) < 24:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                size = struct.unpack(">I", hdr[:4])[0]
                typ = hdr[4:8].decode("latin1", errors="replace")
                order.append(typ)
                if size == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        break
                    size = struct.unpack(">Q", ext)[0]
                    f.seek(size - 16, 1)
                elif size == 0:
                    break
                else:
                    f.seek(size - 8, 1)
    except Exception:
        return None, order
    if "moov" in order and "mdat" in order:
        return order.index("moov") < order.index("mdat"), order
    return None, order


def probe_video(path: str) -> VideoProbe:
    """Run ffprobe + atom scan and classify the recommended worker action."""
    p = VideoProbe(path=str(path))
    if not os.path.exists(path):
        p.errors.append("file not found")
        return p
    p.exists = True
    p.size_bytes = os.path.getsize(path)
    _require_tools()

    try:
        out = subprocess.run(
            [
                FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        data = json.loads(out.stdout or "{}")
    except Exception as exc:
        p.errors.append(f"ffprobe failed: {str(exc)[:160]}")
        return p

    fmt = data.get("format", {})
    p.container = fmt.get("format_name")
    p.duration_s = float(fmt["duration"]) if fmt.get("duration") else None
    p.overall_bitrate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None

    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and p.video_codec is None:
            p.video_codec = s.get("codec_name")
            p.profile = s.get("profile")
            p.level = s.get("level")
            p.pix_fmt = s.get("pix_fmt")
            p.width = s.get("width")
            p.height = s.get("height")
            p.fps = _parse_fps(s.get("r_frame_rate"))
            p.nb_frames = int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None
            p.bit_depth, p.chroma_subsampling = _pixfmt_depth_chroma(p.pix_fmt)
        elif s.get("codec_type") == "audio" and not p.has_audio:
            p.has_audio = True
            p.audio_codec = s.get("codec_name")

    fast, atoms = _moov_before_mdat(path)
    p.faststart = fast
    p.top_level_atoms = atoms
    p.ok = p.video_codec is not None

    # Classify recommended action.
    web_codec = p.video_codec == "h264"
    web_pix = p.pix_fmt in {"yuv420p", "yuvj420p"}
    p.browser_compatible = bool(web_codec and web_pix and fast)
    if not p.ok:
        p.recommended_action = "unsupported"
    elif p.browser_compatible:
        p.recommended_action = "none"
    elif web_codec and web_pix and fast is False:
        p.recommended_action = "remux_faststart"  # only moov misplaced
    else:
        p.recommended_action = "transcode"  # codec/pixfmt not web-safe
    return p


def enforce_limits(
    probe: VideoProbe,
    *,
    max_bytes: int = MAX_VIDEO_BYTES,
    max_duration_s: float = MAX_DURATION_S,
    max_width: int = MAX_WIDTH,
    max_height: int = MAX_HEIGHT,
    max_fps: float = MAX_FPS,
) -> None:
    """Raise :class:`VideoLimitError` if the probed video exceeds any cap."""
    if probe.size_bytes > max_bytes:
        raise VideoLimitError(f"file too large: {probe.size_bytes} > {max_bytes} bytes")
    if probe.duration_s and probe.duration_s > max_duration_s:
        raise VideoLimitError(f"duration too long: {probe.duration_s:.0f}s > {max_duration_s}s")
    if probe.width and probe.width > max_width:
        raise VideoLimitError(f"width too large: {probe.width} > {max_width}")
    if probe.height and probe.height > max_height:
        raise VideoLimitError(f"height too large: {probe.height} > {max_height}")
    if probe.fps and probe.fps > max_fps:
        raise VideoLimitError(f"fps too high: {probe.fps} > {max_fps}")


@lru_cache(maxsize=1)
def detect_hwaccel() -> dict[str, Any]:
    """Detect NVENC/NVDEC availability by querying ffmpeg's encoder/decoder lists."""
    info: dict[str, Any] = {
        "ffmpeg": bool(FFMPEG),
        "ffprobe": bool(FFPROBE),
        "nvenc": [],
        "nvdec_cuvid": [],
        "nvenc_available": False,
        "nvdec_available": False,
    }
    if not FFMPEG:
        return info
    try:
        enc = subprocess.run(
            [FFMPEG, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
        dec = subprocess.run(
            [FFMPEG, "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout
        info["nvenc"] = [c for c in ("h264_nvenc", "hevc_nvenc", "av1_nvenc") if c in enc]
        info["nvdec_cuvid"] = [
            c for c in ("h264_cuvid", "hevc_cuvid", "vp9_cuvid", "av1_cuvid") if c in dec
        ]
        info["nvenc_available"] = bool(info["nvenc"])
        info["nvdec_available"] = bool(info["nvdec_cuvid"])
    except Exception as exc:
        info["error"] = str(exc)[:160]
    return info


def _run_ffmpeg(args: list[str], *, timeout: float = 1800) -> dict[str, Any]:
    """Run an ffmpeg argv (shell=False). Returns a structured result."""
    _require_tools()
    import time as _t

    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", *args]
    t0 = _t.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    dt = (_t.perf_counter() - t0) * 1000.0
    return {
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed_ms": round(dt, 1),
        "stderr_tail": (proc.stderr or "")[-800:],
        "cmd": cmd,
    }


def remux_faststart(input_path: str, output_path: str, *, timeout: float = 600) -> dict[str, Any]:
    """LOSSLESS remux: move the moov atom to the front (``-c copy``). No re-encode.

    Use when the codec/pixfmt are already browser-safe but moov is at the end.
    """
    args = [
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    res = _run_ffmpeg(args, timeout=timeout)
    res["mode"] = "remux_faststart_lossless"
    if res["ok"] and os.path.exists(output_path):
        res["output_size_bytes"] = os.path.getsize(output_path)
        res["output_faststart"] = _moov_before_mdat(output_path)[0]
    return res


def transcode_browser_h264(
    input_path: str,
    output_path: str,
    *,
    preset: str = "720p",
    use_nvenc: bool | None = None,
    timeout: float = 1800,
) -> dict[str, Any]:
    """Transcode to browser-safe H.264 (yuv420p, avc1 tag, faststart, AAC audio).

    ``preset`` MUST be a key of :data:`BROWSER_PRESETS` (preset-only security).
    ``use_nvenc=None`` auto-selects NVENC when available, else libx264.
    """
    if preset not in BROWSER_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; allowed: {sorted(BROWSER_PRESETS)}")
    cfg = BROWSER_PRESETS[preset]
    hw = detect_hwaccel()
    nvenc = hw["nvenc_available"] if use_nvenc is None else (use_nvenc and hw["nvenc_available"])

    vf = []
    if cfg["height"]:
        # even-dimension safe downscale; -2 keeps aspect & divisibility by 2
        vf.append(f"scale=-2:{int(cfg['height'])}")
    vf.append("format=yuv420p")
    vfilter = ",".join(vf)

    args = ["-i", str(input_path), "-map", "0:v:0", "-map", "0:a?", "-vf", vfilter]
    if nvenc:
        args += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p5",
            "-rc",
            "vbr",
            "-cq",
            str(int(cfg["nvenc_cq"])),
            "-profile:v",
            "high",
        ]
        encoder = "h264_nvenc"
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(int(cfg["crf"])),
            "-profile:v",
            "high",
        ]
        encoder = "libx264"
    args += [
        "-pix_fmt",
        "yuv420p",
        "-tag:v",
        "avc1",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    res = _run_ffmpeg(args, timeout=timeout)
    res["mode"] = "transcode_browser_h264"
    res["encoder"] = encoder
    res["preset"] = preset
    res["nvenc_used"] = nvenc
    if res["ok"] and os.path.exists(output_path):
        res["output_size_bytes"] = os.path.getsize(output_path)
        res["output_faststart"] = _moov_before_mdat(output_path)[0]
    return res


def remux_or_transcode(
    input_path: str, output_path: str, *, preset: str = "720p"
) -> dict[str, Any]:
    """Probe then choose the minimal correct operation (remux if possible, else transcode)."""
    probe = probe_video(input_path)
    enforce_limits(probe)
    action = probe.recommended_action
    if action == "none":
        return {
            "ok": True,
            "mode": "none",
            "reason": "already browser-compatible",
            "probe": probe.to_dict(),
        }
    if action == "remux_faststart":
        res = remux_faststart(input_path, output_path)
    elif action == "transcode":
        res = transcode_browser_h264(input_path, output_path, preset=preset)
    else:
        return {
            "ok": False,
            "mode": "unsupported",
            "reason": "no video stream",
            "probe": probe.to_dict(),
        }
    res["probe"] = probe.to_dict()
    return res


__all__ = [
    "BROWSER_PRESETS",
    "FFmpegUnavailableError",
    "VideoLimitError",
    "VideoProbe",
    "detect_hwaccel",
    "enforce_limits",
    "probe_video",
    "remux_faststart",
    "remux_or_transcode",
    "transcode_browser_h264",
]

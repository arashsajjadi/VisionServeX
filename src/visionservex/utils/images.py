"""Safe image loading helpers.

These helpers enforce dimension and pixel limits to mitigate
decompression-bomb attacks and predictable resource exhaustion.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

# Disable partial-image loading. Truncated images should fail loudly.
ImageFile.LOAD_TRUNCATED_IMAGES = False


class ImageValidationError(ValueError):
    """Raised when an incoming image violates safety limits."""


def open_safe(
    data: bytes | Path | str | io.IOBase,
    *,
    max_pixels: int,
    max_dim: int,
) -> Image.Image:
    """Open an image with bomb-protection limits applied."""
    # Pillow's global guard. Setting per-call to avoid leaking state across modes.
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        if isinstance(data, (bytes, bytearray)):
            buf: io.IOBase = io.BytesIO(bytes(data))
        elif isinstance(data, (str, Path)):
            buf = open(data, "rb")  # noqa: SIM115 - closed via context below
        else:
            buf = data

        try:
            try:
                image = Image.open(buf)
                image.load()  # force decode to catch bombs and truncations
            except (UnidentifiedImageError, OSError) as exc:
                raise ImageValidationError(f"could not decode image: {exc}") from exc

            width, height = image.size
            if width > max_dim or height > max_dim:
                raise ImageValidationError(
                    f"image dimension {width}x{height} exceeds limit {max_dim}"
                )
            if width * height > max_pixels:
                raise ImageValidationError(
                    f"image area {width * height} exceeds max_pixels {max_pixels}"
                )
            return image.convert("RGB") if image.mode != "RGB" else image
        finally:
            if isinstance(data, (str, Path)) and isinstance(buf, io.IOBase):
                buf.close()
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a contiguous H,W,C uint8 array."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def encode_jpeg(image: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def image_size(image: Image.Image) -> tuple[int, int]:
    return image.size


__all__ = [
    "ImageValidationError",
    "encode_jpeg",
    "encode_png",
    "image_size",
    "open_safe",
    "to_numpy",
]

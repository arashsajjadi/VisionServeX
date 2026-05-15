# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Arash Sajjadi
"""Generate the bundled sample images.

Run once to regenerate ``examples/images/*.jpg``. The synthetic images are
intentionally simple so they do not embed any third-party imagery and stay
small in the source tree.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def make_simple_shapes(path: Path) -> None:
    img = Image.new("RGB", (640, 480), color=(200, 220, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 80, 260, 280], outline=(20, 60, 120), width=4, fill=(80, 130, 200))
    d.ellipse([320, 100, 540, 320], outline=(120, 20, 60), width=4, fill=(220, 100, 120))
    d.polygon([(150, 380), (250, 380), (200, 460)], outline=(40, 100, 40), fill=(120, 200, 120))
    d.text((10, 10), "VisionServeX sample: simple shapes", fill=(20, 20, 20), font=_font(14))
    img.save(path, "JPEG", quality=88)


def make_street_like(path: Path) -> None:
    img = Image.new("RGB", (640, 480), color=(180, 200, 220))
    d = ImageDraw.Draw(img)
    # Ground
    d.rectangle([0, 320, 640, 480], fill=(120, 120, 130))
    # Cars
    for x in (60, 220, 400):
        d.rectangle([x, 300, x + 140, 360], outline=(0, 0, 0), width=2, fill=(220, 80, 80))
        d.ellipse([x + 10, 350, x + 40, 380], fill=(20, 20, 20))
        d.ellipse([x + 100, 350, x + 130, 380], fill=(20, 20, 20))
    # Person
    d.ellipse([300, 200, 330, 230], fill=(255, 220, 180))
    d.rectangle([305, 230, 325, 290], fill=(60, 60, 120))
    d.text((10, 10), "VisionServeX sample: street-like", fill=(0, 0, 0), font=_font(14))
    img.save(path, "JPEG", quality=88)


def make_dog_like(path: Path) -> None:
    img = Image.new("RGB", (480, 480), color=(240, 230, 200))
    d = ImageDraw.Draw(img)
    d.ellipse([100, 200, 380, 380], outline=(120, 80, 40), width=3, fill=(180, 130, 80))  # body
    d.ellipse([300, 140, 440, 280], outline=(120, 80, 40), width=3, fill=(180, 130, 80))  # head
    d.polygon([(320, 150), (340, 110), (360, 150)], fill=(150, 100, 60))                   # ear
    d.ellipse([350, 190, 370, 210], fill=(20, 20, 20))                                     # eye
    d.text((10, 10), "VisionServeX sample: dog-like (not a real dog)",
           fill=(40, 40, 40), font=_font(14))
    img.save(path, "JPEG", quality=88)


def main() -> None:
    here = Path(__file__).resolve().parent
    make_simple_shapes(here / "simple_shapes.jpg")
    make_street_like(here / "street.jpg")
    make_dog_like(here / "dog.jpg")
    print("regenerated:", [p.name for p in here.glob("*.jpg")])


if __name__ == "__main__":
    main()

"""Crop a region out of an original figure PDF at high resolution.

Reference-only helper for reading small labels in the source figures.

Usage: crop_reference.py <figure_number> <x0> <y0> <x1> <y1> [scale]
Fractions are relative to the page (0-1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium

OUT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python/_reference")


def main() -> None:
    n = sys.argv[1]
    x0, y0, x1, y1 = (float(v) for v in sys.argv[2:6])
    scale = float(sys.argv[6]) if len(sys.argv) > 6 else 6.0
    pdf = Path(f"F:/EBV/EBV_xiaomi/Figure{n}/Figure{n}.pdf")
    doc = pdfium.PdfDocument(pdf)
    img = doc[0].render(scale=scale).to_pil()
    w, h = img.size
    box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
    crop = img.crop(box)
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"Figure{n}_crop.png"
    crop.save(target)
    print(f"{target}  {crop.width}x{crop.height}")


if __name__ == "__main__":
    main()

"""Rasterise the original assembled figure PDFs for visual reference only.

These previews are never used as figure content; they only let the rebuild be
checked against the original panel composition.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium

SRC = Path("F:/EBV/EBV_xiaomi")
OUT = Path("F:/EBV/EBV_xiaomi/Figures_Nature_Python/_reference")


def render(pdf_path: Path, stem: str, scale: float = 1.6) -> None:
    doc = pdfium.PdfDocument(pdf_path)
    for i in range(len(doc)):
        page = doc[i]
        img = page.render(scale=scale).to_pil()
        suffix = "" if len(doc) == 1 else f"_p{i + 1}"
        target = OUT / f"{stem}{suffix}.png"
        img.save(target)
        print(f"{target.name}  {img.width}x{img.height}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or [str(n) for n in range(1, 11)]
    for n in targets:
        folder = SRC / f"Figure{n}"
        pdf = folder / f"Figure{n}.pdf"
        if not pdf.exists():
            print(f"Figure{n}: no assembled PDF at {pdf}")
            continue
        render(pdf, f"Figure{n}_original")


if __name__ == "__main__":
    main()

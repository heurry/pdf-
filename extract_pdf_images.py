"""
Extract images from a PDF, add a white border, and skip images that are too small.

Usage:
    python extract_pdf_images.py input.pdf out_dir --border 20 --min-width 200 --min-height 200 --min-area 40000

Dependencies:
    pip install pymupdf pillow
"""

import argparse
import io
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps


def should_keep_image(img: Image.Image, min_size: Tuple[int, int], min_area: int) -> bool:
    """Return True if the image meets width/height and area thresholds."""
    width_ok = img.width >= min_size[0]
    height_ok = img.height >= min_size[1]
    area_ok = img.width * img.height >= min_area
    return area_ok


def extract_images(
    pdf_path: Path,
    out_dir: Path,
    border: int = 20,
    min_size: Tuple[int, int] = (200, 200),
    min_area: int = 40_000,
) -> int:
    """
    Extract images from a PDF and save them with a white border.

    Args:
        pdf_path: Path to the input PDF.
        out_dir: Output directory for saved images.
        border: Border width in pixels to add around each image.
        min_size: Minimum (width, height) in pixels for an image to be kept.
        min_area: Minimum area (width * height) in pixels for an image to be kept.

    Returns:
        The number of images written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    seen_xrefs: Set[int] = set()  # Avoid saving duplicate xrefs across pages.

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                extracted = doc.extract_image(xref)
                img_bytes = extracted["image"]
                img_ext = extracted["ext"]
                img = Image.open(io.BytesIO(img_bytes))

                if not should_keep_image(img, min_size, min_area):
                    continue

                if border > 0:
                    img = ImageOps.expand(img, border=border, fill="white")

                filename = f"page{page_index:03d}_img{image_index:03d}.{img_ext}"
                img.save(out_dir / filename)
                written += 1

    return written


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract images from a PDF with borders and size filtering.")
    parser.add_argument("pdf", type=Path, help="Input PDF file.")
    parser.add_argument("out_dir", type=Path, help="Directory to save extracted images.")
    parser.add_argument("--border", type=int, default=20, help="Border width in pixels (default: 20).")
    parser.add_argument("--min-width", type=int, default=200, help="Minimum width in pixels to keep an image.")
    parser.add_argument("--min-height", type=int, default=200, help="Minimum height in pixels to keep an image.")
    parser.add_argument(
        "--min-area",
        type=int,
        default=14400,
        help="Minimum area (width*height) in pixels to keep an image.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    count = extract_images(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        border=args.border,
        min_size=(args.min_width, args.min_height),
        min_area=args.min_area,
    )
    print(f"Saved {count} image(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

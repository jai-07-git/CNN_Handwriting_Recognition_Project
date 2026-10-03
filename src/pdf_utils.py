"""
Render each page of an uploaded PDF to a BGR numpy array, so the same
segmentation + classification pipeline used for photos can run on PDF pages.
Uses PyMuPDF (fitz) — pure Python wheel, no external Poppler/ImageMagick
binary required, which keeps Windows setup simple.
"""
import cv2
import numpy as np

try:
    import pymupdf as fitz  # the modern import name (pymupdf >= 1.24)
except ImportError:  # pragma: no cover - fallback for older pymupdf releases
    import fitz

from config import PDF_RENDER_DPI


def pdf_to_images(file_bytes: bytes, dpi: int = PDF_RENDER_DPI):
    """Return a list of BGR numpy arrays, one per page, in page order."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    images = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            images.append(arr)
    finally:
        doc.close()

    return images
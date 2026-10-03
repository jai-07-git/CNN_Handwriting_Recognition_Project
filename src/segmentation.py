"""
Segment a photo or scanned page containing multiple digits/characters (possibly
across several lines) into individual character crops, in reading order, so
each one can be classified separately by the existing single-character models.

Approach: binarize -> light dilation to merge strokes of one character
(e.g. the dot and stem of 'i') -> connected-component bounding boxes ->
cluster boxes into text lines by vertical overlap -> sort each line
left-to-right -> flag large horizontal gaps as word spaces.
"""
import cv2
import numpy as np


def _binarize(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)
    return thresh


def segment_characters(img_bgr: np.ndarray, min_area: int = 40, pad_ratio: float = 0.25):
    """
    Returns a list of lines (top-to-bottom); each line is a list of dicts,
    left-to-right, each:
        {
          "bbox": (x, y, w, h),           # in original image coordinates
          "crop": <BGR ndarray>,           # padded crop ready for preprocess_image()
          "space_before": bool,            # True if there's a word gap before this char
        }
    Returns [] if no character-like regions are found.
    """
    img_h, img_w = img_bgr.shape[:2]
    thresh = _binarize(img_bgr)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] >= min_area]
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1])
    avg_h = sum(b[3] for b in boxes) / len(boxes)

    lines = []
    for box in boxes:
        _, y, _, h = box
        cy = y + h / 2
        placed = False
        for line in lines:
            if abs(cy - line["cy"]) < avg_h * 0.6:
                n = len(line["boxes"])
                line["cy"] = (line["cy"] * n + cy) / (n + 1)
                line["boxes"].append(box)
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "boxes": [box]})

    lines.sort(key=lambda l: l["cy"])
    for line in lines:
        line["boxes"].sort(key=lambda b: b[0])

    result = []
    for line in lines:
        boxes_sorted = line["boxes"]
        median_w = sorted(b[2] for b in boxes_sorted)[len(boxes_sorted) // 2]
        entries = []
        prev_right = None
        for (x, y, w, h) in boxes_sorted:
            pad = int(max(w, h) * pad_ratio)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
            crop = img_bgr[y0:y1, x0:x1]

            space_before = prev_right is not None and (x - prev_right) > median_w * 1.3
            entries.append({"bbox": (x, y, w, h), "crop": crop, "space_before": space_before})
            prev_right = x + w
        result.append(entries)

    return result
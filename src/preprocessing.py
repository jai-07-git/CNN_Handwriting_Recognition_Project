"""
Preprocessing pipeline used at inference time (canvas drawing or uploaded image)
so that user input matches the MNIST/EMNIST training distribution:

    grayscale -> noise reduction -> Otsu threshold -> crop -> center -> resize 28x28 -> normalize
"""
import base64
import re
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from config import IMG_SIZE


def data_url_to_image(data_url: str) -> np.ndarray:
    """Decode a base64 data URL (from <canvas>.toDataURL() or a file upload) to a BGR numpy array."""
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline. Returns a (1, 28, 28, 1) float32 array in [0, 1],
    with the stroke as bright pixels on a dark background (MNIST/EMNIST style).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blurred = cv2.medianBlur(gray, 5)

    # Otsu threshold. THRESH_BINARY_INV assumes dark strokes on a light background
    # (typical for a canvas drawing or a photographed page).
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If that guess was wrong (input was already light-on-dark), flip it back.
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Crop to the stroke's bounding box
    coords = cv2.findNonZero(thresh)
    if coords is None:
        # Nothing drawn — return a blank canvas rather than raising
        return np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y : y + h, x : x + w]

    # Pad to a square, centering the stroke, with a small margin (~20%)
    side = max(w, h)
    margin = max(int(side * 0.2), 4)
    side += margin * 2
    square = np.zeros((side, side), dtype=np.uint8)
    y_off, x_off = (side - h) // 2, (side - w) // 2
    square[y_off : y_off + h, x_off : x_off + w] = cropped

    # Resize to model input size and normalize
    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0

    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

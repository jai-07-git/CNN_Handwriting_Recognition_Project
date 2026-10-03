"""
Flask web application.

Two recognition features, both private per signed-in Google user:
  1. Single character/digit — draw on a canvas or upload one cropped image.
  2. Whole document — upload a photo or PDF containing many mixed digits and
     characters; the page is segmented into individual characters, each is
     classified, and the reconstructed text plus per-character confidence is
     returned and stored.

Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from flask_login import current_user, login_required

import config
from auth import init_auth
from config import (
    ALLOWED_DOCUMENT_EXTENSIONS,
    CHAR_CLASSES,
    CHAR_MODEL_PATH,
    DIGIT_CLASSES,
    DIGIT_MODEL_PATH,
    MAX_UPLOAD_MB,
)
from database import (
    get_document_history,
    get_history,
    init_db,
    log_document,
    log_prediction,
)
from src.pdf_utils import pdf_to_images
from src.preprocessing import data_url_to_image, preprocess_image
from src.segmentation import segment_characters

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["GOOGLE_CLIENT_ID"] = config.GOOGLE_CLIENT_ID
app.config["GOOGLE_CLIENT_SECRET"] = config.GOOGLE_CLIENT_SECRET
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") == "production"

init_db()
init_auth(app)

_models = {}


def get_model(mode: str):
    if mode not in _models:
        path = DIGIT_MODEL_PATH if mode == "digit" else CHAR_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained {mode} model found at {path}. "
                f"Run train_{mode}_model.py first."
            )
        _models[mode] = tf.keras.models.load_model(path)
    return _models[mode]


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=current_user)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("mode", "digit")
    image_data = payload.get("image")

    if mode not in ("digit", "char"):
        return jsonify({"error": "mode must be 'digit' or 'char'"}), 400
    if not image_data:
        return jsonify({"error": "no image data provided"}), 400

    class_names = DIGIT_CLASSES if mode == "digit" else CHAR_CLASSES

    try:
        img_bgr = data_url_to_image(image_data)
        x = preprocess_image(img_bgr)
    except Exception as exc:
        return jsonify({"error": f"could not process image: {exc}"}), 400

    try:
        model = get_model(mode)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    start = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"label": class_names[i], "confidence": float(probs[i])} for i in top3_idx]

    result = {
        "mode": mode,
        "predicted_class": top3[0]["label"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
        "inference_time_ms": round(inference_time_ms, 2),
    }

    try:
        log_prediction(
            user_id=current_user.id,
            mode=result["mode"],
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            top3=top3,
            inference_time_ms=result["inference_time_ms"],
        )
    except Exception as exc:
        app.logger.warning(f"Failed to log prediction: {exc}")

    return jsonify(result)


@app.route("/history")
@login_required
def history():
    limit = int(request.args.get("limit", 20))
    return jsonify(get_history(current_user.id, limit))


def _classify_crop(crop_bgr, mode: str):
    """
    Classify one segmented character crop.
    mode == 'digit' or 'char': use only that model.
    mode == 'mixed': run both models and keep whichever is more confident.
    Returns (label, type_used, confidence).
    """
    x = preprocess_image(crop_bgr)
    candidates = []

    if mode in ("digit", "mixed"):
        probs = get_model("digit").predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        candidates.append(("digit", DIGIT_CLASSES[i], float(probs[i])))

    if mode in ("char", "mixed"):
        probs = get_model("char").predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        candidates.append(("char", CHAR_CLASSES[i], float(probs[i])))

    candidates.sort(key=lambda c: c[2], reverse=True)
    best_type, best_label, best_conf = candidates[0]
    return best_label, best_type, best_conf


@app.route("/predict-document", methods=["POST"])
@login_required
def predict_document():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "no file uploaded"}), 400

    file = request.files["file"]
    mode = request.form.get("mode", "mixed")
    if mode not in ("mixed", "digit", "char"):
        return jsonify({"error": "mode must be 'mixed', 'digit' or 'char'"}), 400

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({
            "error": f"unsupported file type '{ext}'. Allowed: "
                     f"{', '.join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))}"
        }), 400

    file_bytes = file.read()
    is_pdf = ext == ".pdf"

    try:
        if is_pdf:
            pages = pdf_to_images(file_bytes)
            source_type = "pdf"
        else:
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "could not decode image"}), 400
            pages = [img]
            source_type = "image"
    except Exception as exc:
        return jsonify({"error": f"could not read file: {exc}"}), 400

    try:
        if mode in ("digit", "mixed"):
            get_model("digit")
        if mode in ("char", "mixed"):
            get_model("char")
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    start = time.perf_counter()
    all_chars = []
    text_lines = []
    confidences = []

    for page_num, page_img in enumerate(pages):
        lines = segment_characters(page_img)
        for line_num, line in enumerate(lines):
            line_text = ""
            for idx, entry in enumerate(line):
                label, char_type, conf = _classify_crop(entry["crop"], mode)

                if entry["space_before"]:
                    line_text += " "
                line_text += label
                confidences.append(conf)

                x, y, w, h = entry["bbox"]
                all_chars.append({
                    "page": page_num,
                    "line": line_num,
                    "index": idx,
                    "label": label,
                    "type": char_type,
                    "confidence": round(conf, 4),
                    "bbox": [int(x), int(y), int(w), int(h)],
                })
            text_lines.append(line_text)
        if len(pages) > 1:
            text_lines.append("")

    processing_time_ms = (time.perf_counter() - start) * 1000
    recognized_text = "\n".join(text_lines).rstrip()
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    if not all_chars:
        return jsonify({
            "error": "No characters were detected in this file. Try a clearer, "
                     "higher-contrast image with darker strokes."
        }), 422

    doc_id = None
    try:
        doc_id = log_document(
            user_id=current_user.id,
            source_type=source_type,
            filename=filename,
            mode=mode,
            page_count=len(pages),
            recognized_text=recognized_text,
            avg_confidence=avg_confidence,
            processing_time_ms=round(processing_time_ms, 2),
            details=all_chars,
        )
    except Exception as exc:
        app.logger.warning(f"Failed to log document prediction: {exc}")

    return jsonify({
        "document_id": doc_id,
        "source_type": source_type,
        "page_count": len(pages),
        "recognized_text": recognized_text,
        "avg_confidence": round(avg_confidence, 4),
        "processing_time_ms": round(processing_time_ms, 2),
        "character_count": len(all_chars),
        "characters": all_chars,
    })


@app.route("/document-history")
@login_required
def document_history():
    limit = int(request.args.get("limit", 10))
    return jsonify(get_document_history(current_user.id, limit))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
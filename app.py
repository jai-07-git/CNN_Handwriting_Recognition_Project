"""
Flask web application: accepts a drawn (canvas) or uploaded handwritten image,
runs it through the appropriate CNN (digit or character model), and returns the
predicted class, confidence, Top-3 candidates and inference time. Every
prediction is logged to SQLite.

Run:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import time

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

from config import CHAR_CLASSES, CHAR_MODEL_PATH, DIGIT_CLASSES, DIGIT_MODEL_PATH
from database import get_history, init_db, log_prediction
from src.cnn_model import SpatialAttention
from src.preprocessing import data_url_to_image, preprocess_image


app = Flask(__name__)
@app.route("/health")
def health():
    return jsonify({"status": "ok"})

_models = {}  # lazy-loaded, cached Keras models


def get_model(mode: str):
    if mode not in _models:
        path = DIGIT_MODEL_PATH if mode == "digit" else CHAR_MODEL_PATH

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No trained {mode} model found at {path}. "
                f"Run train_{mode}_model.py first."
            )

        # Digit model contains the custom SpatialAttention layer.
        if mode == "digit":
            _models[mode] = tf.keras.models.load_model(
                path,
                custom_objects={
                    "SpatialAttention": SpatialAttention
                },
                safe_mode=False
            )

        # Character model is loaded normally.
        else:
            _models[mode] = tf.keras.models.load_model(
                path,
                safe_mode=False
            )

    return _models[mode]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}

    mode = payload.get("mode", "digit")
    image_data = payload.get("image")

    if mode not in ("digit", "char"):
        return jsonify({
            "error": "mode must be 'digit' or 'char'"
        }), 400

    if not image_data:
        return jsonify({
            "error": "no image data provided"
        }), 400

    class_names = (
        DIGIT_CLASSES
        if mode == "digit"
        else CHAR_CLASSES
    )

    # -----------------------------------------
    # IMAGE PROCESSING
    # -----------------------------------------

    try:
        img_bgr = data_url_to_image(image_data)
        x = preprocess_image(img_bgr)

    except Exception as exc:
        return jsonify({
            "error": f"could not process image: {exc}"
        }), 400

    # -----------------------------------------
    # LOAD MODEL
    # -----------------------------------------

    try:
        model = get_model(mode)

    except FileNotFoundError as exc:
        return jsonify({
            "error": str(exc)
        }), 503

    except Exception as exc:
        return jsonify({
            "error": f"could not load {mode} model: {exc}"
        }), 500

    # -----------------------------------------
    # PREDICTION
    # -----------------------------------------

    start = time.perf_counter()

    probs = model.predict(
        x,
        verbose=0
    )[0]

    inference_time_ms = (
        time.perf_counter() - start
    ) * 1000

    # -----------------------------------------
    # TOP 3 PREDICTIONS
    # -----------------------------------------

    top3_idx = np.argsort(probs)[::-1][:3]

    top3 = [
        {
            "label": class_names[i],
            "confidence": float(probs[i])
        }
        for i in top3_idx
    ]

    # -----------------------------------------
    # RESULT
    # -----------------------------------------

    result = {
        "mode": mode,
        "predicted_class": top3[0]["label"],
        "confidence": top3[0]["confidence"],
        "top3": top3,
        "inference_time_ms": round(
            inference_time_ms,
            2
        ),
    }

    # -----------------------------------------
    # SAVE PREDICTION TO DATABASE
    # -----------------------------------------

    log_prediction(
        mode=result["mode"],
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        top3=top3,
        inference_time_ms=result["inference_time_ms"],
    )

    return jsonify(result)


@app.route("/history")
def history():
    limit = int(
        request.args.get(
            "limit",
            20
        )
    )

    return jsonify(
        get_history(limit)
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
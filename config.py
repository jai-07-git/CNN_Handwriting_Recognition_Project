"""
Central configuration for the CNN Handwritten Digit & Character Recognition project.
"""
import os

# --- Image / class settings -------------------------------------------------
IMG_SIZE = 28
DIGIT_CLASSES = [str(i) for i in range(10)]              # 0-9
CHAR_CLASSES = [chr(ord("A") + i) for i in range(26)]     # A-Z

# --- Paths -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DIGIT_MODEL_PATH = os.path.join(MODEL_DIR, "digit_model.keras")
CHAR_MODEL_PATH = os.path.join(MODEL_DIR, "char_model.keras")
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

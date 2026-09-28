"""
Generate evaluation metrics for a trained model: accuracy, precision, recall,
F1-score (via sklearn's classification_report) and a confusion matrix image.

Usage:
    python evaluate_model.py --mode digit
    python evaluate_model.py --mode char
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CHAR_CLASSES, CHAR_MODEL_PATH, DIGIT_CLASSES, DIGIT_MODEL_PATH, IMG_SIZE


def load_digit_test():
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return x_test, y_test


def load_char_test():
    import tensorflow_datasets as tfds

    test_ds = tfds.load("emnist/letters", split="test", as_supervised=True)
    images, labels = [], []
    for img, label in tfds.as_numpy(test_ds):
        img = np.transpose(img.squeeze(), (1, 0))
        images.append(img)
        labels.append(label - 1)
    x_test = np.array(images, dtype="float32") / 255.0
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(labels)
    return x_test, y_test


def evaluate(mode: str):
    if mode == "digit":
        model = tf.keras.models.load_model(DIGIT_MODEL_PATH)
        x_test, y_test = load_digit_test()
        class_names = DIGIT_CLASSES
    else:
        model = tf.keras.models.load_model(CHAR_MODEL_PATH)
        x_test, y_test = load_char_test()
        class_names = CHAR_CLASSES

    y_pred_probs = model.predict(x_test, batch_size=256)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{mode.capitalize()} model confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    plt.yticks(range(len(class_names)), class_names, fontsize=6)
    plt.colorbar()
    plt.tight_layout()
    out_path = f"confusion_matrix_{mode}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["digit", "char"], required=True)
    args = parser.parse_args()
    evaluate(args.mode)

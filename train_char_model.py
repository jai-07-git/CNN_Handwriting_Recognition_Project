import os
import gzip
import struct
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "char_model.keras")
DATA_DIR = "emnist_data"

EPOCHS = 5
BATCH_SIZE = 128


BASE_URL = (
    "https://huggingface.co/datasets/Heliosoph/EMNIST/"
    "resolve/main/"
)

FILES = {
    "train_images": "emnist-letters-train-images-idx3-ubyte.gz",
    "train_labels": "emnist-letters-train-labels-idx1-ubyte.gz",
    "test_images": "emnist-letters-test-images-idx3-ubyte.gz",
    "test_labels": "emnist-letters-test-labels-idx1-ubyte.gz",
}


# ============================================================
# DOWNLOAD DATASET
# ============================================================

def download_file(filename):
    os.makedirs(DATA_DIR, exist_ok=True)

    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        print(f"Already downloaded: {filename}")
        return filepath

    url = BASE_URL + filename

    print()
    print("Downloading:")
    print(url)
    print()

    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)

        raise RuntimeError(
            f"Could not download {filename}\n"
            f"URL: {url}\n"
            f"Error: {e}"
        )

    print(f"Downloaded: {filename}")

    return filepath


# ============================================================
# READ IDX IMAGE FILE
# ============================================================

def load_images(filepath):

    with gzip.open(filepath, "rb") as f:

        header = f.read(16)

        magic, num_images, rows, cols = struct.unpack(
            ">IIII",
            header
        )

        if magic != 2051:
            raise ValueError(
                f"Invalid image file: {filepath}"
            )

        data = f.read()

    images = np.frombuffer(
        data,
        dtype=np.uint8
    )

    images = images.reshape(
        num_images,
        rows,
        cols
    )

    # EMNIST images are stored transposed.
    # Transpose each image to normal orientation.
    images = np.transpose(
        images,
        (0, 2, 1)
    )

    return images


# ============================================================
# READ IDX LABEL FILE
# ============================================================

def load_labels(filepath):

    with gzip.open(filepath, "rb") as f:

        header = f.read(8)

        magic, num_labels = struct.unpack(
            ">II",
            header
        )

        if magic != 2049:
            raise ValueError(
                f"Invalid label file: {filepath}"
            )

        data = f.read()

    labels = np.frombuffer(
        data,
        dtype=np.uint8
    )

    if len(labels) != num_labels:
        raise ValueError(
            "Number of labels does not match header."
        )

    # EMNIST Letters labels are 1-26.
    # Convert them to 0-25 for Keras.
    labels = labels - 1

    return labels


# ============================================================
# LOAD EMNIST LETTERS
# ============================================================

def load_dataset():

    print()
    print("=" * 60)
    print("EMNIST LETTERS DATASET")
    print("=" * 60)

    train_images_path = download_file(
        FILES["train_images"]
    )

    train_labels_path = download_file(
        FILES["train_labels"]
    )

    test_images_path = download_file(
        FILES["test_images"]
    )

    test_labels_path = download_file(
        FILES["test_labels"]
    )

    print()
    print("Loading images...")

    x_train = load_images(
        train_images_path
    )

    x_test = load_images(
        test_images_path
    )

    print("Loading labels...")

    y_train = load_labels(
        train_labels_path
    )

    y_test = load_labels(
        test_labels_path
    )

    print()
    print("Dataset loaded successfully.")

    print(
        "Training images:",
        x_train.shape
    )

    print(
        "Training labels:",
        y_train.shape
    )

    print(
        "Testing images:",
        x_test.shape
    )

    print(
        "Testing labels:",
        y_test.shape
    )

    print(
        "Number of classes:",
        len(np.unique(y_train))
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test
    )


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(
    x_train,
    y_train,
    x_test,
    y_test
):

    x_train = x_train.astype(
        "float32"
    ) / 255.0

    x_test = x_test.astype(
        "float32"
    ) / 255.0

    # Add channel dimension.
    x_train = np.expand_dims(
        x_train,
        axis=-1
    )

    x_test = np.expand_dims(
        x_test,
        axis=-1
    )

    return (
        x_train,
        y_train,
        x_test,
        y_test
    )


# ============================================================
# BUILD CNN MODEL
# ============================================================

def build_model():

    model = models.Sequential([

        layers.Input(
            shape=(28, 28, 1)
        ),

        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Conv2D(
            128,
            (3, 3),
            activation="relu",
            padding="same"
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(
            0.3
        ),

        layers.Dense(
            26,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("CNN HANDWRITTEN CHARACTER RECOGNITION")
    print("=" * 60)

    # Load dataset
    (
        x_train,
        y_train,
        x_test,
        y_test
    ) = load_dataset()

    # Preprocess
    (
        x_train,
        y_train,
        x_test,
        y_test
    ) = preprocess_data(
        x_train,
        y_train,
        x_test,
        y_test
    )

    print()
    print("Building CNN model...")

    model = build_model()

    print()
    model.summary()

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_test,
            y_test
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print()
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print()
    print(
        f"Test Accuracy: {accuracy * 100:.2f}%"
    )

    # Save final model
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save(
        MODEL_PATH
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print()
    print(
        "Character model saved at:"
    )

    print(
        os.path.abspath(
            MODEL_PATH
        )
    )

    print()
    print(
        "Classes: A-Z"
    )

    print(
        "Total classes: 26"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
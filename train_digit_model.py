"""
Train the digit-recognition CNN on MNIST (10 classes, 0-9).

Usage:
    python train_digit_model.py [--epochs 15] [--batch-size 128]
"""
import argparse
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import DIGIT_MODEL_PATH, IMG_SIZE, MODEL_DIR
from src.cnn_model import build_cnn


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return (x_train, y_train), (x_test, y_test)


def main(epochs=15, batch_size=128):
    os.makedirs(MODEL_DIR, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    # Data augmentation: rotation, translation, zoom (per project objectives)
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(x_train)

    model = build_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10, name="digit_cnn")
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ModelCheckpoint(DIGIT_MODEL_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nDigit model — test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}")

    model.save(DIGIT_MODEL_PATH)
    print(f"Saved digit model to {DIGIT_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size)

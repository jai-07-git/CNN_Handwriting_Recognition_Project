"""
CNN backbone enhanced with Residual Blocks and Spatial Attention modules,
used for both the digit (MNIST, 10 classes) and character (EMNIST Letters,
26 classes) models.

Architecture, per slide 8/9 of the project review:
    input -> stem conv
          -> [Residual Block -> Spatial Attention -> MaxPool]  x2
          -> Residual Block -> Spatial Attention
          -> Global Average Pool -> Dense -> Dropout -> Softmax
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def spatial_attention_block(x, name="spatial_attention"):
    """
    Spatial attention: pool across channels (avg + max), learn a single-channel
    attention map with a 7x7 conv + sigmoid, and re-weight the input feature map.
    Lightweight version of the attention idea in CBAM (Woo et al., 2018).
    """
    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name=f"{name}_avgpool"
    )(x)
    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name=f"{name}_maxpool"
    )(x)
    concat = layers.Concatenate(axis=-1, name=f"{name}_concat")([avg_pool, max_pool])
    attention_map = layers.Conv2D(
        1, kernel_size=7, padding="same", activation="sigmoid", name=f"{name}_conv"
    )(concat)
    return layers.Multiply(name=f"{name}_mul")([x, attention_map])


def residual_block(x, filters, name="res_block"):
    """
    Standard 2-conv residual block (He et al., 2016) with a 1x1 projection
    shortcut whenever the channel count changes.
    """
    shortcut = x

    y = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)

    y = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    y = layers.ReLU(name=f"{name}_relu2")(y)
    return y


def build_cnn(input_shape=(28, 28, 1), num_classes=10, name="cnn_model"):
    """Build and return the (uncompiled) Keras functional model."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Stem
    x = layers.Conv2D(32, 3, padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Stage 1: 28x28 -> 14x14
    x = residual_block(x, 32, name="res1")
    x = spatial_attention_block(x, name="sa1")
    x = layers.MaxPooling2D(name="pool1")(x)

    # Stage 2: 14x14 -> 7x7
    x = residual_block(x, 64, name="res2")
    x = spatial_attention_block(x, name="sa2")
    x = layers.MaxPooling2D(name="pool2")(x)

    # Stage 3: 7x7 (no further downsampling)
    x = residual_block(x, 128, name="res3")
    x = spatial_attention_block(x, name="sa3")

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name=name)


if __name__ == "__main__":
    # Quick sanity check: python -m src.cnn_model
    m = build_cnn(num_classes=10, name="digit_cnn")
    m.summary()

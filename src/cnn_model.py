import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="Custom")
class SpatialAttention(layers.Layer):
    """
    Spatial Attention module.

    Performs channel-wise average pooling and maximum pooling,
    concatenates both maps, and learns a spatial attention map.
    """

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid"
        )

    def call(self, inputs):
        # Average across channels
        avg_pool = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        # Maximum across channels
        max_pool = tf.reduce_max(
            inputs,
            axis=-1,
            keepdims=True
        )

        # Combine both attention descriptors
        concat = tf.concat(
            [avg_pool, max_pool],
            axis=-1
        )

        # Learn spatial attention
        attention_map = self.conv(concat)

        # Apply attention
        return inputs * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size
        })
        return config


def residual_block(x, filters, name="residual_block"):
    """
    Standard two-convolution residual block.

    A 1x1 projection is used when the input channel count
    does not match the requested number of filters.
    """

    shortcut = x

    # First convolution
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{name}_conv1"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn1"
    )(x)

    x = layers.ReLU(
        name=f"{name}_relu1"
    )(x)

    # Second convolution
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        name=f"{name}_conv2"
    )(x)

    x = layers.BatchNormalization(
        name=f"{name}_bn2"
    )(x)

    # Check whether shortcut channels match
    input_channels = shortcut.shape[-1]

    if input_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            padding="same",
            use_bias=False,
            name=f"{name}_projection"
        )(shortcut)

        shortcut = layers.BatchNormalization(
            name=f"{name}_projection_bn"
        )(shortcut)

    # Residual addition
    x = layers.Add(
        name=f"{name}_add"
    )([x, shortcut])

    x = layers.ReLU(
        name=f"{name}_relu2"
    )(x)

    return x


def build_cnn(
    input_shape=(28, 28, 1),
    num_classes=10,
    name="cnn"
):
    """
    Build CNN with:

    Input
       ↓
    Stem Conv
       ↓
    Residual Block
       ↓
    Spatial Attention
       ↓
    Max Pool
       ↓
    Residual Block
       ↓
    Spatial Attention
       ↓
    Max Pool
       ↓
    Residual Block
       ↓
    Spatial Attention
       ↓
    Global Average Pool
       ↓
    Dense
       ↓
    Dropout
       ↓
    Softmax
    """

    inputs = layers.Input(
        shape=input_shape,
        name="input_image"
    )

    # --------------------------------------------------
    # STEM
    # --------------------------------------------------

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        use_bias=False,
        name="stem_conv"
    )(inputs)

    x = layers.BatchNormalization(
        name="stem_bn"
    )(x)

    x = layers.ReLU(
        name="stem_relu"
    )(x)

    # --------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------

    x = residual_block(
        x,
        32,
        name="residual_block_1"
    )

    x = SpatialAttention(
        name="spatial_attention_1"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool_1"
    )(x)

    # --------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------

    x = residual_block(
        x,
        64,
        name="residual_block_2"
    )

    x = SpatialAttention(
        name="spatial_attention_2"
    )(x)

    x = layers.MaxPooling2D(
        (2, 2),
        name="pool_2"
    )(x)

    # --------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------

    x = residual_block(
        x,
        128,
        name="residual_block_3"
    )

    x = SpatialAttention(
        name="spatial_attention_3"
    )(x)

    # --------------------------------------------------
    # CLASSIFICATION HEAD
    # --------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="fc1"
    )(x)

    x = layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=name
    )

    return model


if __name__ == "__main__":

    model = build_cnn(
        input_shape=(28, 28, 1),
        num_classes=10,
        name="digit_cnn"
    )

    model.summary()
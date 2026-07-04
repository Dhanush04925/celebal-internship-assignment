"""
autoencoder.py
==============
Convolutional Denoising Autoencoder architecture built with the Keras
Functional API: a Conv2D/BatchNorm/MaxPooling encoder that compresses the
image into a latent representation, and a Conv2D/UpSampling decoder that
reconstructs the clean digit.
"""

from __future__ import annotations

from typing import Tuple

from tensorflow.keras import Model, layers


def build_encoder(inputs: "layers.Input") -> "layers.Layer":
    """Build the convolutional encoder stack.

    Args:
        inputs: Keras Input tensor, shape `(28, 28, 1)`.

    Returns:
        The latent-space output tensor of the encoder.
    """
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 28x28 -> 14x14

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)  # 14x14 -> 7x7

    # Latent space bottleneck.
    latent = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="latent_space")(x)
    return latent


def build_decoder(latent: "layers.Layer") -> "layers.Layer":
    """Build the convolutional decoder stack.

    Args:
        latent: Latent-space tensor produced by `build_encoder`.

    Returns:
        The reconstructed-image output tensor, shape `(28, 28, 1)`.
    """
    x = layers.Conv2D(64, (3, 3), padding="same")(latent)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)  # 7x7 -> 14x14

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.UpSampling2D((2, 2))(x)  # 14x14 -> 28x28

    outputs = layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid", name="reconstruction")(x)
    return outputs


def build_autoencoder(input_shape: Tuple[int, int, int] = (28, 28, 1)) -> Model:
    """Assemble the full encoder-decoder autoencoder model.

    Args:
        input_shape: Shape of a single input image, defaults to `(28, 28, 1)`.

    Returns:
        A compiled-ready (uncompiled) `tf.keras.Model` mapping noisy images
        to reconstructed clean images.
    """
    inputs = layers.Input(shape=input_shape, name="noisy_input")
    latent = build_encoder(inputs)
    outputs = build_decoder(latent)

    model = Model(inputs=inputs, outputs=outputs, name="denoising_autoencoder")
    return model


if __name__ == "__main__":
    autoencoder = build_autoencoder()
    autoencoder.summary()

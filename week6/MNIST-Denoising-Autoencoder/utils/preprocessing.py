"""
preprocessing.py
=================
Normalization, reshaping, and Gaussian-noise injection utilities used to
turn raw MNIST pixel arrays into clean/noisy training tensors for the
denoising autoencoder.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Scale pixel intensities from [0, 255] to [0, 1].

    Args:
        images: Array of pixel values, any shape, dtype convertible to float32.

    Returns:
        A `float32` array of the same shape with values in [0, 1].
    """
    return images.astype("float32") / 255.0


def reshape_images(images: np.ndarray, target_shape: Tuple[int, int, int] = (28, 28, 1)) -> np.ndarray:
    """Reshape a batch of flat or 2D images into `(N, H, W, C)`.

    Args:
        images: Array shaped `(N, 28, 28)` or `(N, 784)`.
        target_shape: Desired per-sample shape, defaults to `(28, 28, 1)`.

    Returns:
        Array shaped `(N, *target_shape)`.
    """
    n_samples = images.shape[0]
    return images.reshape((n_samples, *target_shape))


def add_gaussian_noise(
    images: np.ndarray,
    noise_factor: float = 0.4,
    random_seed: int | None = None,
) -> np.ndarray:
    """Corrupt normalized images with additive Gaussian noise, then clip.

    Args:
        images: Normalized images in [0, 1], any shape.
        noise_factor: Standard-deviation multiplier for the noise term.
        random_seed: Optional seed for reproducible noise generation.

    Returns:
        Noisy images clipped back to the valid [0, 1] pixel range.
    """
    rng = np.random.default_rng(random_seed)
    noise = noise_factor * rng.normal(loc=0.0, scale=1.0, size=images.shape)
    noisy_images = images + noise
    return np.clip(noisy_images, 0.0, 1.0).astype("float32")


def preprocess_pipeline(
    x_train: np.ndarray,
    x_test: np.ndarray,
    noise_factor: float = 0.4,
    image_shape: Tuple[int, int, int] = (28, 28, 1),
    random_seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the full normalize -> reshape -> noise-injection pipeline.

    Args:
        x_train: Raw training images, shape `(N, 28, 28)`, range [0, 255].
        x_test: Raw test images, shape `(M, 28, 28)`, range [0, 255].
        noise_factor: Gaussian noise standard-deviation multiplier.
        image_shape: Target per-sample tensor shape.
        random_seed: Optional seed for reproducible noise.

    Returns:
        `(x_train_clean, x_train_noisy, x_test_clean, x_test_noisy)`, all
        `float32` arrays shaped `(N, *image_shape)` in range [0, 1].
    """
    x_train_clean = reshape_images(normalize_images(x_train), image_shape)
    x_test_clean = reshape_images(normalize_images(x_test), image_shape)

    x_train_noisy = add_gaussian_noise(x_train_clean, noise_factor, random_seed)
    x_test_noisy = add_gaussian_noise(
        x_test_clean,
        noise_factor,
        None if random_seed is None else random_seed + 1,
    )

    return x_train_clean, x_train_noisy, x_test_clean, x_test_noisy

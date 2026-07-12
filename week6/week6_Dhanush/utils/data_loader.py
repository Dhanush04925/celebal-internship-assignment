"""
data_loader.py
===============
Loads the raw Kaggle MNIST CSV files (`train.csv`, `test.csv`), separates
pixel data from labels, and provides helpers for caching preprocessed
NumPy arrays to disk.

The expected raw CSV layout (Kaggle "awsaf49/mnist-dataset" format) is:
    column 0      -> integer digit label (0-9)
    columns 1-784 -> flattened 28x28 grayscale pixel values (0-255)
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")


def load_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single MNIST CSV file and split it into images and labels.

    Args:
        csv_path: Path to a Kaggle-format MNIST CSV file.

    Returns:
        A tuple `(images, labels)` where `images` has shape
        `(n_samples, 28, 28)` with dtype `float32` in range [0, 255],
        and `labels` has shape `(n_samples,)` with dtype `int64`.

    Raises:
        FileNotFoundError: If `csv_path` does not exist.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'. Download the Kaggle MNIST "
            "dataset (awsaf49/mnist-dataset) and place train.csv / "
            "test.csv under data/MNIST/raw/."
        )

    logger.info("Loading CSV file: %s", csv_path)
    df = pd.read_csv(csv_path)

    labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    images = pixels.reshape(-1, 28, 28)

    logger.info("Loaded %d samples from %s", images.shape[0], csv_path)
    return images, labels


def load_image_folder(
    folder_path: str, image_size: Tuple[int, int] = (28, 28)
) -> Tuple[np.ndarray, np.ndarray]:
    """Load an ImageFolder-style directory (one subfolder per digit class).

    Expected layout::

        folder_path/
            0/  *.png|*.jpg|*.jpeg|*.bmp
            1/  ...
            ...
            9/  ...

    Args:
        folder_path: Path to the `training/` or `testing/` directory.
        image_size: Target `(H, W)` each image is resized to if needed.

    Returns:
        A tuple `(images, labels)` where `images` has shape
        `(n_samples, H, W)` with dtype `float32` in range [0, 255], and
        `labels` has shape `(n_samples,)` with dtype `int64`.

    Raises:
        FileNotFoundError: If `folder_path` does not exist.
        ValueError: If no class subfolders or no images are found.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Could not find directory '{folder_path}'.")

    class_dirs = sorted(
        (d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))),
        key=lambda d: (len(d), d),
    )
    if not class_dirs:
        raise ValueError(
            f"No class subfolders (e.g. '0', '1', ... '9') found under '{folder_path}'."
        )

    logger.info("Loading image folder: %s (%d class folders)", folder_path, len(class_dirs))

    images: list[np.ndarray] = []
    labels: list[int] = []

    for class_dir in class_dirs:
        try:
            label = int(class_dir)
        except ValueError:
            logger.warning("Skipping non-numeric class folder: %s", class_dir)
            continue

        class_path = os.path.join(folder_path, class_dir)
        filenames = sorted(
            f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        for filename in filenames:
            img_path = os.path.join(class_path, filename)
            with Image.open(img_path) as img:
                img = img.convert("L")  # ensure single-channel grayscale
                if img.size != image_size:
                    img = img.resize(image_size)
                images.append(np.asarray(img, dtype=np.float32))
            labels.append(label)

    if not images:
        raise ValueError(f"No images found under '{folder_path}'.")

    images_arr = np.stack(images, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)

    logger.info("Loaded %d images from %s", images_arr.shape[0], folder_path)
    return images_arr, labels_arr


def load_raw_dataset(config: Config) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load the raw train/test set, auto-detecting CSV vs. image-folder format.

    Checks for `train.csv` / `test.csv` first; if absent, falls back to the
    `training/<digit>/` / `testing/<digit>/` image-folder layout.

    Args:
        config: Project configuration holding both raw-data path styles.

    Returns:
        `(x_train, y_train, x_test, y_test)` as raw uint8-range float32
        arrays shaped `(N, 28, 28)`.

    Raises:
        FileNotFoundError: If neither CSV files nor image folders are found.
    """
    has_csv = os.path.exists(config.raw_train_csv) and os.path.exists(config.raw_test_csv)
    has_folders = os.path.isdir(config.raw_train_dir) and os.path.isdir(config.raw_test_dir)

    if has_csv:
        logger.info("Detected CSV-format raw data.")
        x_train, y_train = load_csv(config.raw_train_csv)
        x_test, y_test = load_csv(config.raw_test_csv)
    elif has_folders:
        logger.info("Detected image-folder-format raw data.")
        image_size = (config.image_shape[0], config.image_shape[1])
        x_train, y_train = load_image_folder(config.raw_train_dir, image_size)
        x_test, y_test = load_image_folder(config.raw_test_dir, image_size)
    else:
        raise FileNotFoundError(
            "No raw MNIST data found. Provide either:\n"
            f"  - CSV files at '{config.raw_train_csv}' and '{config.raw_test_csv}', or\n"
            f"  - Image folders at '{config.raw_train_dir}' and '{config.raw_test_dir}' "
            "(each containing subfolders '0'..'9' of images)."
        )

    return x_train, y_train, x_test, y_test


def save_processed_arrays(
    config: Config,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Persist preprocessed arrays to `config.processed_dir` as `.npy` files.

    Args:
        config: Project configuration holding the processed-data directory.
        x_train: Preprocessed training images.
        y_train: Training labels.
        x_test: Preprocessed test images.
        y_test: Test labels.
    """
    os.makedirs(config.processed_dir, exist_ok=True)
    np.save(os.path.join(config.processed_dir, "x_train.npy"), x_train)
    np.save(os.path.join(config.processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(config.processed_dir, "x_test.npy"), x_test)
    np.save(os.path.join(config.processed_dir, "y_test.npy"), y_test)
    logger.info("Saved processed arrays to %s", config.processed_dir)


def load_processed_arrays(config: Config) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Load previously cached preprocessed `.npy` arrays.

    Args:
        config: Project configuration holding the processed-data directory.

    Returns:
        `(x_train, y_train, x_test, y_test)` as saved on disk.

    Raises:
        FileNotFoundError: If any expected `.npy` file is missing.
    """
    paths = {
        "x_train": os.path.join(config.processed_dir, "x_train.npy"),
        "y_train": os.path.join(config.processed_dir, "y_train.npy"),
        "x_test": os.path.join(config.processed_dir, "x_test.npy"),
        "y_test": os.path.join(config.processed_dir, "y_test.npy"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing processed array '{name}' at {path}")

    return (
        np.load(paths["x_train"]),
        np.load(paths["y_train"]),
        np.load(paths["x_test"]),
        np.load(paths["y_test"]),
    )


def processed_arrays_exist(config: Config) -> bool:
    """Check whether cached preprocessed arrays already exist on disk.

    Args:
        config: Project configuration holding the processed-data directory.

    Returns:
        `True` if all four expected `.npy` files are present.
    """
    filenames = ("x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy")
    return all(
        os.path.exists(os.path.join(config.processed_dir, f)) for f in filenames
    )

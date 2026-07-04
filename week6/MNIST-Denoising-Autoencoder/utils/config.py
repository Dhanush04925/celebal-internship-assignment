"""
config.py
=========
Centralized configuration for the MNIST Denoising Autoencoder project.

All paths, hyperparameters, and reproducibility settings live in a single
`Config` dataclass so that `train.py`, `predict.py`, and the notebook stay
in sync and command-line overrides are trivial to apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


# --------------------------------------------------------------------------- #
# Project root resolution
# --------------------------------------------------------------------------- #
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    """Holds every configurable value used across the project.

    Attributes:
        raw_train_csv: Path to the raw Kaggle MNIST training CSV.
        raw_test_csv: Path to the raw Kaggle MNIST test CSV.
        processed_dir: Directory where preprocessed .npy arrays are stored.
        checkpoints_dir: Directory where model checkpoints are saved.
        figures_dir: Directory where evaluation/visualization plots are saved.
        logs_dir: Directory used by TensorBoard and the Python logger.
        history_dir: Directory where training-history CSV files are saved.
        image_shape: Target shape (H, W, C) for every MNIST image.
        noise_factor: Standard-deviation scale of the Gaussian noise added
            to clean images to build the noisy inputs.
        noise_factors_to_compare: A list of noise factors used by the
            optional noise-level comparison experiment.
        val_split: Fraction of the training set reserved for validation.
        batch_size: Mini-batch size used during training.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the Adam optimizer.
        loss: Reconstruction loss function name ("binary_crossentropy" or
            "mse").
        random_seed: Global random seed for NumPy / TensorFlow / Python.
        early_stopping_patience: Epochs with no val_loss improvement before
            stopping training early.
        reduce_lr_patience: Epochs with no val_loss improvement before the
            learning rate is reduced.
        reduce_lr_factor: Multiplicative factor applied to the learning
            rate by ReduceLROnPlateau.
    """

    # Data paths (CSV format: Kaggle awsaf49/mnist-dataset "train.csv" / "test.csv")
    raw_train_csv: str = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw", "train.csv")
    raw_test_csv: str = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw", "test.csv")

    # Data paths (image-folder format: raw/training/<digit>/*.png, raw/testing/<digit>/*.png)
    raw_train_dir: str = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw", "training")
    raw_test_dir: str = os.path.join(PROJECT_ROOT, "data", "MNIST", "raw", "testing")

    processed_dir: str = os.path.join(PROJECT_ROOT, "data", "MNIST", "processed")

    # Output paths
    checkpoints_dir: str = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
    figures_dir: str = os.path.join(PROJECT_ROOT, "outputs", "figures")
    logs_dir: str = os.path.join(PROJECT_ROOT, "outputs", "logs")
    history_dir: str = os.path.join(PROJECT_ROOT, "outputs", "history")

    # Data properties
    image_shape: Tuple[int, int, int] = (28, 28, 1)

    # Noise settings
    noise_factor: float = 0.4
    noise_factors_to_compare: Tuple[float, ...] = (0.2, 0.4, 0.6)

    # Training hyperparameters
    val_split: float = 0.1
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-3
    loss: str = "binary_crossentropy"

    # Reproducibility
    random_seed: int = 42

    # Callback settings
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5

    # Model artifact names
    model_filename: str = "denoising_autoencoder_best.keras"
    history_filename: str = "training_history.csv"

    def ensure_directories(self) -> None:
        """Create every output directory declared in this config if absent."""
        for directory in (
            self.processed_dir,
            self.checkpoints_dir,
            self.figures_dir,
            self.logs_dir,
            self.history_dir,
        ):
            os.makedirs(directory, exist_ok=True)

    @property
    def model_path(self) -> str:
        """Full path to the best-checkpoint model file."""
        return os.path.join(self.checkpoints_dir, self.model_filename)

    @property
    def history_path(self) -> str:
        """Full path to the saved training-history CSV."""
        return os.path.join(self.history_dir, self.history_filename)


DEFAULT_CONFIG: Config = Config()

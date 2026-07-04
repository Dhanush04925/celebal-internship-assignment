#!/usr/bin/env python3
"""
train.py
========
End-to-end training entry point for the MNIST Denoising Autoencoder.

Pipeline:
    1. Load raw MNIST CSVs (or cached processed arrays, if present).
    2. Normalize, reshape, and inject Gaussian noise.
    3. Cache processed arrays to disk.
    4. Build the convolutional autoencoder.
    5. Train with EarlyStopping / ReduceLROnPlateau / ModelCheckpoint /
       TensorBoard callbacks.
    6. Save the training history and diagnostic plots.

Usage:
    python train.py
    python train.py --epochs 30 --batch-size 64 --noise-factor 0.3
    python train.py --loss mse
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

import numpy as np
import tensorflow as tf

from models.autoencoder import build_autoencoder
from models.trainer import Trainer
from utils.config import DEFAULT_CONFIG
from utils.data_loader import (
    load_raw_dataset,
    processed_arrays_exist,
    load_processed_arrays,
    save_processed_arrays,
)
from utils.logger import get_logger
from utils.preprocessing import add_gaussian_noise, preprocess_pipeline
from utils.visualization import plot_original_noisy_reconstructed, plot_training_history

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration overrides."""
    parser = argparse.ArgumentParser(description="Train the MNIST Denoising Autoencoder.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size.")
    parser.add_argument("--noise-factor", type=float, default=None, help="Gaussian noise std multiplier.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam learning rate.")
    parser.add_argument(
        "--loss", type=str, default=None, choices=["binary_crossentropy", "mse"],
        help="Reconstruction loss function.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument(
        "--force-reprocess", action="store_true",
        help="Ignore cached processed arrays and rebuild them from raw CSVs.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    """Run the full data -> model -> training -> evaluation pipeline."""
    args = parse_args()

    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "noise_factor": args.noise_factor,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
            "random_seed": args.seed,
        }.items()
        if v is not None
    }
    config = replace(DEFAULT_CONFIG, **overrides)
    config.ensure_directories()
    set_global_seed(config.random_seed)

    logger.info("Configuration: %s", config)

    # --- 1 & 2. Load and preprocess data -------------------------------- #
    if processed_arrays_exist(config) and not args.force_reprocess:
        logger.info("Found cached processed arrays; loading clean images from disk.")
        x_train_clean, _, _, _ = load_processed_arrays(config)
    else:
        logger.info("Loading raw MNIST CSVs.")
        x_train_2d, y_train, x_test_2d, y_test = load_raw_dataset(config)
        x_train_clean, _, x_test_clean, _ = preprocess_pipeline(
            x_train_2d, x_test_2d, config.noise_factor, config.image_shape, config.random_seed
        )
        save_processed_arrays(config, x_train_clean, y_train, x_test_clean, y_test)

    # Noisy inputs are always (re)generated at the active noise_factor so
    # that --noise-factor overrides take effect even with cached clean data.
    x_train_noisy = add_gaussian_noise(x_train_clean, config.noise_factor, config.random_seed)

    logger.info(
        "Data ready: x_train_clean=%s, x_train_noisy=%s",
        x_train_clean.shape,
        x_train_noisy.shape,
    )

    # --- 3. Build model --------------------------------------------------- #
    autoencoder = build_autoencoder(config.image_shape)
    autoencoder.summary(print_fn=logger.info)

    # --- 4. Train ----------------------------------------------------------#
    trainer = Trainer(autoencoder, config)
    trainer.compile()
    trainer.fit(x_train_noisy, x_train_clean)
    trainer.save_history_csv()
    trainer.save_final_model()

    # --- 5. Diagnostic plots ---------------------------------------------- #
    plot_training_history(
        trainer.history.history,
        metrics=("loss", "mse"),
        save_path=f"{config.figures_dir}/training_history.png",
    )

    sample_reconstructions = autoencoder.predict(x_train_noisy[:200], verbose=0)
    plot_original_noisy_reconstructed(
        x_train_clean[:200],
        x_train_noisy[:200],
        sample_reconstructions,
        n_samples=10,
        save_path=f"{config.figures_dir}/train_reconstructions_sample.png",
        random_seed=config.random_seed,
    )

    logger.info("Training pipeline finished successfully.")
    logger.info("Best model saved at: %s", config.model_path)
    logger.info("Training history CSV: %s", config.history_path)


if __name__ == "__main__":
    main()

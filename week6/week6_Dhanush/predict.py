#!/usr/bin/env python3
"""
predict.py
==========
Inference and evaluation entry point. Loads the trained autoencoder
checkpoint, denoises the MNIST test set, saves comparison plots, and
reports MSE / PSNR / SSIM metrics.

Usage:
    python predict.py
    python predict.py --model-path outputs/checkpoints/denoising_autoencoder_best.keras
    python predict.py --noise-factor 0.6 --n-samples 15
"""

from __future__ import annotations

import argparse
import json
import os

from utils.config import DEFAULT_CONFIG
from utils.data_loader import load_processed_arrays, processed_arrays_exist
from utils.logger import get_logger
from utils.preprocessing import add_gaussian_noise
from utils.visualization import plot_original_noisy_reconstructed
from models.predictor import Predictor

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the prediction/evaluation run."""
    parser = argparse.ArgumentParser(description="Evaluate the MNIST Denoising Autoencoder.")
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to a saved model checkpoint (defaults to the configured best model).",
    )
    parser.add_argument(
        "--noise-factor", type=float, default=None,
        help="Gaussian noise std multiplier applied to the test set.",
    )
    parser.add_argument("--n-samples", type=int, default=10, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for noise/sampling.")
    return parser.parse_args()


def main() -> None:
    """Run inference on the test set and report evaluation metrics."""
    args = parse_args()
    config = DEFAULT_CONFIG
    config.ensure_directories()

    model_path = args.model_path or config.model_path
    noise_factor = args.noise_factor if args.noise_factor is not None else config.noise_factor
    seed = args.seed if args.seed is not None else config.random_seed

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at '{model_path}'. Run train.py first, or pass --model-path."
        )
    if not processed_arrays_exist(config):
        raise FileNotFoundError(
            "No processed test data found. Run train.py first to generate "
            f"processed arrays under {config.processed_dir}."
        )

    _, _, x_test_clean, y_test = load_processed_arrays(config)
    x_test_noisy = add_gaussian_noise(x_test_clean, noise_factor, seed)

    logger.info("Loaded test set: %s samples", x_test_clean.shape[0])

    predictor = Predictor(model_path)
    reconstructions, report = predictor.denoise_and_evaluate(x_test_noisy, x_test_clean)

    plot_path = os.path.join(config.figures_dir, "test_reconstructions_sample.png")
    plot_original_noisy_reconstructed(
        x_test_clean,
        x_test_noisy,
        reconstructions,
        n_samples=args.n_samples,
        save_path=plot_path,
        random_seed=seed,
    )
    logger.info("Saved comparison plot to %s", plot_path)

    report_path = os.path.join(config.figures_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.as_dict(), f, indent=2)
    logger.info("Saved evaluation report to %s", report_path)

    print("\n=== Evaluation Report ===")
    print(report)


if __name__ == "__main__":
    main()

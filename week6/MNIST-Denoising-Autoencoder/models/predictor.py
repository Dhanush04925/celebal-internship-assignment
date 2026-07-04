"""
predictor.py
============
`Predictor` class that loads a saved autoencoder checkpoint and generates
denoised reconstructions for new noisy images, plus a convenience method
that also computes evaluation metrics against ground-truth clean images.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from utils.logger import get_logger
from utils.metrics import EvaluationReport, evaluate_reconstructions

logger = get_logger(__name__)


class Predictor:
    """Loads a trained model and runs inference / evaluation.

    Args:
        model_path: Path to a saved `.keras` (or `.h5`) model file.
    """

    def __init__(self, model_path: str) -> None:
        logger.info("Loading model from %s", model_path)
        self.model: Model = tf.keras.models.load_model(model_path)

    def denoise(self, noisy_images: np.ndarray) -> np.ndarray:
        """Generate reconstructions for a batch of noisy images.

        Args:
            noisy_images: Noisy images, shape `(N, H, W, C)`, range [0, 1].

        Returns:
            Reconstructed (denoised) images, same shape as `noisy_images`.
        """
        reconstructions = self.model.predict(noisy_images, verbose=0)
        return reconstructions

    def denoise_and_evaluate(
        self,
        noisy_images: np.ndarray,
        clean_images: np.ndarray,
    ) -> tuple[np.ndarray, EvaluationReport]:
        """Denoise a batch and score the reconstructions against ground truth.

        Args:
            noisy_images: Noisy images, shape `(N, H, W, C)`.
            clean_images: Corresponding ground-truth clean images.

        Returns:
            A tuple `(reconstructions, report)` where `report` contains the
            aggregated MSE / PSNR / SSIM scores.
        """
        reconstructions = self.denoise(noisy_images)
        report = evaluate_reconstructions(clean_images, reconstructions)
        logger.info("Evaluation on %d samples -> %s", clean_images.shape[0], report)
        return reconstructions, report

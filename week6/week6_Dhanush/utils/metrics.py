"""
metrics.py
==========
Reconstruction-quality metrics for the denoising autoencoder: MSE, PSNR,
and SSIM, computed per-image and aggregated across a batch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass
class EvaluationReport:
    """Aggregate reconstruction-quality metrics over a dataset.

    Attributes:
        mse: Mean squared error averaged over all images.
        psnr: Peak signal-to-noise ratio averaged over all images (dB).
        ssim: Structural similarity index averaged over all images.
        n_samples: Number of images the metrics were computed over.
    """

    mse: float
    psnr: float
    ssim: float
    n_samples: int

    def as_dict(self) -> Dict[str, float]:
        """Return the report as a plain dictionary for CSV/JSON export."""
        return {
            "mse": self.mse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        return (
            f"MSE={self.mse:.6f} | PSNR={self.psnr:.2f} dB | "
            f"SSIM={self.ssim:.4f} | n={self.n_samples}"
        )


def compute_mse(originals: np.ndarray, reconstructions: np.ndarray) -> float:
    """Compute the mean squared error between two image batches.

    Args:
        originals: Ground-truth clean images, shape `(N, H, W, C)`.
        reconstructions: Model outputs, same shape as `originals`.

    Returns:
        Scalar MSE averaged over every pixel and every sample.
    """
    return float(np.mean(np.square(originals.astype("float64") - reconstructions.astype("float64"))))


def compute_psnr(originals: np.ndarray, reconstructions: np.ndarray) -> float:
    """Compute mean PSNR (dB) across an image batch.

    Args:
        originals: Ground-truth clean images in [0, 1], shape `(N, H, W, C)`.
        reconstructions: Model outputs in [0, 1], same shape as `originals`.

    Returns:
        Average PSNR in decibels over all samples.
    """
    scores = [
        peak_signal_noise_ratio(originals[i], reconstructions[i], data_range=1.0)
        for i in range(originals.shape[0])
    ]
    return float(np.mean(scores))


def compute_ssim(originals: np.ndarray, reconstructions: np.ndarray) -> float:
    """Compute mean SSIM across an image batch.

    Args:
        originals: Ground-truth clean images in [0, 1], shape `(N, H, W, C)`.
        reconstructions: Model outputs in [0, 1], same shape as `originals`.

    Returns:
        Average structural similarity index over all samples.
    """
    scores = [
        structural_similarity(
            originals[i].squeeze(),
            reconstructions[i].squeeze(),
            data_range=1.0,
        )
        for i in range(originals.shape[0])
    ]
    return float(np.mean(scores))


def evaluate_reconstructions(
    originals: np.ndarray, reconstructions: np.ndarray
) -> EvaluationReport:
    """Compute MSE, PSNR, and SSIM together and package into a report.

    Args:
        originals: Ground-truth clean images in [0, 1], shape `(N, H, W, C)`.
        reconstructions: Model outputs in [0, 1], same shape as `originals`.

    Returns:
        An `EvaluationReport` with aggregated metrics.
    """
    mse = compute_mse(originals, reconstructions)
    psnr = compute_psnr(originals, reconstructions)
    ssim = compute_ssim(originals, reconstructions)
    return EvaluationReport(mse=mse, psnr=psnr, ssim=ssim, n_samples=originals.shape[0])

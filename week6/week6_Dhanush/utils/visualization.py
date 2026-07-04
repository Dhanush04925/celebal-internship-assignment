"""
visualization.py
=================
Plotting helpers for comparing original / noisy / reconstructed MNIST
digits, and for visualizing training history curves.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_original_noisy_reconstructed(
    originals: np.ndarray,
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    n_samples: int = 10,
    save_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> None:
    """Plot rows of Original / Noisy / Reconstructed digits side by side.

    Args:
        originals: Clean images, shape `(N, H, W, C)`.
        noisy: Noisy images, same shape as `originals`.
        reconstructed: Model reconstructions, same shape as `originals`.
        n_samples: Number of random samples to display (columns).
        save_path: If provided, save the figure to this path.
        random_seed: Optional seed for reproducible sample selection.
    """
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(originals.shape[0], size=n_samples, replace=False)

    fig, axes = plt.subplots(3, n_samples, figsize=(1.5 * n_samples, 4.5))
    row_titles = ("Original", "Noisy", "Reconstructed")
    image_sets = (originals, noisy, reconstructed)

    for row, (title, images) in enumerate(zip(row_titles, image_sets)):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=11, rotation=90)

    fig.suptitle("Original vs. Noisy vs. Reconstructed MNIST Digits", fontsize=13)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(
    history: dict,
    metrics: Sequence[str] = ("loss", "mse"),
    save_path: Optional[str] = None,
) -> None:
    """Plot training/validation curves for one or more metrics.

    Args:
        history: A Keras `History.history`-style dict mapping metric names
            (and their `val_` counterparts) to per-epoch lists.
        metrics: Base metric names to plot (each plotted against its
            validation counterpart if present).
        save_path: If provided, save the figure to this path.
    """
    available = [m for m in metrics if m in history]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        ax.plot(history[metric], label=f"train_{metric}")
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], label=f"val_{metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Training vs. Validation {metric.upper()}")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_factor_comparison(
    clean_image: np.ndarray,
    noisy_versions: dict,
    save_path: Optional[str] = None,
) -> None:
    """Visualize the same digit corrupted at several noise factors.

    Args:
        clean_image: A single clean image, shape `(H, W, C)`.
        noisy_versions: Mapping of `{noise_factor: noisy_image}`.
        save_path: If provided, save the figure to this path.
    """
    n_cols = len(noisy_versions) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 2.5))

    axes[0].imshow(clean_image.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Clean")
    axes[0].axis("off")

    for ax, (factor, image) in zip(axes[1:], noisy_versions.items()):
        ax.imshow(image.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"noise={factor}")
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

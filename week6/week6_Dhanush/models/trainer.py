"""
trainer.py
==========
`Trainer` class that compiles the autoencoder, wires up all training
callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard),
runs `model.fit`, and persists both the best model and the training
history.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Encapsulates compilation, callback setup, and fitting for the model.

    Args:
        model: An uncompiled `tf.keras.Model` (typically from
            `models.autoencoder.build_autoencoder`).
        config: Project configuration with hyperparameters and output paths.
    """

    def __init__(self, model: Model, config: Config) -> None:
        self.model = model
        self.config = config
        self.history: Optional[tf.keras.callbacks.History] = None

    def compile(self) -> None:
        """Compile the model with Adam, the configured loss, and MSE metric."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=self.config.loss,
            metrics=["mse"],
        )
        logger.info(
            "Compiled model with optimizer=Adam(lr=%.1e), loss=%s",
            self.config.learning_rate,
            self.config.loss,
        )

    def _build_callbacks(self) -> list[Callback]:
        """Construct the standard callback stack used during training."""
        self.config.ensure_directories()
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_dir = os.path.join(self.config.logs_dir, run_id)

        callbacks: list[Callback] = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.config.reduce_lr_factor,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=self.config.model_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=tensorboard_dir),
        ]
        return callbacks

    def fit(
        self,
        x_train_noisy: np.ndarray,
        x_train_clean: np.ndarray,
    ) -> tf.keras.callbacks.History:
        """Train the autoencoder to map noisy inputs to clean targets.

        Args:
            x_train_noisy: Noisy training images (model input).
            x_train_clean: Clean training images (reconstruction target).

        Returns:
            The Keras `History` object returned by `model.fit`.
        """
        callbacks = self._build_callbacks()

        logger.info(
            "Starting training for up to %d epochs (batch_size=%d)",
            self.config.epochs,
            self.config.batch_size,
        )
        self.history = self.model.fit(
            x_train_noisy,
            x_train_clean,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.val_split,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )
        logger.info("Training complete.")
        return self.history

    def save_history_csv(self) -> str:
        """Persist the per-epoch training history to a CSV file.

        Returns:
            The path the history CSV was written to.

        Raises:
            RuntimeError: If called before `fit()`.
        """
        if self.history is None:
            raise RuntimeError("Call fit() before save_history_csv().")

        os.makedirs(self.config.history_dir, exist_ok=True)
        df = pd.DataFrame(self.history.history)
        df.index.name = "epoch"
        df.to_csv(self.config.history_path)
        logger.info("Saved training history to %s", self.config.history_path)
        return self.config.history_path

    def save_final_model(self) -> str:
        """Explicitly save the current model state (in addition to the
        best-checkpoint saved automatically during training).

        Returns:
            The path the model was saved to.
        """
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        self.model.save(self.config.model_path)
        logger.info("Saved model to %s", self.config.model_path)
        return self.config.model_path

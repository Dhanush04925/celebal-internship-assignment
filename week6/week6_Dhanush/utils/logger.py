"""
logger.py
=========
Project-wide logging setup. Provides a single `get_logger` factory so every
module logs in a consistent format, both to the console and to a rotating
file under `outputs/logs/`.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str,
    log_dir: str = "outputs/logs",
    log_filename: str = "project.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or fetch) a configured logger.

    Args:
        name: Logger name, typically `__name__` of the calling module.
        log_dir: Directory where the rotating log file will be written.
        log_filename: File name for the rotating log file.
        level: Logging level (e.g. `logging.INFO`, `logging.DEBUG`).

    Returns:
        A `logging.Logger` instance with a console handler and a rotating
        file handler attached exactly once.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Avoid attaching duplicate handlers if this logger was already set up.
        return logger

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger

"""Logging helpers."""
from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the package logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("fmri_testing")

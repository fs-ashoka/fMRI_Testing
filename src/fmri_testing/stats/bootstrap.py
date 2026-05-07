"""Bootstrap statistics for encoding comparisons."""
from __future__ import annotations

import numpy as np


def bootstrap_mean_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Bootstrap the mean paired difference a - b over rows."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    vals = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)]
    return {"mean_difference": float(diff.mean()), "ci_low": float(np.percentile(vals, 2.5)), "ci_high": float(np.percentile(vals, 97.5))}

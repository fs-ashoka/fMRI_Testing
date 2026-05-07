"""Permutation tests and FDR correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import false_discovery_control


def paired_sign_permutation(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided paired sign-flip permutation p-value for mean difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    obs = abs(diff.mean())
    null = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(diff))
        null.append(abs((diff * signs).mean()))
    return float((np.sum(np.asarray(null) >= obs) + 1) / (n_perm + 1))


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjust p-values via SciPy."""
    return false_discovery_control(np.asarray(p_values), method="bh")

"""Encoding-model evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def pearsonr_per_voxel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute Pearson r independently for each voxel."""
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    return np.divide((yt * yp).sum(axis=0), denom, out=np.zeros(y_true.shape[1]), where=denom > 1e-12)


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | list[float]]:
    """Return voxelwise and aggregate encoding metrics."""
    r = pearsonr_per_voxel(y_true, y_pred)
    r2 = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
    cutoff = max(1, int(np.ceil(0.05 * len(r))))
    return {
        "pearson_r": r.tolist(),
        "r2": r2.tolist(),
        "mean_r": float(np.nanmean(r)),
        "median_r": float(np.nanmedian(r)),
        "top5_percent_mean_r": float(np.nanmean(np.sort(r)[-cutoff:])),
    }

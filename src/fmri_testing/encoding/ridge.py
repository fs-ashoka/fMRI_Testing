"""Ridge encoding models and baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, KFold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler

from fmri_testing.encoding.evaluate import summarize_predictions
from fmri_testing.utils.io import write_json


def make_feature_sets(raw: np.ndarray, sae: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """Construct raw, PCA, SAE, random projection, and shuffled-SAE features."""
    dim = sae.shape[1]
    n_comp = min(dim, raw.shape[0] - 1, raw.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit_transform(raw)
    if n_comp < dim:
        pca = np.pad(pca, ((0, 0), (0, dim - n_comp)))
    rp = GaussianRandomProjection(n_components=dim, random_state=seed).fit_transform(raw)
    rng = np.random.default_rng(seed)
    shuffled = sae.copy()
    rng.shuffle(shuffled, axis=0)
    return {"raw": raw, "pca": pca.astype(np.float32), "sae": sae, "random_projection": rp.astype(np.float32), "shuffled_sae": shuffled}


def grouped_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, alphas: list[float], n_splits: int = 5
) -> tuple[np.ndarray, list[Ridge]]:
    """Cross-validated ridge predictions grouped by image identity when possible."""
    n_splits = min(n_splits, len(x))
    if groups is not None and len(np.unique(groups)) >= 2:
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        splits = splitter.split(x, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        splits = splitter.split(x, y)
    pred = np.zeros_like(y, dtype=np.float32)
    models: list[Ridge] = []
    for train, test in splits:
        scaler = StandardScaler().fit(x[train])
        x_train = scaler.transform(x[train])
        x_test = scaler.transform(x[test])
        cv = min(3, len(train))
        model = RidgeCV(alphas=alphas, cv=cv).fit(x_train, y[train]) if len(train) >= 3 else Ridge(alpha=alphas[0]).fit(x_train, y[train])
        pred[test] = model.predict(x_test).astype(np.float32)
        final = Ridge(alpha=float(getattr(model, "alpha_", alphas[0]))).fit(scaler.transform(x), y)
        final.scaler_ = scaler  # type: ignore[attr-defined]
        models.append(final)
    return pred, models


def fit_encoding_pipeline(cfg: dict[str, Any], subject: str) -> tuple[Path, Path]:
    """Fit all configured feature sets for one subject and write summaries."""
    out = Path(cfg["paths"]["output_dir"])
    betas = np.load(out / f"prepared_betas_subject_{subject}.npz")["betas"].astype(np.float32)
    raw = np.load(out / "vision_features.npz")["raw"].astype(np.float32)[: len(betas)]
    sae = np.load(out / "sae_latents_all.npy").astype(np.float32)[: len(betas)]
    meta = pd.read_csv(out / f"trial_metadata_subject_{subject}.csv")
    groups = meta["image_name"].to_numpy() if "image_name" in meta else None
    feature_sets = make_feature_sets(raw, sae, int(cfg["project"].get("seed", 0)))
    rows = []
    metrics: dict[str, Any] = {}
    for name in cfg["encoding"].get("feature_sets", list(feature_sets)):
        pred, _models = grouped_predictions(feature_sets[name], betas, groups, list(map(float, cfg["encoding"]["alphas"])), int(cfg["encoding"].get("n_splits", 5)))
        summ = summarize_predictions(betas, pred)
        metrics[name] = summ
        rows.append({"subject": subject, "feature_set": name, **{k: v for k, v in summ.items() if not isinstance(v, list)}})
        np.save(out / f"encoding_{name}_pearson_r.npy", np.asarray(summ["pearson_r"], dtype=np.float32))
        if cfg["encoding"].get("save_predictions", False):
            np.save(out / f"predictions_{name}_{subject}.npy", pred)
    metrics_path = out / "metrics.json"
    summary_path = out / "encoding_summary.csv"
    write_json(metrics_path, metrics)
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return metrics_path, summary_path


def fit_final_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Ridge:
    """Fit a full-data ridge model with attached standardizer for ablation."""
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    model.scaler_ = scaler  # type: ignore[attr-defined]
    return model

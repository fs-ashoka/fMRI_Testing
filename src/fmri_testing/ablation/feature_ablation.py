"""SAE feature ablation analyses."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fmri_testing.encoding.evaluate import pearsonr_per_voxel
from fmri_testing.encoding.ridge import fit_final_ridge
from fmri_testing.utils.io import write_json


def ablate_feature_drops(latents: np.ndarray, betas: np.ndarray, max_features: int | None = None) -> pd.DataFrame:
    """Zero each SAE feature and measure mean prediction-correlation drop."""
    n_features = latents.shape[1] if max_features is None else min(max_features, latents.shape[1])
    model = fit_final_ridge(latents, betas)
    scaler = model.scaler_  # type: ignore[attr-defined]
    base_pred = model.predict(scaler.transform(latents))
    base_r = pearsonr_per_voxel(betas, base_pred)
    rows = []
    freq = (latents != 0).mean(axis=0)
    var = latents.var(axis=0)
    for feature_id in range(n_features):
        x_ablated = latents.copy()
        x_ablated[:, feature_id] = 0.0
        pred = model.predict(scaler.transform(x_ablated))
        r = pearsonr_per_voxel(betas, pred)
        drop = base_r - r
        rows.append({
            "feature_id": feature_id,
            "mean_drop": float(np.nanmean(drop)),
            "median_drop": float(np.nanmedian(drop)),
            "max_drop": float(np.nanmax(drop)),
            "activation_frequency": float(freq[feature_id]),
            "feature_variance": float(var[feature_id]),
        })
    df = pd.DataFrame(rows)
    if len(df) > 1:
        freq = df["activation_frequency"].to_numpy()
        var = df["feature_variance"].to_numpy()
        freq_z = (freq - freq.mean()) / (freq.std() + 1e-12)
        var_z = (var - var.mean()) / (var.std() + 1e-12)
        matched_ids = []
        matched_drops = []
        for i in range(len(df)):
            dist = (freq_z - freq_z[i]) ** 2 + (var_z - var_z[i]) ** 2
            dist[i] = np.inf
            j = int(np.argmin(dist))
            matched_ids.append(int(df.iloc[j]["feature_id"]))
            matched_drops.append(float(df.iloc[j]["mean_drop"]))
        df["matched_control_feature_id"] = matched_ids
        df["matched_control_mean_drop"] = matched_drops
        df["drop_minus_matched_control"] = df["mean_drop"] - df["matched_control_mean_drop"]
    else:
        df["matched_control_feature_id"] = -1
        df["matched_control_mean_drop"] = np.nan
        df["drop_minus_matched_control"] = np.nan
    return df


def run_ablation_pipeline(cfg: dict[str, Any], subject: str) -> tuple[Path, Path, Path]:
    """Run SAE feature ablation and write requested output tables."""
    out = Path(cfg["paths"]["output_dir"])
    latents = np.load(out / "sae_latents_all.npy").astype(np.float32)
    betas = np.load(out / f"prepared_betas_subject_{subject}.npz")["betas"].astype(np.float32)
    latents = latents[: len(betas)]
    df = ablate_feature_drops(latents, betas, cfg["ablation"].get("max_features"))
    drops_path = out / "ablation_feature_drops.csv"
    roi_path = out / "ablation_roi_matrix.csv"
    summary_path = out / "ablation_summary.json"
    df.to_csv(drops_path, index=False)
    pd.DataFrame({"feature_id": df["feature_id"], "whole_brain_valid_voxels": df["mean_drop"]}).to_csv(roi_path, index=False)
    write_json(summary_path, {
        "n_features_tested": int(len(df)),
        "mean_drop": float(df["mean_drop"].mean()) if len(df) else 0.0,
        "top_feature_id": int(df.sort_values("mean_drop", ascending=False).iloc[0]["feature_id"]) if len(df) else -1,
        "roi_mode": "whole_brain_valid_voxels_when_roi_masks_absent",
    })
    return drops_path, roi_path, summary_path

#!/usr/bin/env python
"""Run ROI-wise encoding on BOLD5000 GLMsingle ROI beta files.

This script expects:
- outputs/default/trial_metadata_subject_CSI1.csv or configured output_dir equivalent
- outputs/default/vision_features.npz
- outputs/default/sae_latents_all.npy
- data/bold5000/BOLD5000_GLMsingle_ROI_betas.zip

It extracts CSI1 ROI .npy beta files from the ROI zip and compares raw, PCA,
SAE, random projection, and shuffled SAE features using cross-validated ridge
encoding.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fmri_testing.config import ensure_dirs, load_config


def _extract_roi_files(zip_path: Path, roi_dir: Path, subject: str) -> list[Path]:
    roi_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = [
            m for m in z.namelist()
            if f"BOLD5000_GLMsingle_ROI_betas/py/{subject}_" in m
            and m.endswith(".npy")
            and "__MACOSX" not in m
        ]
        for member in members:
            target = roi_dir / Path(member).name
            if not target.exists():
                with z.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
    return sorted(roi_dir.glob(f"{subject}_*.npy"))


def _align_roi(arr: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        return None
    max_idx = int(idx.max())
    if arr.shape[0] > max_idx:
        y = arr[idx, :]
    elif arr.shape[1] > max_idx:
        y = arr[:, idx].T
    else:
        return None
    good = np.isfinite(y).mean(axis=0) > 0.99
    good &= np.nanstd(y, axis=0) > 1e-8
    y = y[:, good]
    if y.shape[1] == 0:
        return None
    mu = np.nanmean(y, axis=0, keepdims=True)
    sd = np.nanstd(y, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    y = (y - mu) / sd
    y[~np.isfinite(y)] = 0.0
    return y.astype(np.float32)


def _corr_per_voxel(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    return np.divide((yt * yp).sum(axis=0), denom, out=np.full(y_true.shape[1], np.nan), where=denom > 1e-12)


def _cv_encode(x: np.ndarray, y: np.ndarray, n_splits: int, alphas: list[float]) -> dict[str, float | int]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=7)
    pred = np.zeros_like(y, dtype=np.float32)
    for train, test in kf.split(x):
        model = make_pipeline(StandardScaler(with_mean=True), RidgeCV(alphas=np.asarray(alphas, dtype=float)))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test]).astype(np.float32)
    r = _corr_per_voxel(y, pred)
    finite = r[np.isfinite(r)]
    top_n = max(1, int(0.05 * len(finite)))
    return {
        "mean_r": float(np.nanmean(r)),
        "median_r": float(np.nanmedian(r)),
        "top5_percent_mean_r": float(np.nanmean(np.sort(finite)[-top_n:])),
        "n_voxels": int(y.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subject", default="CSI1")
    parser.add_argument("--roi-zip", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, "configs/default.yaml")
    ensure_dirs(cfg)
    subject = args.subject
    out_dir = Path(cfg["paths"]["output_dir"])
    bold_dir = Path(cfg["paths"]["bold5000_dir"])
    roi_zip = Path(args.roi_zip) if args.roi_zip else bold_dir / "BOLD5000_GLMsingle_ROI_betas.zip"
    roi_dir = bold_dir / "roi_betas"

    if not roi_zip.exists():
        raise FileNotFoundError(f"ROI zip not found: {roi_zip}")

    roi_files = _extract_roi_files(roi_zip, roi_dir, subject)
    if not roi_files:
        raise FileNotFoundError(f"No ROI .npy files found for {subject} in {roi_zip}")

    meta = pd.read_csv(out_dir / f"trial_metadata_subject_{subject}.csv")
    idx_col = "beta_row_original" if "beta_row_original" in meta.columns else "beta_row"
    orig_idx = meta[idx_col].to_numpy(dtype=int)

    features = np.load(out_dir / "vision_features.npz")
    raw = features["raw"].astype(np.float32)
    sae = np.load(out_dir / "sae_latents_all.npy").astype(np.float32)
    if len(raw) != len(meta) or len(sae) != len(meta):
        raise ValueError(f"Row mismatch: raw={len(raw)} sae={len(sae)} meta={len(meta)}")

    rng = np.random.default_rng(7)
    rp_dim = sae.shape[1]
    random_matrix = rng.normal(size=(raw.shape[1], rp_dim)).astype(np.float32) / np.sqrt(raw.shape[1])
    random_projection = raw @ random_matrix
    shuffled_sae = sae[rng.permutation(len(sae))]
    pca_dim = min(rp_dim, raw.shape[0] - 2, raw.shape[1])
    pca = make_pipeline(StandardScaler(with_mean=True), PCA(n_components=pca_dim, random_state=7)).fit_transform(raw)

    feature_sets = {
        "raw": raw,
        "pca": pca,
        "sae": sae,
        "random_projection": random_projection,
        "shuffled_sae": shuffled_sae,
    }

    n_splits = int(cfg.get("encoding", {}).get("n_splits", 3))
    n_splits = min(n_splits, max(2, len(meta) // 20))
    alphas = [float(a) for a in cfg.get("encoding", {}).get("alphas", [1.0, 10.0, 100.0])]

    rows: list[dict[str, object]] = []
    for roi_file in roi_files:
        roi = roi_file.stem.replace(f"{subject}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_", "")
        y = _align_roi(np.load(roi_file), orig_idx)
        if y is None:
            print(f"Skipping unusable ROI: {roi_file.name}")
            continue
        print(f"ROI {roi}: {y.shape}")
        for feature_set, x in feature_sets.items():
            rows.append({"subject": subject, "roi": roi, "feature_set": feature_set, **_cv_encode(x, y, n_splits, alphas)})

    df = pd.DataFrame(rows)
    csv_path = out_dir / "roi_encoding_summary.csv"
    json_path = out_dir / "roi_encoding_summary.json"
    df.to_csv(csv_path, index=False)
    summary = {
        "n_rois": int(df["roi"].nunique()),
        "n_rows": int(len(df)),
        "best_counts": df.sort_values("mean_r", ascending=False).groupby("roi").head(1)["feature_set"].value_counts().to_dict(),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== BEST FEATURE SET PER ROI ===")
    print(df.sort_values("mean_r", ascending=False).groupby("roi").head(1).sort_values("mean_r", ascending=False).to_string(index=False))
    print("\n=== FEATURE SET AVERAGES ACROSS ROIs ===")
    print(df.groupby("feature_set")[["mean_r", "median_r", "top5_percent_mean_r"]].mean().sort_values("mean_r", ascending=False).to_string())
    print(f"\nSaved {csv_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()

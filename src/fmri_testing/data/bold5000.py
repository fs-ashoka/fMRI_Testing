"""BOLD5000 beta preparation and synthetic debug data generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from fmri_testing.utils.io import save_npz


def _subject_file_candidates(bold_dir: Path, subject: str, preferred_regex: str | None) -> list[Path]:
    """Find local beta files for one subject."""
    suffixes = (".nii", ".nii.gz", ".npz", ".npy")
    files = [p for p in bold_dir.rglob("*") if p.is_file() and p.name.endswith(suffixes) and subject.lower() in p.name.lower()]
    if preferred_regex:
        pattern = re.compile(preferred_regex, flags=re.IGNORECASE)
        preferred = [p for p in files if pattern.search(p.name)]
        if preferred:
            return sorted(preferred)
    return sorted(files)


def _load_beta_file(path: Path) -> np.ndarray:
    """Load a beta file and return an array shaped trials x voxels."""
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        loaded = np.load(path)
        key = "betas" if "betas" in loaded else loaded.files[0]
        arr = loaded[key]
    elif path.name.endswith(".nii") or path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except Exception as exc:  # pragma: no cover
            raise ImportError("nibabel is required to load BOLD5000 NIfTI beta files") from exc
        arr = np.asanyarray(nib.load(str(path)).dataobj)
    else:
        raise ValueError(f"Unsupported beta file type: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        # Common NIfTI layout: X x Y x Z x trials.
        arr = np.moveaxis(arr, -1, 0).reshape(arr.shape[-1], -1)
    elif arr.ndim == 3:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Expected 2D/3D/4D beta array, got shape {arr.shape} from {path}")
    return arr.astype(np.float32)


def _zscore_session(arr: np.ndarray) -> np.ndarray:
    """Z-score trials within a beta session per voxel."""
    mu = np.nanmean(arr, axis=0, keepdims=True)
    sd = np.nanstd(arr, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    z = (arr - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def _valid_voxel_mask(betas: np.ndarray) -> np.ndarray:
    finite = np.isfinite(betas).mean(axis=0) > 0.99
    var = np.nanvar(betas, axis=0) > 1e-8
    return finite & var


def _synthetic_betas(cfg: dict[str, Any], subject: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Create deterministic synthetic betas and metadata for debug runs."""
    n_trials = int(cfg["data"].get("synthetic_trials", 48))
    n_voxels = int(cfg["data"].get("max_voxels", 64) or 64)
    seed = int(cfg.get("project", {}).get("seed", 0)) + sum(ord(c) for c in subject)
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_trials, 6)).astype(np.float32)
    weights = rng.normal(size=(6, n_voxels)).astype(np.float32)
    betas = latent @ weights + 0.25 * rng.normal(size=(n_trials, n_voxels)).astype(np.float32)
    betas = _zscore_session(betas)
    image_names = [f"synthetic_{i:04d}.png" for i in range(n_trials)]
    meta = pd.DataFrame(
        {
            "subject": subject,
            "session": 1,
            "trial": np.arange(n_trials),
            "beta_row": np.arange(n_trials),
            "image_name": image_names,
            "image_id": image_names,
        }
    )
    return betas.astype(np.float32), meta


def prepare_subject_betas(cfg: dict[str, Any], subject: str) -> tuple[Path, Path, Path]:
    """Prepare one subject's BOLD5000 betas or synthetic debug data.

    Outputs:
      prepared_betas_subject_<subject>.npz with key ``betas``
      trial_metadata_subject_<subject>.csv
      voxel_mask_subject_<subject>.npz with key ``mask``
    """
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    bold_dir = Path(cfg["paths"]["bold5000_dir"])
    preferred = cfg.get("data", {}).get("preferred_regex")
    files = _subject_file_candidates(bold_dir, subject, preferred)

    if not files:
        if cfg.get("data", {}).get("synthetic_if_missing", False):
            betas, meta = _synthetic_betas(cfg, subject)
            mask = np.ones(betas.shape[1], dtype=bool)
        else:
            raise FileNotFoundError(
                f"No local beta files found for {subject} under {bold_dir}. Run scripts/00_download_bold5000.py or enable synthetic_if_missing."
            )
    else:
        sessions: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        beta_offset = 0
        for session_idx, path in enumerate(files, start=1):
            arr = _zscore_session(_load_beta_file(path))
            sessions.append(arr)
            for trial in range(arr.shape[0]):
                rows.append(
                    {
                        "subject": subject,
                        "session": session_idx,
                        "source_file": path.name,
                        "trial": trial,
                        "beta_row": beta_offset + trial,
                        "image_name": f"{subject}_session{session_idx:02d}_trial{trial:05d}.png",
                        "image_id": f"{subject}_session{session_idx:02d}_trial{trial:05d}",
                    }
                )
            beta_offset += arr.shape[0]
        betas = np.vstack(sessions).astype(np.float32)
        mask = _valid_voxel_mask(betas)
        max_voxels = cfg.get("data", {}).get("max_voxels")
        if max_voxels:
            valid_idx = np.flatnonzero(mask)[: int(max_voxels)]
            limited_mask = np.zeros_like(mask, dtype=bool)
            limited_mask[valid_idx] = True
            mask = limited_mask
        betas = betas[:, mask]
        meta = pd.DataFrame(rows)

    beta_path = out / f"prepared_betas_subject_{subject}.npz"
    meta_path = out / f"trial_metadata_subject_{subject}.csv"
    mask_path = out / f"voxel_mask_subject_{subject}.npz"
    save_npz(beta_path, betas=betas.astype(np.float32))
    meta.to_csv(meta_path, index=False)
    save_npz(mask_path, mask=mask.astype(bool))
    return beta_path, meta_path, mask_path

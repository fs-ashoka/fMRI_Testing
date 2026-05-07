"""Training utilities for the TopK SAE."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

from fmri_testing.sae.model import TopKSparseAutoencoder
from fmri_testing.utils.io import write_json


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def compute_latents(model: TopKSparseAutoencoder, x: np.ndarray, device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    """Compute sparse latents for all rows."""
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            _, sparse, _ = model(xb)
            outs.append(sparse.cpu().numpy())
    return np.vstack(outs).astype(np.float32)


def train_sae_from_features(cfg: dict[str, Any], feature_path: str | Path) -> tuple[Path, Path, Path, Path]:
    """Train SAE on raw activations and save model, latents, and metrics."""
    raw = np.load(feature_path)["raw"].astype(np.float32)
    x, mu, sd = _standardize(raw)
    train_idx, val_idx = train_test_split(np.arange(len(x)), test_size=0.2, random_state=int(cfg["project"].get("seed", 0)))
    sae_cfg = cfg["sae"]
    device = sae_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = TopKSparseAutoencoder(x.shape[1], int(sae_cfg["latent_dim"]), int(sae_cfg["top_k"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(sae_cfg["learning_rate"]), weight_decay=float(sae_cfg.get("weight_decay", 0.0)))
    loader = DataLoader(TensorDataset(torch.as_tensor(x[train_idx])), batch_size=int(sae_cfg.get("batch_size", 128)), shuffle=True)
    loss_fn = torch.nn.MSELoss()
    metrics: dict[str, Any] = {"train_reconstruction_loss": [], "validation_reconstruction_loss": []}
    for _epoch in trange(int(sae_cfg.get("epochs", 20)), desc="train sae"):
        model.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            recon, sparse, _ = model(xb)
            loss = loss_fn(recon, xb) + float(sae_cfg.get("l1_penalty", 0.0)) * sparse.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vb = torch.as_tensor(x[val_idx], dtype=torch.float32, device=device)
            val_loss = float(loss_fn(model(vb)[0], vb).cpu())
        metrics["train_reconstruction_loss"].append(float(np.mean(losses)))
        metrics["validation_reconstruction_loss"].append(val_loss)
    latents_all = compute_latents(model, x, device=device)
    freq = (latents_all != 0).mean(axis=0)
    var = latents_all.var(axis=0)
    metrics.update({
        "mean_l0": float((latents_all != 0).sum(axis=1).mean()),
        "dead_feature_count": int(np.sum(freq == 0)),
        "feature_activation_frequency": freq.tolist(),
        "feature_variance": var.tolist(),
    })
    ckpt = Path(cfg["paths"]["checkpoint_dir"]); ckpt.mkdir(parents=True, exist_ok=True)
    out = Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    model_path = ckpt / "sae_model.pt"
    torch.save({"state_dict": model.state_dict(), "input_mean": mu, "input_std": sd, "config": sae_cfg}, model_path)
    train_latents_path = out / "sae_latents_train.npy"
    all_latents_path = out / "sae_latents_all.npy"
    metrics_path = out / "sae_metrics.json"
    np.save(train_latents_path, latents_all[train_idx])
    np.save(all_latents_path, latents_all)
    write_json(metrics_path, metrics)
    return model_path, train_latents_path, all_latents_path, metrics_path

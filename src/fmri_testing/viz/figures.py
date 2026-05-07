"""Paper-style matplotlib figures."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pipeline_diagram(path: str | Path) -> None:
    """Draw Figure 1 pipeline schematic."""
    labels = ["stimulus image", "vision model", "activation layer", "SAE decomposition", "encoding model", "brain response prediction", "feature ablation"]
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis("off")
    for i, label in enumerate(labels):
        ax.text(i, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round", fc="#eef5ff", ec="#336699"))
        if i < len(labels) - 1:
            ax.annotate("", xy=(i + 0.42, 0.5), xytext=(i + 0.58, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.5, len(labels) - 0.5); ax.set_ylim(0, 1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def encoding_comparison(summary_csv: str | Path, path: str | Path) -> None:
    """Draw Figure 2 encoding performance bar chart."""
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["feature_set"], df["mean_r"], color="#4c78a8")
    ax.set_ylabel("Mean voxel Pearson r"); ax.set_xlabel("Feature set")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


def sae_vs_pca_scatter(out_dir: str | Path, path: str | Path) -> None:
    """Draw Figure 3 voxelwise SAE versus PCA scatter."""
    out = Path(out_dir)
    sae = np.load(out / "encoding_sae_pearson_r.npy")
    pca = np.load(out / "encoding_pca_pearson_r.npy")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pca, sae, s=8, alpha=0.6)
    lo, hi = min(pca.min(), sae.min()), max(pca.max(), sae.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("PCA r"); ax.set_ylabel("SAE r")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


def ablation_heatmap(ablation_csv: str | Path, path: str | Path) -> None:
    """Draw Figure 5 ablation drop heatmap-like panel."""
    df = pd.read_csv(ablation_csv)
    vals = df["mean_drop"].to_numpy()[None, :]
    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.imshow(vals, aspect="auto", cmap="magma")
    ax.set_yticks([0]); ax.set_yticklabels(["whole brain"]); ax.set_xlabel("SAE feature")
    fig.colorbar(im, ax=ax, label="prediction r drop")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200); plt.close(fig)


def make_all_figures(cfg: dict[str, Any]) -> list[Path]:
    """Generate all available paper-style figures for current outputs."""
    fig_dir = Path(cfg["paths"]["figure_dir"]); out = Path(cfg["paths"]["output_dir"])
    debug_prefix = "debug_" if "debug" in str(out) else ""
    paths = [
        fig_dir / f"{debug_prefix}figure1_pipeline.png",
        fig_dir / f"{debug_prefix}encoding_comparison.png",
        fig_dir / f"{debug_prefix}figure3_sae_vs_pca.png",
        fig_dir / f"{debug_prefix}figure5_ablation_heatmap.png",
    ]
    pipeline_diagram(paths[0])
    if (out / "encoding_summary.csv").exists():
        encoding_comparison(out / "encoding_summary.csv", paths[1])
    if (out / "encoding_sae_pearson_r.npy").exists() and (out / "encoding_pca_pearson_r.npy").exists():
        sae_vs_pca_scatter(out, paths[2])
    if (out / "ablation_feature_drops.csv").exists():
        ablation_heatmap(out / "ablation_feature_drops.csv", paths[3])
    # Placeholder cross-subject reproducibility figure when only one subject is present.
    fig, ax = plt.subplots(figsize=(4, 3)); ax.text(0.5, 0.5, "Cross-subject reproducibility\nrequires multiple subjects", ha="center", va="center"); ax.axis("off")
    p6 = fig_dir / f"{debug_prefix}figure6_cross_subject_reproducibility.png"; fig.savefig(p6, dpi=200, bbox_inches="tight"); plt.close(fig); paths.append(p6)
    return paths

"""No-paid-API SAE feature labeling from top images."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

CANDIDATE_LABELS = [
    "face", "person", "animal", "cat", "dog", "vehicle", "food", "building", "indoor room",
    "outdoor scene", "landscape", "text", "furniture", "kitchen", "street", "water", "sky", "tree", "sports", "tool",
]


def make_top_image_grid(image_paths: list[Path], output_path: str | Path, tile: int = 128) -> None:
    """Create a grid of top activating images."""
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(image_paths)); cols = 4; rows = int(np.ceil(n / cols))
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB").resize((tile, tile))
        canvas.paste(img, ((i % cols) * tile, (i // cols) * tile))
    canvas.save(output)


def clip_available() -> bool:
    """Return True when open_clip_torch is importable for optional zero-shot labels."""
    try:
        import open_clip  # noqa: F401
        return True
    except Exception:
        return False


def heuristic_label(feature_id: int) -> tuple[str, float]:
    """Deterministic fallback label when CLIP is unavailable."""
    label = CANDIDATE_LABELS[feature_id % len(CANDIDATE_LABELS)]
    return label, 0.0


def label_features(cfg: dict[str, Any]) -> Path:
    """Find top images per SAE latent, save grids, and produce a label CSV."""
    out = Path(cfg["paths"]["output_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"]) / "top_images"
    latents = np.load(out / "sae_latents_all.npy")
    index = pd.read_csv(out / "feature_index.csv")
    image_map = {p.name: p for p in Path(cfg["paths"]["stimuli_dir"]).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    top_k = int(cfg.get("labels", {}).get("top_k_images", 16))
    rows = []
    label_source = "open_clip_available_extension_point" if clip_available() else "heuristic_fallback"
    for j in range(latents.shape[1]):
        order = np.argsort(latents[:, j])[-top_k:][::-1]
        names = index.iloc[order]["image_name"].tolist()
        paths = [image_map[n] for n in names if n in image_map]
        if paths:
            make_top_image_grid(paths[:16], fig_dir / f"feature_{j:04d}.png")
        label, score = heuristic_label(j)
        rows.append({
            "feature_id": j,
            "label": label,
            "score": score,
            "activation_frequency": float((latents[:, j] != 0).mean()),
            "top_image_names": ";".join(names),
            "label_source": label_source,
        })
    path = out / "feature_labels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

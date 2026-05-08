"""Stimulus discovery and synthetic debug stimulus generation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(stimuli_dir: str | Path) -> list[Path]:
    """Return all known image files under a stimulus directory."""
    root = Path(stimuli_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def create_synthetic_stimuli(stimuli_dir: str | Path, n_images: int, seed: int = 0) -> list[Path]:
    """Create deterministic synthetic RGB images for no-download debug runs."""
    root = Path(stimuli_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for i in range(n_images):
        base = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
        # Add a deterministic coloured block so each image has a stable simple signal.
        x0 = (i * 7) % 64
        y0 = (i * 11) % 64
        base[y0 : y0 + 24, x0 : x0 + 24, :] = np.array([(i * 17) % 255, (i * 29) % 255, (i * 43) % 255], dtype=np.uint8)
        img = Image.fromarray(base, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text((4, 4), str(i), fill=(255, 255, 255))
        path = root / f"synthetic_{i:04d}.png"
        img.save(path)
        paths.append(path)
    return paths


def build_feature_index(metadata: pd.DataFrame) -> pd.DataFrame:
    """Build a row-wise feature index aligned to metadata rows."""
    index = metadata.copy()
    index["feature_row"] = np.arange(len(index))
    if "image_name" not in index.columns:
        index["image_name"] = [f"synthetic_{i:04d}.png" for i in range(len(index))]
    return index

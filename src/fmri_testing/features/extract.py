"""Feature extraction entry points."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from fmri_testing.data.stimuli import build_feature_index, create_synthetic_stimuli, find_images
from fmri_testing.features.vision_models import VisionFeatureExtractor


def extract_features(cfg: dict[str, Any], metadata_path: str | Path) -> tuple[Path, Path]:
    """Extract model features in metadata row order and write arrays plus index."""
    meta = pd.read_csv(metadata_path)
    stimuli_dir = Path(cfg["paths"]["stimuli_dir"])
    if cfg["data"].get("synthetic_if_missing", False) and not find_images(stimuli_dir):
        create_synthetic_stimuli(stimuli_dir, len(meta), int(cfg["project"].get("seed", 0)))
    image_map = {p.name: p for p in find_images(stimuli_dir)}
    paths = [image_map.get(name) for name in meta["image_name"]]
    if any(p is None for p in paths):
        missing = [n for n, p in zip(meta["image_name"], paths) if p is None][:5]
        raise FileNotFoundError(f"Missing stimulus images for metadata rows, examples: {missing}")
    extractor = VisionFeatureExtractor(
        cfg["features"]["model_name"], cfg["features"]["layers"], cfg["features"].get("device", "cpu"), cfg["features"].get("pretrained", True)
    )
    if "synthetic_dim" in cfg["features"]:
        extractor.synthetic_dim = int(cfg["features"]["synthetic_dim"])
    batch_size = int(cfg["features"].get("batch_size", 16))
    layer_chunks: dict[str, list[np.ndarray]] = {layer: [] for layer in cfg["features"]["layers"]}
    for start in tqdm(range(0, len(paths), batch_size), desc="extract features"):
        imgs = [Image.open(p) for p in paths[start : start + batch_size] if p is not None]
        acts = extractor.extract_batch(imgs)
        for layer, arr in acts.items():
            layer_chunks.setdefault(layer, []).append(arr)
    arrays = {layer: np.vstack(chunks) for layer, chunks in layer_chunks.items() if chunks}
    raw = np.concatenate([arrays[layer] for layer in sorted(arrays)], axis=1).astype(np.float32)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "vision_features.npz"
    np.savez_compressed(feat_path, raw=raw, **arrays)
    index = build_feature_index(meta)
    index_path = out_dir / "feature_index.csv"
    index.to_csv(index_path, index=False)
    return feat_path, index_path

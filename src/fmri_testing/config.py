"""Configuration loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` with ``override`` and return a new dict."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, defaults: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file, optionally overlaying it on defaults."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if defaults is not None and Path(defaults).resolve() != cfg_path.resolve():
        with Path(defaults).open("r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        cfg = deep_update(base, cfg)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create configured output, data, checkpoint, and figure directories."""
    for key in ["data_dir", "bold5000_dir", "stimuli_dir", "output_dir", "checkpoint_dir", "figure_dir"]:
        value = cfg.get("paths", {}).get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)

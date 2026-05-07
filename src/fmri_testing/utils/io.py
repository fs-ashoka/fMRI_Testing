"""Input/output helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable indentation."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read JSON data."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Save compressed NumPy arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

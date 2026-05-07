#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.sae.train import train_sae_from_features
from fmri_testing.utils.logging import setup_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); args = p.parse_args()
    cfg = load_config(args.config, "configs/default.yaml"); ensure_dirs(cfg); log = setup_logging()
    paths = train_sae_from_features(cfg, Path(cfg["paths"]["output_dir"]) / "vision_features.npz"); log.info("SAE outputs: %s", paths)
if __name__ == "__main__": main()

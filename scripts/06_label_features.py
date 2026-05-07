#!/usr/bin/env python
from __future__ import annotations
import argparse
from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.features.feature_labels import label_features

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); args = p.parse_args()
    cfg = load_config(args.config, "configs/default.yaml"); ensure_dirs(cfg); label_features(cfg)
if __name__ == "__main__": main()

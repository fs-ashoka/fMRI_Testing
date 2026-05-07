#!/usr/bin/env python
from __future__ import annotations
import argparse
from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.ablation.feature_ablation import run_ablation_pipeline
from fmri_testing.utils.logging import setup_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); args = p.parse_args()
    cfg = load_config(args.config, "configs/default.yaml"); ensure_dirs(cfg); log = setup_logging()
    subject = cfg["data"].get("subjects", ["CSI1"])[0]
    log.info("Ablating %s", subject); run_ablation_pipeline(cfg, subject)
if __name__ == "__main__": main()

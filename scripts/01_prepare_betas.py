#!/usr/bin/env python
from __future__ import annotations
import argparse
from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.data.bold5000 import prepare_subject_betas
from fmri_testing.utils.logging import setup_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); args = p.parse_args()
    cfg = load_config(args.config, "configs/default.yaml"); ensure_dirs(cfg); log = setup_logging()
    for subject in cfg["data"].get("subjects", ["CSI1"]):
        paths = prepare_subject_betas(cfg, subject); log.info("Prepared %s: %s", subject, paths)
if __name__ == "__main__": main()

#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.features.extract import extract_features
from fmri_testing.utils.logging import setup_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/default.yaml"); p.add_argument("--subject", default=None); args = p.parse_args()
    cfg = load_config(args.config, "configs/default.yaml"); ensure_dirs(cfg); log = setup_logging()
    subject = args.subject or cfg["data"].get("subjects", ["CSI1"])[0]
    meta = Path(cfg["paths"]["output_dir"]) / f"trial_metadata_subject_{subject}.csv"
    log.info("Extracting features using %s", meta); extract_features(cfg, meta)
if __name__ == "__main__": main()

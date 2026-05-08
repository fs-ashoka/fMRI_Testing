#!/usr/bin/env python
"""List or download BOLD5000 Release 2.0 files through public Figshare API."""
from __future__ import annotations

import argparse
from pathlib import Path

from fmri_testing.config import ensure_dirs, load_config
from fmri_testing.data.figshare import download_selected, fetch_article_metadata, files_to_dataframe, filter_files, parse_figshare_files
from fmri_testing.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--regex", default=None)
    parser.add_argument("--subject", action="append", default=None)
    parser.add_argument("--min-free-gb", type=float, default=None, help="Override config data.min_free_gb for small smoke-test downloads.")
    parser.add_argument("--full", action="store_true", help="Download all matching files; otherwise subject/regex defaults apply.")
    args = parser.parse_args()
    log = setup_logging()
    cfg = load_config(args.config, "configs/default.yaml")
    ensure_dirs(cfg)
    article = fetch_article_metadata(int(cfg["data"]["figshare_article_id"]))
    files = parse_figshare_files(article)
    df = files_to_dataframe(files)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(cfg["paths"]["output_dir"]) / "figshare_files.csv", index=False)
    regex = args.regex or cfg["data"].get("preferred_regex")
    subjects = args.subject or cfg["data"].get("subjects")
    selected = files if args.full else filter_files(files, regex=regex, subjects=subjects)
    selected_df = files_to_dataframe(selected)
    log.info("Article has %d files; selected %d files", len(files), len(selected))
    print(selected_df.to_string(index=False))
    if args.list_only:
        return
    if args.full and not cfg["data"].get("download_full", False):
        raise RuntimeError("Full download requested but config data.download_full is false. Set it true to acknowledge large download.")
    min_free_gb = args.min_free_gb if args.min_free_gb is not None else float(cfg["data"].get("min_free_gb", 1))
    download_selected(selected, cfg["paths"]["bold5000_dir"], float(min_free_gb))

if __name__ == "__main__":
    main()

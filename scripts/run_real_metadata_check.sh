#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/00_download_bold5000.py --config configs/real_metadata_only.yaml --list-only
python scripts/check_bold5000_public_metadata.py

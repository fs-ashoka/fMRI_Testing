#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG="configs/full_bold5000.yaml"
echo "Safety: this script does not download the full 121 GB dataset automatically."
echo "First inspect public Figshare files:"
python scripts/00_download_bold5000.py --config "$CONFIG" --list-only
if [[ "${FMRI_TESTING_RUN_FULL:-0}" != "1" ]]; then
  echo "Set FMRI_TESTING_RUN_FULL=1 after confirming selected BOLD5000 files and stimuli are present."
  exit 1
fi
python scripts/01_prepare_betas.py --config "$CONFIG"
python scripts/02_extract_vision_features.py --config "$CONFIG"
python scripts/03_train_sae.py --config "$CONFIG"
python scripts/04_fit_encoding_models.py --config "$CONFIG"
python scripts/05_run_feature_ablation.py --config "$CONFIG"
python scripts/06_label_features.py --config "$CONFIG"
python scripts/07_make_figures.py --config "$CONFIG"

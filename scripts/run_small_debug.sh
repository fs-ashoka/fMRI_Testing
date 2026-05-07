#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
CONFIG="configs/small_debug.yaml"
python scripts/01_prepare_betas.py --config "$CONFIG"
python scripts/02_extract_vision_features.py --config "$CONFIG"
python scripts/03_train_sae.py --config "$CONFIG"
python scripts/04_fit_encoding_models.py --config "$CONFIG"
python scripts/05_run_feature_ablation.py --config "$CONFIG"
python scripts/06_label_features.py --config "$CONFIG"
python scripts/07_make_figures.py --config "$CONFIG"

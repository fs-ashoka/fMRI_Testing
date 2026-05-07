# Mechanistic Feature Alignment Between Vision Models and Human fMRI

This repository implements a reproducible Python research pipeline for **mechanistic interpretability inspired brain alignment**. It tests whether sparse, interpretable features extracted from open vision-model activations align with BOLD5000 human fMRI beta responses better than raw activations, PCA features, random projections, or shuffled sparse-autoencoder (SAE) features.

Central question:

> Do sparse autoencoder features extracted from open vision models provide more interpretable and region-selective alignment with human visual fMRI responses than raw hidden activations or PCA features?

The project deliberately makes voxel-level and region-level claims only. It does **not** claim neuron-level interpretability or causal mechanisms inside the human brain. SAE ablation is causal inside the model representation and predictive with respect to fMRI responses.

## Why BOLD5000?

BOLD5000 Release 2.0 provides a large natural-image fMRI resource with repeated visual stimulus presentations and public optimized beta files. The default recommended beta processing in this repository targets files matching `TYPED-FITHRF-GLMDENOISE-RR`, the optimized GLM-denoised beta estimates. Public metadata are accessed through the no-token Figshare API article `14456124`.

Dataset and project links:

- Dataset page: <https://kilthub.cmu.edu/articles/dataset/BOLD5000_Release_2_0/14456124>
- Public Figshare API endpoint: <https://api.figshare.com/v2/articles/14456124>
- Official code repository: <https://github.com/BOLD5000-dataset/BOLD5000>
- Stimulus image download notes: <https://bold5000-dataset.github.io/website/download.html>

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is required. Optional `open_clip_torch` support can be installed separately, but the core project does not require it and does not use paid APIs.

## Small debug run

The debug configuration creates synthetic stimuli and synthetic beta responses when BOLD5000 files are not present. This verifies the complete workflow quickly without downloading large data.

```bash
bash scripts/run_small_debug.sh
```

Expected debug outputs:

- `outputs/debug/metrics.json`
- `outputs/debug/encoding_summary.csv`
- `outputs/debug/ablation_summary.json`
- `figures/debug_encoding_comparison.png`

## Listing and downloading BOLD5000 files

List all selected public Figshare files without downloading:

```bash
python scripts/00_download_bold5000.py --config configs/default.yaml --list-only
```

Download the default small recommended subset for one subject, selected by the configured `TYPED-FITHRF-GLMDENOISE-RR` regex and subject list:

```bash
python scripts/00_download_bold5000.py --config configs/default.yaml
```

Download a narrower custom subset:

```bash
python scripts/00_download_bold5000.py --config configs/default.yaml --regex 'TYPED-FITHRF-GLMDENOISE-RR.*CSI1' --subject CSI1
```

The full BOLD5000 Release 2.0 collection is very large. The repository intentionally does not initiate a 121 GB download unless explicitly configured with `data.download_full: true` and `--full`.

## Full pipeline

After placing the selected BOLD5000 beta files and stimulus images under the configured `data/` paths, run:

```bash
bash scripts/run_full_pipeline.sh
```

The script first lists public Figshare files as a safety check and then runs preparation, feature extraction, SAE training, encoding, ablation, feature labeling, and figure generation.

## Pipeline stages

1. **Download/list metadata**: public Figshare API, no token.
2. **Prepare fMRI betas**: load NIfTI with nibabel, flatten 4D sessions into trials by voxels, z-score within session, remove near-zero-variance voxels, and use ROI masks when supplied.
3. **Extract vision activations**: ResNet50 layers `layer2`, `layer3`, `layer4`, and `avgpool`; torchvision ViT-B-16 when available; optional no-login backends are guarded.
4. **Train TopK SAE**: linear encoder, TopK sparse latents, decoder reconstruction, reconstruction metrics, L0, dead features, activation frequencies, and variances.
5. **Fit encoding models**: ridge regression with image-identity grouped cross-validation.
6. **Compare controls**: raw activations, PCA at equal dimension, random projection, SAE, and shuffled SAE.
7. **Run ablations**: zero SAE features and measure voxel/ROI prediction drops, with matched-control scaffolding by feature frequency and variance.
8. **Label features**: top activating images and no-paid-API heuristic labels, with optional CLIP extension point.
9. **Make figures**: pipeline diagram, encoding comparisons, SAE-vs-PCA scatter, top images, ablation heatmap, and cross-subject reproducibility placeholder when only one subject is available.

## Claims allowed

- SAE features can be evaluated as a sparse, interpretable model-representation basis for predicting voxel-level or ROI-level fMRI responses.
- Encoding and ablation results can show predictive alignment, region selectivity, and model-representation sensitivity.
- Negative or null results are valid: if SAE does not outperform PCA or raw activations, report that honestly as a reproducible test.

## Claims not allowed

- Do not claim human neuron-level mechanistic interpretability.
- Do not claim SAE features are causal mechanisms in the human brain.
- Do not infer biological mechanisms from predictive ablation alone.

## Testing

```bash
pytest
```

## Repository layout

- `configs/`: reproducible small, default, and full configurations.
- `src/fmri_testing/`: data, features, SAE, encoding, ablation, stats, visualization, and utility code.
- `scripts/`: numbered command-line stages and end-to-end runners.
- `docs/`: methods, paper outline, and data notes.
- `tests/`: unit tests for metadata parsing, SAE sparsity/shapes, encoding, and ablation.

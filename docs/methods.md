# Methods

## Overview

We implement a reproducible test of mechanistic interpretability inspired brain alignment between open vision models and human fMRI responses. The analysis asks whether sparse autoencoder (SAE) features derived from model activations explain voxel-level or ROI-level BOLD5000 responses as well as or better than raw activations, PCA features of matched dimension, random projections, and shuffled SAE features.

## Dataset

The primary dataset is BOLD5000 Release 2.0, accessed through the public Figshare API article `14456124`. The preferred beta files match `TYPED-FITHRF-GLMDENOISE-RR`, corresponding to optimized GLM-denoised beta estimates. Download utilities list all public files and support regex and subject filters to avoid downloading the complete release by default.

## fMRI beta preparation

For each subject, 4D NIfTI beta files are loaded with nibabel memory mapping. Each session is transformed from spatial dimensions by trials into a two-dimensional matrix of `n_trials x n_voxels`. Betas are z-scored within session and per voxel. Voxels with near-zero variance or excessive non-finite values are removed. If ROI masks are available, analyses can be restricted or summarized by ROI; otherwise, the output is explicitly labeled as whole-brain valid-voxel analysis.

## Vision-model features

Stimulus images are passed through open models. The required implementation supports torchvision ResNet50 layers `layer2`, `layer3`, `layer4`, and `avgpool`, and torchvision ViT-B-16 when installed in the local torchvision version. Optional CLIP or DINOv2 support is guarded so unavailable models do not break the pipeline. Spatial convolutional activations are average-pooled into one vector per image and concatenated across configured layers.

## Sparse autoencoder

The SAE uses a linear encoder, TopK sparsification, and a linear decoder. For each sample, only the `k` largest absolute latent activations are retained. The training objective is reconstruction mean squared error plus an optional L1 penalty. Metrics include train and validation reconstruction loss, mean L0, dead feature count, feature activation frequency, and feature variance.

## Baselines and controls

Encoding comparisons include raw activations, PCA features with dimensionality matched to the SAE latent dimension, Gaussian random projection features with matched dimensionality, SAE latents, and shuffled SAE latents as a negative control.

## Encoding models

Ridge regression predicts fMRI beta responses from each feature set. Cross-validation is grouped by image identity to prevent repeated presentations of the same stimulus from leaking across train and test folds. Metrics include Pearson correlation per voxel, R2 per voxel, mean correlation, median correlation, top-five-percent voxel mean correlation, and ROI means when ROI masks exist.

## Statistical testing

The project includes bootstrap utilities for paired mean differences over images or voxels, sign-flip permutation tests for paired comparisons, and Benjamini-Hochberg FDR correction for voxel-level or ROI-level comparisons. The main planned comparisons are SAE versus PCA, SAE versus raw activations, SAE versus shuffled SAE, and SAE versus random projections.

## Feature ablation

For SAE feature ablation, a trained ridge model is evaluated after zeroing one SAE latent feature at a time. Prediction drops are computed as changes in voxelwise Pearson correlation. The ablation is causal within the model representation and predictive with respect to fMRI; it is not evidence for causal mechanisms in the human brain.

## Feature interpretability

For each SAE feature, top activating images are collected and saved as image grids. Candidate labels come from available metadata and a fixed list of simple categories. Optional CLIP zero-shot labeling can be added without paid APIs; a deterministic heuristic fallback is provided for reproducible no-login runs.

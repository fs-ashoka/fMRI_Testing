# Paper Outline

## Title

Mechanistic Feature Alignment Between Vision Models and Human fMRI

## Abstract draft

Sparse feature decompositions have improved the interpretability of artificial neural network representations, but it remains unclear whether such features provide a useful bridge to human brain responses measured at the voxel or region level. We present a reproducible no-login, no-payment pipeline using BOLD5000 Release 2.0 to compare raw vision-model activations, PCA features, random projections, shuffled sparse-autoencoder controls, and TopK sparse-autoencoder features for predicting visual fMRI beta responses. Ridge encoding models are evaluated with image-identity grouped cross-validation. We quantify predictive performance, feature interpretability through top activating images, and representational sensitivity through SAE-feature ablations. The project frames results as mechanistic interpretability inspired brain alignment and avoids neuron-level or causal brain claims.

## Introduction outline

- Vision models as computational hypotheses for visual representation.
- Encoding models for voxel-level and ROI-level fMRI prediction.
- Mechanistic interpretability and sparse feature decompositions in machine learning.
- Gap: sparse model features have not been systematically tested as interpretable fMRI predictors with strong controls.
- Research question and hypotheses.

## Methods outline

- BOLD5000 Release 2.0 dataset and preferred beta files.
- Stimulus indexing and beta preprocessing.
- Vision-model activation extraction.
- TopK sparse autoencoder architecture and training.
- Baselines: raw, PCA, random projection, shuffled SAE.
- Grouped ridge encoding models.
- Bootstrap, permutation, and FDR procedures.
- Feature ablation and top-image labeling.

## Results outline

- Figure 1: pipeline diagram.
- Figure 2: encoding performance across feature sets.
- Figure 3: SAE versus PCA voxelwise scatter.
- Figure 4: top images and labels for selected SAE features.
- Figure 5: ablation drop heatmap.
- Figure 6: cross-subject reproducibility of feature effects.

## Discussion outline

- Whether sparse features improve, match, or underperform PCA/raw features.
- Interpretability advantages and limitations of top-image labels.
- Region-selective predictive effects versus causal brain mechanisms.
- Negative controls and what they rule out.

## Limitations

- fMRI is voxel-level and region-level, not neuron-level.
- BOLD responses are indirect and low temporal resolution.
- Feature labels are approximate and depend on candidate categories.
- Encoding models are predictive, not mechanistic brain interventions.
- Full BOLD5000 analysis requires substantial storage and compute.

## Future work

- Add ROI-specific masks and anatomical reporting.
- Compare more open vision models and layers.
- Add cross-subject feature-effect reliability metrics.
- Use richer no-paid labeling sources where licenses permit.
- Extend to temporal or recurrent visual models.

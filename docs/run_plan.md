# Run Plan

A. Debug run

Run the small debug shell script. It should create metrics, encoding summary, ablation summary, and a debug figure.

B. Metadata run

Run the metadata shell script. It should create public BOLD5000 manifest files without downloading the large dataset.

C. One subject run

Use one CSI1 recommended beta file, prepare betas, extract features, train the SAE, and fit encoding models.

D. Full run

Use all available subjects and compare raw, PCA, SAE, random projection, and shuffled SAE features.

E. Reporting

Report voxel-level and region-level results only. Do not present the output as neuron-level evidence.

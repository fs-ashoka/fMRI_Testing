# Data Notes

## Dataset source

Primary data come from BOLD5000 Release 2.0:

- Dataset page: <https://kilthub.cmu.edu/articles/dataset/BOLD5000_Release_2_0/14456124>
- Public Figshare API article id: `14456124`
- API endpoint: <https://api.figshare.com/v2/articles/14456124>
- Official code: <https://github.com/BOLD5000-dataset/BOLD5000>
- Stimulus download page: <https://bold5000-dataset.github.io/website/download.html>

## License notes

This repository contains code only and does not redistribute BOLD5000 data, stimuli, NIfTI files, model checkpoints, or derived large arrays. Users are responsible for checking BOLD5000 dataset terms and any stimulus-source licenses before publication or redistribution.

## Download method

`scripts/00_download_bold5000.py` uses `requests` to call the public Figshare API, parses the returned `files` list, writes `figshare_files.csv`, and downloads selected files by regex and subject filter. Public Figshare files do not require a token.

## File size warning

The full BOLD5000 Release 2.0 collection is approximately 121 GB. The default configuration selects recommended `TYPED-FITHRF-GLMDENOISE-RR` files for one subject rather than downloading everything. Full download requires explicit configuration.

## Expected folder layout

```text
data/
  bold5000/
    <downloaded BOLD5000 beta files and masks>
  stimuli/
    <BOLD5000 stimulus images>
outputs/
  default or debug outputs
checkpoints/
  sae_model.pt
figures/
  generated paper-style figures
```

Generated data and figures are intentionally ignored by Git.

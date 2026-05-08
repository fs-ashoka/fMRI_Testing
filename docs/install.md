# Install notes

Use Python 3.10 or newer.

## pip

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## conda

```bash
conda env create -f environment.yml
conda activate fmri-testing
python -m pip install -e .
```

## uv

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install uv
uv pip install -r requirements.txt
uv pip install -e .
```

## Verify without data

```bash
bash scripts/run_small_debug.sh
```

Expected files:

```text
outputs/debug/metrics.json
outputs/debug/encoding_summary.csv
outputs/debug/ablation_summary.json
figures/debug_encoding_comparison.png
```

## Verify public BOLD5000 metadata without download

```bash
bash scripts/check_bold5000_metadata.sh
```

This writes:

```text
outputs/bold5000_file_manifest.csv
outputs/bold5000_file_manifest.json
```

## Notes

Some restricted sandboxes block package-index access. That is an environment problem, not a code problem. Use a normal local machine, Colab, GitHub Codespaces, or conda if pip is blocked.

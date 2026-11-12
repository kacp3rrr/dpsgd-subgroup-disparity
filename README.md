# DPSGD Subgroup Disparity

Reproduction and analysis of disparate impact in DP-SGD (Esipova et al., ICLR 2023). 

## Setup

**Requirements:**

- [uv](https://docs.astral.sh/uv/). It manages Python 3.12 and every package for you.
- [Quarto](https://quarto.org/docs/get-started/) ≥ 1.5, only needed to build the site.

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create .venv with Python 3.12 + all dependencies (pinned in uv.lock)
uv sync

# 3. Check it worked
uv run python -c "import torch, opacus; print(torch.__version__, torch.cuda.is_available())"
```

On Linux and Windows, PyTorch installs with CUDA 12.8. This version is required for RTX 50-series GPUs, and it also runs on CPU-only machines. On macOS it installs from PyPI.

## Usage

```bash
uv run jupyter lab        # work on notebooks in notebooks/
uv run pytest             # run tests
uv run python scripts/sweep.py --dataset adult   # (re)train 3 methods x 5 seeds into results/adult/ (~15 min CPU; MNIST: use a GPU)
quarto preview            # live-preview the site
quarto render             # build the site into _site/
```

## Layout

```
dpsgd-subgroup-disparity/
├── src/
│   └── dpdisparity/           # importable package: data, models, training, DP, analysis, plots
│       └── data/              # dataset loaders
├── notebooks/                 # one notebook per site page; only call the package
├── scripts/                   # CLI for multi-seed runs
├── results/
│   ├── legacy/                # frozen dp-testing outputs (shown by default)
│   └── adult/                 # reruns from this repo
├── theme/                     # site styles (light / dark)
├── tests/                     # pytest checks
└── data/                      # raw dataset downloads (gitignored)
```

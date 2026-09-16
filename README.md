# Dual-Encoder Cross-Attention for Volatility Forecasting

This repository contains an undergraduate capstone project on multi-horizon,
probabilistic volatility forecasting. The project studies whether daily
text-derived macroeconomic signals improve forecasts of realized volatility
for U.S. equity ETFs, and whether a dual-encoder cross-attention architecture
uses those signals more effectively than standard feature concatenation.

## Project overview

The forecasting target is daily log realized volatility computed from
5-minute intraday returns. The dataset covers five U.S. ETFs:

- **SPY**: S&P 500
- **XLE**: Energy Select Sector
- **XLF**: Financial Select Sector
- **XLK**: Technology Select Sector
- **XLY**: Consumer Discretionary Select Sector

The sample runs from **2018-12-03 to 2024-01-30** and contains 1,297 trading
days per asset before rolling-feature warm-up. Four daily text-based indicators
provide exogenous information:

- Economic Policy Uncertainty (**EPU**)
- Geopolitical Risk (**GPR**)
- Trade Policy Uncertainty (**TPU**)
- Federal Reserve Bank of San Francisco Daily News Sentiment Index (**FNSI**)

A 120-day rolling PCA reduces these four indicators to two sentiment factors.
After the PCA warm-up, the modeling dataset contains 1,177 dates for each of
the five assets, or 5,885 asset-date observations.

The models produce quantile forecasts at the 0.1, 0.5, and 0.9 levels for
1-, 2-, 3-, 5-, and 10-trading-day horizons. The main dual-encoder model is
compared with:

- a HAR-RV quantile baseline;
- a price-only LSTM; and
- an LSTM using direct volatility/sentiment concatenation.

## Repository structure

```text
.
├── data/                   # Local data only (excluded from version control)
├── reports/
│   ├── figures/            # Publication-ready plots
│   │   ├── single_split/
│   │   └── cross_validation/
│   └── tables/             # Result tables and statistical summaries
│       ├── single_split/
│       └── cross_validation/
├── preprocess.ipynb        # Intraday RV and indicator preprocessing
├── results_notebook.ipynb  # Result tables and visualizations
├── data_prep.py            # Rolling PCA and long-format dataset creation
├── experiments.py          # Single train/test experiment
├── experiments_cv.py       # Walk-forward CV and statistical evaluation
├── models.py               # LSTM and dual-encoder model definitions
├── train.py                # Neural-network training loop
├── harrv.py                # HAR-RV quantile baseline
├── eval.py                 # Bootstrap and Diebold-Mariano tests
└── viz.py                  # Calibration and attention visualizations
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch automatically uses CUDA when it is available; otherwise, the scripts
run on CPU.

## Running the project

Run all commands from the repository root.

### 1. Prepare the model-ready dataset

Data files are not included in this repository. Create a local `data/`
directory and provide either:

- `data/dataset.csv`, containing the aligned daily RV and four indicator
  series; or
- `data/consolidated.csv`, containing the final long-format modeling data.

If `data/dataset.csv` is available, generate the long-format dataset and the
rolling PCA factors with:

```bash
python data_prep.py
```

This creates `data/consolidated.csv`, which is the direct input to the modeling
scripts. At minimum, the final file must contain the columns `Date`, `asset`,
`lrv`, `pc1`, and `pc2`.

To rebuild `data/dataset.csv` from source files, place the five intraday ETF
files and the EPU, GPR, TPU, and FNSI files in the local `data/` directory, then
use `preprocess.ipynb`. Its file-loading cells expect the working directory to
be `data/`.

### 2. Run the single-split experiment

```bash
python experiments.py
```

This uses a 60-day lookback and an 80/20 chronological train/test split. The
pooled losses are printed to the console, and working figures are written to
the local `results/` directory. This working directory is excluded from version
control; the curated GitHub outputs are stored under `reports/`.

### 3. Run walk-forward cross-validation

```bash
python experiments_cv.py --past-lens 20 60 120
```

The script first performs quick screening and then runs the full evaluation for
the selected lookback lengths. To explicitly request the full configuration or
parameter-budget matching, use:

```bash
python experiments_cv.py --past-lens 20 60 120 --full
python experiments_cv.py --past-lens 20 60 120 --full --equalize-params
```

Working cross-validation outputs are written to the local `results_cv/`
directory, which is excluded from version control. Curated tables and figures
are stored under `reports/tables/cross_validation/` and
`reports/figures/cross_validation/`. Training can take a substantial amount of
time on CPU.

### 4. Review the results

After generating local working outputs, open `results_notebook.ipynb` to inspect
aggregated loss matrices, calibration, per-asset results, attention
visualizations, bootstrap results, and Diebold-Mariano statistics. The curated
CSV and PNG outputs can be reviewed directly under `reports/` without rerunning
the notebook.

## Results

Lower pinball loss is better. In the saved single-split results, the pooled
mean pinball losses are:

| Model | Pooled pinball loss |
|---|---:|
| Dual encoder | **0.1550** |
| Concatenation LSTM | 0.1626 |
| HAR-RV | 0.1649 |
| Price-only LSTM | 0.1695 |

In the saved full cross-validation summary, the 60-day lookback gives the
lowest pooled dual-encoder loss among the tested windows:

| Lookback | Pooled dual-encoder loss |
|---:|---:|
| 20 days | 0.1570 |
| 60 days | **0.1521** |
| 120 days | 0.1624 |

These results support the project's main finding: separating volatility and
sentiment into dedicated encoders before cross-attention performs better than
both price-only modeling and naive feature concatenation in the saved
experiments. Detailed horizon/quantile losses, calibration tables, bootstrap
comparisons, and fold-level Diebold-Mariano tests are available under
`reports/tables/cross_validation/`.

## Reporting layout

The reporting structure follows the same pattern as the Boot Camp project:

- `reports/tables/single_split/`: pooled and pairwise single-split results;
- `reports/tables/cross_validation/`: full/quick summaries, aggregate matrices,
  per-asset metrics, and fold-level statistics;
- `reports/figures/single_split/`: attention and calibration figures;
- `reports/figures/cross_validation/`: coverage, loss, per-asset, and event
  attention figures; and
- `artifacts/attention/`: local compressed attention arrays, excluded from Git.

## Data sources

- Baker, Bloom, and Davis, *Measuring Economic Policy Uncertainty*.
- Caldara and Iacoviello, *Measuring Geopolitical Risk*.
- Caldara, Iacoviello, Molligo, Prestipino, and Raffo, *The Economic Effects of
  Trade Policy Uncertainty*.
- Shapiro, Sudhof, and Wilson, *Measuring News Sentiment*; Federal Reserve Bank
  of San Francisco Daily News Sentiment Index.

The repository does not currently document the vendor or download URL for the
5-minute ETF price files. That provenance should be documented separately so
the local dataset can be reproduced without distributing the files here.

## Data availability

No raw, intermediate, or processed data are distributed with this repository.
The entire `data/` directory is excluded from version control. Users must
obtain the source data under the applicable provider terms and reproduce the
local input files before running the experiments.

## Reproducibility notes

- Random seeds are set in the training scripts.
- Neural-network features are standardized using training data only.
- Evaluation is chronological rather than randomly shuffled across time.
- Forecast quality is measured with quantile (pinball) loss and empirical
  coverage.
- Statistical comparisons include block bootstrap and Diebold-Mariano tests.
- Compressed attention arrays (`*.npz`) are generated intermediates and are
  intentionally excluded from version control.

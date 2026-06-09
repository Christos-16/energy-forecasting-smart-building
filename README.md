# A Comparative Analysis of Machine Learning Models for Energy Forecasting in a Commercial Smart Building

## Overview

This repository contains the complete, reproducible code pipeline for a benchmark study of short-term load forecasting (STLF) in a commercial smart building. We compare roughly 18-20 models spanning classical machine learning, deep learning, and time-series foundation models across 10 power-consumption sensors from a real-world IoT deployment. Performance is assessed at multiple forecast horizons using walk-forward (rolling-origin) validation, producing 540 experiments, with rigorous statistical comparison via the Friedman test and Nemenyi post-hoc analysis (critical-difference ranking). The headline finding is that simple, regularized linear and tree-based models (for example, Ridge regression) are competitive with — and often outrank — far heavier deep-learning and foundation models, while requiring orders of magnitude less computation (a Green AI result). True out-of-sample validation on previously unseen future data further characterizes the generalization gap between these model families.

## Dataset

The preprocessed dataset is published on Zenodo:

- **DOI:** [10.5281/zenodo.20272424](https://doi.org/10.5281/zenodo.20272424)

The cleaned dataset comprises **2,367,500 observations** (about 2.4 million) at 5-minute resolution, derived from roughly **39.8 million** raw event records collected over the building deployment.

**The data is not shipped with this repository.** Download the per-sensor files from Zenodo and place them under a top-level `cleaned_data/` directory, one file per sensor:

```
cleaned_data/<sensor>/clean_full.csv
```

All scripts expect this layout and will look for the data there.

## Repository Structure

```
common/                          Shared utilities, model definitions, feature engineering, and helpers
analysis/                        Statistical analysis and result-processing utilities
cleaning/                        Data-cleaning and preprocessing routines
results/                         Headline result CSVs (full per-model outputs are regenerable)

run_upgraded_benchmark.py        Main orchestrator: runs ML + DL + foundation models (20 models)
run_complete_benchmark.py        Full machine-learning / classical benchmark
run_deep_learning_all.py         Deep-learning models (RNN variants, Transformer architectures)
run_foundation_models.py         Foundation models (Chronos, zero-shot)
run_statistical_analysis.py      Friedman test + Nemenyi post-hoc (critical-difference) analysis
validate_all_models.py           End-to-end validation of all configured models

oos_validation.py                Out-of-sample validation (machine-learning models)
oos_validation_lstm.py           Out-of-sample validation (LSTM / deep-learning)
oos_validation_chronos.py        Out-of-sample validation (Chronos foundation model)
generate_oos_comparison_table.py Aggregates the out-of-sample results into a comparison table

generate_paper_figures.py        Generates the main paper figures
generate_oos_figure.py           Generates the out-of-sample comparison figure
generate_feature_importance.py   Generates the feature-importance figure
```

## Installation

Requires **Python 3.13**.

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Caveats

- **Chronos** is pinned to a specific git commit and installed directly from GitHub (via pip from git, not from PyPI). This is already declared in `requirements.txt`, so a standard `pip install -r requirements.txt` will fetch the correct commit.
- **TimesFM** is intentionally omitted. It requires Python < 3.12 and is therefore incompatible with the Python 3.13 environment used here. It is left commented out in `requirements.txt`.

## Reproducing the Results

Run **all scripts from the repository root**.

1. **Download the data.** Fetch the preprocessed dataset from Zenodo ([10.5281/zenodo.20272424](https://doi.org/10.5281/zenodo.20272424)) and place the per-sensor files under `cleaned_data/<sensor>/clean_full.csv`.

2. **Run the main benchmark.** This orchestrates the machine-learning, deep-learning, and foundation models (20 models):

   ```bash
   python run_upgraded_benchmark.py
   ```

3. **Run the statistical analysis.** This performs the Friedman test and Nemenyi post-hoc (critical-difference) comparison:

   ```bash
   python run_statistical_analysis.py
   ```

4. **Run the out-of-sample validation**, then aggregate the comparison table:

   ```bash
   python oos_validation.py
   python oos_validation_lstm.py
   python oos_validation_chronos.py
   python generate_oos_comparison_table.py
   ```

5. **Generate the figures:**

   ```bash
   python generate_paper_figures.py
   python generate_oos_figure.py
   python generate_feature_importance.py
   ```

## Results

The `results/` directory holds the headline result CSVs reported in the paper. The full per-model outputs are not stored in the repository; they are fully regenerable by running the scripts above from the repository root.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). If you use this code, please cite the associated paper:

```bibtex
@article{anastasiou2026comparative,
  title   = {A Comparative Analysis of Machine Learning Models for Energy Forecasting in a Commercial Smart Building},
  author  = {Anastasiou, Christos G. and Hitiris, Christos and Gkola, Cleopatra and Vergados, Dimitrios J. and Pliatsios, Dimitrios and Sarigiannidis, Panagiotis and Michalas, Angelos},
  journal = {IEEE Access},
  year    = {2026},
  note    = {To appear (under review)}
  % DOI to be added upon publication
}
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Data Availability

The preprocessed dataset supporting this study is openly available on Zenodo at [10.5281/zenodo.20272424](https://doi.org/10.5281/zenodo.20272424).

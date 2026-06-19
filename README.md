# Pocket ML Lab

[![CI](https://github.com/get2salam/pocket-ml-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/get2salam/pocket-ml-lab/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Pocket ML Lab** is a lightweight, zero-dependency Python toolkit for running explainable baseline machine-learning experiments on small CSV datasets — with no NumPy, no pandas, no scikit-learn required.

It is designed for:

- Rapid prototyping when you want interpretable baselines before reaching for heavy frameworks
- Teaching ML concepts from scratch (the entire implementation is readable Python)
- Generating structured JSON + Markdown reports you can drop straight into a repo or presentation

---

## Quickstart

```bash
# Install (no extra dependencies)
pip install -e .

# Profile a CSV dataset
pocket-ml profile examples/iris_small.csv --target species

# Run a baseline classification experiment
pocket-ml run examples/iris_small.csv --target species --task classification

# Run a baseline regression experiment
pocket-ml run examples/house_prices_small.csv --target price --task regression

# Write reports to a directory
pocket-ml run examples/iris_small.csv --target species --task classification --out reports/
```

---

## Features

| Feature | Details |
|---|---|
| **CSV loading** | Robust reader with type inference, missing-value detection |
| **Schema profiling** | Numeric/categorical summaries, cardinality, null counts |
| **Deterministic split** | Reproducible train/test split (seed-controlled Fisher-Yates shuffle) |
| **Majority classifier** | Predicts the most-frequent class — your minimum useful baseline |
| **Mean regressor** | Predicts the training-set mean — the simplest regression baseline |
| **Nearest-centroid classifier** | Per-class centroids; classifies by Euclidean distance |
| **Classification metrics** | Accuracy, per-class precision/recall/F1, macro averages, confusion matrix |
| **Regression metrics** | MAE, RMSE, R² |
| **JSON reports** | Machine-readable experiment records |
| **Markdown reports** | Human-readable tables and metric summaries |
| **Zero dependencies** | Standard library only — runs anywhere Python 3.9+ is present |

---

## Example output

```
=== Pocket ML Lab ===
Dataset : examples/iris_small.csv
Task    : classification
Target  : species
Split   : 80 train / 20 test (seed=42)

--- Majority Classifier ---
Accuracy : 0.3333
Macro F1 : 0.1667

--- Nearest-Centroid Classifier ---
Accuracy : 0.9333
Macro F1 : 0.9330
```

---

## Architecture

```
pocket_ml_lab/
├── loader.py        # CSV → list[dict] with type coercion
├── profiler.py      # Schema inference and statistical summaries
├── splitter.py      # Deterministic train/test split
├── metrics/         # Accuracy, F1, MAE, RMSE, R²
├── baselines/       # MajorityClassifier, MeanRegressor, NearestCentroid
├── experiment.py    # Ties loader → split → model → metrics → report
├── reporters/       # JSON and Markdown report writers
└── cli.py           # `pocket-ml` command
```

See [docs/architecture.md](docs/architecture.md) for a deeper walkthrough.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Write tests for any new behaviour
4. Submit a pull request — all CI checks must pass

---

## License

MIT — see [LICENSE](LICENSE).

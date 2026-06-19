# Changelog

All notable changes to Pocket ML Lab are documented here.

---

## [0.1.0] — 2026-06-19

### Added

- **CSV loader** (`loader.py`) with automatic int/float/string coercion and null detection
- **Schema profiler** (`profiler.py`) — numeric summaries (min, max, mean, std, median) and categorical summaries (cardinality, top values)
- **Deterministic train/test split** (`splitter.py`) — seeded Fisher-Yates shuffle using a linear-congruential generator
- **Majority-class classifier** baseline — always predicts the most frequent training class
- **Mean regressor** baseline — always predicts the training-set mean
- **Nearest-centroid classifier** baseline — assigns samples to the class whose feature-space centroid is nearest in Euclidean distance
- **Classification metrics** — accuracy, per-class precision/recall/F1, macro averages, confusion matrix
- **Regression metrics** — MAE, RMSE, R²
- **Experiment runner** (`experiment.py`) — `ExperimentConfig` + `run_experiment()` tying all stages together
- **JSON reporter** — machine-readable experiment records
- **Markdown reporter** — human-readable tables and metric summaries
- **HTML reporter** — self-contained browser-viewable report with inline CSS
- **CLI** (`pocket-ml`) with `profile` and `run` subcommands
- **Example datasets** — `iris_small.csv` (classification, 30 rows) and `house_prices_small.csv` (regression, 25 rows)
- **Full test suite** — 55 tests covering loader, profiler, splitter, baselines, metrics, and CLI
- **GitHub Actions CI** — runs on Python 3.9, 3.10, 3.11, 3.12
- **Makefile** convenience targets

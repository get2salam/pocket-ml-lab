# Pocket ML Lab — Architecture Notes

## Design goals

- **Zero runtime dependencies** — standard library only; works wherever Python 3.9+ is installed
- **Readable implementation** — every algorithm is a clean, self-contained Python file; no magic
- **Deterministic by default** — same seed → same split → same results, every run
- **Composable** — each layer (load → profile → split → model → report) is independently usable

---

## Data flow

```
CSV file
   │
   ▼
loader.load_csv()          → list[dict[str, Any]]   (rows)
   │
   ├─► profiler.profile_dataset()  → schema + statistics dict
   │
   └─► splitter.train_test_split() → (train_rows, test_rows)
            │
            ├─► baselines.MajorityClassifier.fit/predict
            ├─► baselines.NearestCentroidClassifier.fit/predict
            └─► baselines.MeanRegressor.fit/predict
                       │
                       ▼
            metrics.classification_report()   (or regression metrics)
                       │
                       ▼
            reporters.write_json_report()
            reporters.write_markdown_report()
            reporters.write_html_report()
```

---

## Module descriptions

### `loader.py`

Wraps `csv.DictReader` with automatic type coercion:
- `int` → `float` → `str` (first match wins)
- Empty cells become `None`
- Accepts file paths or raw strings (for tests)

### `profiler.py`

Detects column types (`numeric` / `categorical` / `mixed`) by inspecting runtime Python types after coercion — avoids re-parsing raw strings.  
Computes standard descriptive statistics (min, max, mean, sample std, median) for numeric columns, and value counts / cardinality for categorical columns.

### `splitter.py`

Implements **Fisher-Yates shuffle** with a **linear-congruential generator (LCG)** seeded by the user-provided integer.  
LCG parameters match glibc (`a=1103515245, c=12345, m=2³¹`) — well-tested for uniformity in shuffles of small arrays.  
The split is index-based, so the original rows are never mutated.

### `baselines/`

Each model follows a minimal sklearn-like interface (`fit(X, y)` → `predict(X)`) but is implemented entirely from scratch:

| Model | Algorithm |
|---|---|
| `MajorityClassifier` | Count labels; predict argmax class |
| `MeanRegressor` | Compute mean; return scalar repeated N times |
| `NearestCentroidClassifier` | Average feature vectors per class; predict by min Euclidean distance |

All models implement `describe()` returning a JSON-serialisable dict for embedding in reports.

### `metrics/`

**Classification** — pure Python contingency-table arithmetic:
- `confusion_matrix` → indexed by sorted class labels
- `per_class_metrics` → TP/FP/FN computed from the confusion matrix rows/columns
- `classification_report` → wraps per-class metrics with macro averages

**Regression** — closed-form:
- MAE: `mean(|y - ŷ|)`
- RMSE: `sqrt(mean((y - ŷ)²))`
- R²: `1 - SS_res / SS_tot`

### `experiment.py`

`ExperimentConfig` is a plain dataclass-like class (no dataclass decorator for 3.9 compat) that validates and stores experiment parameters.

`run_experiment(config)` orchestrates the full pipeline and returns a single nested dict — this dict is then passed to any reporter without further transformation.

### `reporters/`

- `json_reporter` — thin wrapper around `json.dumps` with a custom fallback for `set`/`frozenset`
- `markdown_reporter` — string-building loop; emits GitHub-flavoured Markdown tables
- HTML output wraps the Markdown in a self-contained page with inline CSS (no external CDN)

### `cli.py`

Built on `argparse` with two subcommands:
- `profile` — loads and profiles; no model is run
- `run` — full pipeline; optionally writes all three report formats

Exit codes follow Unix convention: `0` = success, `1` = user error (bad path, bad column name, etc.).

---

## Extending the toolkit

### Adding a new baseline model

1. Create `pocket_ml_lab/baselines/my_model.py` with `fit(X, y)`, `predict(X)`, and `describe()` methods.
2. Import it in `pocket_ml_lab/baselines/__init__.py`.
3. Add it to the model list in `experiment.py` under the appropriate task branch.
4. Write tests in `tests/test_baselines.py`.

### Adding a new metric

1. Add a function to `pocket_ml_lab/metrics/classification.py` or `regression.py`.
2. Export it from `pocket_ml_lab/metrics/__init__.py`.
3. Plug it into `experiment.py` and the relevant reporter.

### Supporting a new output format

1. Create `pocket_ml_lab/reporters/csv_reporter.py` (for example).
2. Export from `pocket_ml_lab/reporters/__init__.py`.
3. Add a `--format` flag to `cli.py`.

---

## Testing philosophy

- All tests use only the public API (`pocket_ml_lab.*`); no internal mocking.
- The test suite never reads from disk except for the bundled example CSVs.
- `load_csv_string` allows inline fixture data without temporary files.
- Determinism tests (`test_split_determinism`) guard against accidental PRNG state pollution.

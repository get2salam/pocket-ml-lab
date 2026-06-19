"""Experiment runner: ties loader → split → model(s) → metrics → results dict."""

from typing import Any

from .loader import load_csv, column_names
from .profiler import profile_dataset
from .splitter import train_test_split
from .baselines import MajorityClassifier, MeanRegressor, NearestCentroidClassifier
from .metrics.classification import classification_report
from .metrics.regression import mae, rmse, r2_score


class ExperimentConfig:
    """Holds all parameters for a single experiment run."""

    def __init__(
        self,
        csv_path: str,
        target: str,
        task: str = "classification",
        test_size: float = 0.2,
        seed: int = 42,
        feature_cols: list[str] | None = None,
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"Unknown task '{task}'. Choose 'classification' for categorical targets "
                f"or 'regression' for numeric targets."
            )
        self.csv_path = csv_path
        self.target = target
        self.task = task
        self.test_size = test_size
        self.seed = seed
        self.feature_cols = feature_cols

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv_path": self.csv_path,
            "target": self.target,
            "task": self.task,
            "test_size": self.test_size,
            "seed": self.seed,
            "feature_cols": self.feature_cols,
        }


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute a full experiment and return a structured results dictionary."""
    rows = load_csv(config.csv_path)

    all_cols = column_names(rows)
    if config.target not in all_cols:
        raise ValueError(
            f"Target column '{config.target}' not found in dataset. "
            f"Available columns: {all_cols}"
        )

    feature_cols = config.feature_cols or [c for c in all_cols if c != config.target]

    profile = profile_dataset(rows)
    train_rows, test_rows = train_test_split(rows, config.test_size, config.seed)

    X_train = [{k: row[k] for k in feature_cols if k in row} for row in train_rows]
    y_train = [row[config.target] for row in train_rows]
    X_test = [{k: row[k] for k in feature_cols if k in row} for row in test_rows]
    y_test = [row[config.target] for row in test_rows]

    model_results = []

    if config.task == "classification":
        models = [
            MajorityClassifier(),
            NearestCentroidClassifier(),
        ]
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred)
            model_results.append({
                "model": model.name,
                "description": model.describe(),
                "metrics": report,
            })

    else:  # regression
        y_train_num = [float(v) for v in y_train if isinstance(v, (int, float))]
        y_test_num = [float(v) for v in y_test if isinstance(v, (int, float))]
        models_reg = [MeanRegressor()]
        for model in models_reg:
            model.fit(X_train, y_train_num)
            y_pred = model.predict(X_test)
            model_results.append({
                "model": model.name,
                "description": model.describe(),
                "metrics": {
                    "mae": round(mae(y_test_num, y_pred), 6),
                    "rmse": round(rmse(y_test_num, y_pred), 6),
                    "r2": round(r2_score(y_test_num, y_pred), 6),
                },
            })

    return {
        "config": config.to_dict(),
        "dataset": {
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "n_train": len(train_rows),
            "n_test": len(test_rows),
            "target": config.target,
            "features": feature_cols,
            "profile": profile,
        },
        "results": model_results,
    }

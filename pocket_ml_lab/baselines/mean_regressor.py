"""Mean regressor — predicts the training-set mean for all samples."""

from typing import Any


class MeanRegressor:
    """Baseline regressor that always predicts the mean of the training targets.

    Equivalent to the model that minimises squared error with no features.
    Any real regressor must achieve lower RMSE than this.
    """

    name = "MeanRegressor"

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X: list[dict[str, Any]], y: list[float]) -> "MeanRegressor":
        numeric = [v for v in y if isinstance(v, (int, float))]
        if not numeric:
            raise ValueError("Target column contains no numeric values")
        self._mean = sum(numeric) / len(numeric)
        return self

    def predict(self, X: list[dict[str, Any]]) -> list[float]:
        if self._mean is None:
            raise RuntimeError("Call fit() before predict()")
        return [self._mean] * len(X)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "training_mean": self._mean}

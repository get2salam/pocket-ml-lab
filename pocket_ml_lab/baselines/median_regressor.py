"""Median regressor — predicts the training-set median for all samples."""

from __future__ import annotations

from typing import Any


class MedianRegressor:
    """Baseline regressor that always predicts the median of the training targets.

    The median is insensitive to outliers, unlike the mean.  When a target
    distribution is skewed (e.g. house prices, salary), the median baseline
    provides a more representative central-tendency prediction and exposes
    whether the mean is inflated by extreme values.

    Compare MedianRegressor against MeanRegressor: if their RMSE values
    diverge significantly, the target likely contains influential outliers.
    """

    name = "MedianRegressor"

    def __init__(self) -> None:
        self._median: float | None = None

    def fit(self, X: list[dict[str, Any]], y: list[float]) -> "MedianRegressor":
        numeric = sorted(float(v) for v in y if isinstance(v, (int, float)))
        if not numeric:
            raise ValueError("Target column contains no numeric values")
        n = len(numeric)
        mid = n // 2
        self._median = numeric[mid] if n % 2 == 1 else (numeric[mid - 1] + numeric[mid]) / 2.0
        return self

    def predict(self, X: list[dict[str, Any]]) -> list[float]:
        if self._median is None:
            raise RuntimeError("Call fit() before predict()")
        return [self._median] * len(X)

    def describe(self) -> dict[str, Any]:
        return {"model": self.name, "training_median": self._median}

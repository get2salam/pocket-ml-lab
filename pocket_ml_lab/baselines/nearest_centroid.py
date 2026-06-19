"""Nearest-centroid classifier — classify by Euclidean distance to per-class centroids."""

import math
from typing import Any


class NearestCentroidClassifier:
    """Baseline classifier that computes a centroid per class from numeric features.

    At prediction time each sample is assigned to the class whose centroid
    is nearest in Euclidean space.  Non-numeric feature columns are ignored.
    """

    name = "NearestCentroidClassifier"

    def __init__(self) -> None:
        self._centroids: dict[Any, list[float]] = {}
        self._features: list[str] = []

    def fit(
        self,
        X: list[dict[str, Any]],
        y: list[Any],
        feature_cols: list[str] | None = None,
    ) -> "NearestCentroidClassifier":
        if not X or not y:
            raise ValueError("X and y must be non-empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        # Discover numeric feature columns
        candidate_cols = feature_cols if feature_cols else list(X[0].keys())
        self._features = [
            col for col in candidate_cols
            if any(isinstance(row.get(col), (int, float)) for row in X)
        ]
        if not self._features:
            raise ValueError("No numeric feature columns found — cannot compute centroids")

        # Accumulate per-class sums and counts
        sums: dict[Any, list[float]] = {}
        counts: dict[Any, int] = {}
        for row, label in zip(X, y):
            if label not in sums:
                sums[label] = [0.0] * len(self._features)
                counts[label] = 0
            for i, col in enumerate(self._features):
                val = row.get(col)
                sums[label][i] += float(val) if isinstance(val, (int, float)) else 0.0
            counts[label] += 1

        self._centroids = {
            label: [s / counts[label] for s in sums[label]]
            for label in sums
        }
        return self

    def predict(self, X: list[dict[str, Any]]) -> list[Any]:
        if not self._centroids:
            raise RuntimeError("Call fit() before predict()")
        predictions = []
        for row in X:
            vec = [
                float(row.get(col)) if isinstance(row.get(col), (int, float)) else 0.0
                for col in self._features
            ]
            nearest = min(
                self._centroids,
                key=lambda lbl: _euclidean(vec, self._centroids[lbl]),
            )
            predictions.append(nearest)
        return predictions

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "features": self._features,
            "centroids": {str(k): v for k, v in self._centroids.items()},
        }


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
